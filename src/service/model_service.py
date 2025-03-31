import gc
import logging
import os
import re
from threading import Thread
from typing import Any

import ftfy
import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from trl import SFTTrainer, SFTConfig

from src.exception.fine_tuning_disabled_error import FineTuningDisabledError
from src.exception.inference_disabled_error import InferenceDisabledError
from src.service import prompt_service
from src.service.database.knowledge import Knowledge_table
from src.service.storage_manager import storage_manager
from src.service.smart_adapt_streamer import SmartAdaptStreamer

load_dotenv()

logger = logging.getLogger(__name__)

# Load the Sentence Transformer Model and the HF token
MODEL_ID = str(os.getenv('MODEL_ID'))
HF_TOKEN = str(os.getenv('HF_TOKEN'))


class ModelService:
    """
    Main service for LLM inference.
    """

    def __init__(self):
        self.token = HF_TOKEN
        self.data_model_name = 'adapters'
        self.logs_dir = 'logs'

        # Define model and tokenizer for reasoning
        self.tokenizer = None
        self.model = None

        # Define the device
        if torch.cuda.is_available():
            current_device_index = torch.cuda.current_device()
            self.device = torch.device(f"cuda:{current_device_index}")
        else:
            self.device = torch.device("cpu")

        # Define the current user
        self.current_user = None
        self.lora_loaded = False
        self.inference_enabled = True
        self.loaded_lora_ids = []

        # Define repetition penalty for completions
        self.repetition_penalty = 1.2

        # Load weights
        self._load_weights()

    def flush(self):
        # Disable inference temporarily
        self.inference_enabled = False

        # Unload the thinker and inference models along with the tokenizers
        del self.tokenizer
        del self.model

        # Reset adapter and user details
        self.current_user = None
        self.lora_loaded = False

        # Clear cache
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        # Load the model again
        self._load_weights()

        # Reset the inference status
        self.inference_enabled = True
        self.loaded_lora_ids = []

    @classmethod
    def calculate_lora_hyperparameters(cls, dataset_size: int):
        """
        Calculate optimal LoRA hyperparameters: rank (r) and scaling factor (lora_alpha).

        Parameters:
        - dataset_size (int): Number of samples in the dataset.

        Returns:
        - dict: A dictionary containing the optimal values for 'r' and 'lora_alpha'.

        Raises:
        - ValueError: If input parameters are invalid.
        """
        # Input validation
        if not dataset_size:
            raise ValueError("Model size and dataset size must be positive")

        # Calculate r and alpha based on the dataset size
        if dataset_size <= 1000:
            r = 8
            lora_alpha = 16
        elif dataset_size <= 100000:
            r = 32
            lora_alpha = 64
        else:
            r = 64
            lora_alpha = 128

        return {'r': r, 'lora_alpha': lora_alpha}

    def get_lora_path(self, data_path: str, knowledge_id: str):
        return os.path.join(data_path, knowledge_id, self.data_model_name)

    def get_logs_path(self, data_path: str, knowledge_id: str):
        return os.path.join(data_path, knowledge_id, self.logs_dir)

    @classmethod
    def _extract_new_query(cls, text: str):
        """
        Extracts text between <new_query> and </new_query> tags.

        Parameters:
            text (str): The input string containing the tags.

        Returns:
            str: Extracted text or None if no match is found.
        """
        # Regex pattern with grouping to capture the text between the tags
        pattern = r"<new_query>(.*?)</new_query>"

        # Search for the pattern and extract the first group if found
        match = re.search(pattern, text)

        # Return the captured group or the original text if no match
        return match.group(1) if match else text

    def _load_adapters(
            self,
            user_id: str,
            knowledge_ids=None,
            merge_and_unload: bool = False
    ):
        """
        Get the inference pipeline for the model.

        Args:
            user_id (str): Unique ID of the user
            knowledge_ids (List[str], optional): List of knowledge ids
            merge_and_unload (bool, optional): Whether to merge the model and unload the LoRA adapters

        Returns:
            text-generation pipeline
        """
        # Check for the current user
        if knowledge_ids is None:
            knowledge_ids = []
        if not self.current_user:
            self.current_user = user_id

        logger.info('Started checking for LoRA adapters...')

        # If the user adapter is already exist, skip loading it again
        if self.lora_loaded:
            if user_id == self.current_user and knowledge_ids == self.loaded_lora_ids:
                logger.info('LoRA adapter already loaded. Skipping auto assign')
                return
            else:
                logger.info('Removing LoRA adapter of the previous user')
                self.flush()

        # Retrieve the user data path
        data_path = storage_manager.get_user_dir(user_id)

        # Load and merge multiple LoRA adapters
        for knowledge_id in knowledge_ids:
            if knowledge_id in self.loaded_lora_ids:
                continue

            lora_path = self.get_lora_path(
                data_path=data_path,
                knowledge_id=knowledge_id
            )
            if not len(os.listdir(lora_path)):
                continue

            # Load the LoRA adapter
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_path,
                adapter_name=knowledge_id
            ).to(self.device)

            self.lora_loaded = True
            self.loaded_lora_ids.append(knowledge_id)

        # Merge the adapter permanently
        if self.lora_loaded and merge_and_unload:
            self.model = self.model.merge_and_unload()

        # Assign the current user
        self.current_user = user_id

    async def rewrite_query(
            self,
            current_query: str,
            history: list
    ):
        logger.info('Started extracting user messages from the history...')

        # Extract previous queries
        previous_queries_list = [
            f"- '{conversation.get('content', '')}'"
            for conversation in history
            if conversation.get('role', '') == 'user'
        ]
        previous_user_queries = '\n'.join(previous_queries_list)

        # Add the current query below the user previous user queries
        prompt = f'Previous Queries:\n{previous_user_queries}\nCurrent Query: *** {current_query} ***\nOutput: '

        logger.info('Preparing messages for prompting the LLM...')

        # Create a new history with the system prompt
        system_prompt = prompt_service.query_rewrite_system_instruction.strip()
        updated_history = [
            {
                "role": "user",
                "content": system_prompt + prompt
            }
        ]

        # Apply chat template
        updated_messages = self.tokenizer.apply_chat_template(updated_history, tokenize=False)

        # Tokenize the messages
        tokens = self.tokenizer(
            updated_messages,
            add_special_tokens=True,
            return_tensors="pt"
        ).to(self.device)

        # Define the streamer
        streamer = SmartAdaptStreamer(
            tokenizer=self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # Define the generation kwargs
        thinker_kwargs = dict(
            input_ids=tokens.input_ids,
            max_new_tokens=512,
            repetition_penalty=self.repetition_penalty,
            streamer=streamer
        )
        logger.info('Started generating the updated query...')

        # Define the thread
        thread = Thread(target=self.model.generate, kwargs=thinker_kwargs)

        # Start the thread
        thread.start()

        # Start streaming the reasoning
        text = ''
        async for chunk in streamer:
            if not chunk:
                continue
            text += chunk

        # Extract the new query
        new_query = self._extract_new_query(text)

        logger.info('Successfully generated the updated query.')

        return new_query

    async def start_completions(
            self,
            user_id: str,
            messages: list,
            stream: bool,
            knowledge_ids: list,
            **params,
    ):
        if not self.inference_enabled:
            raise InferenceDisabledError("The model is being trained. Please try again later!")

        # Load user adapter
        self._load_adapters(user_id, knowledge_ids)

        # Define the streamer
        logger.info('Initializing the tokenizer properties')
        streamer = SmartAdaptStreamer(
            tokenizer=self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # Apply chat template
        logger.info('Converting messages into tokens')
        updated_messages = self.tokenizer.apply_chat_template(messages, tokenize=False)

        # Tokenize the messages
        tokens = self.tokenizer(
            updated_messages,
            add_special_tokens=True,
            return_tensors="pt"
        ).to(self.device)

        logger.info('Preparing the additional kwargs for inference')

        # Define the generation kwargs
        kwargs = dict(
            input_ids=tokens.input_ids,
            max_new_tokens=2048,
            repetition_penalty=self.repetition_penalty,
            streamer=streamer,
            **params
        )

        logger.info('Started the response generation')

        # Define the thread
        thread = Thread(target=self.model.generate, kwargs=kwargs)

        # Start the thread
        thread.start()

        # Start streaming the reasoning
        if stream:
            async for chunk in streamer:
                yield chunk
        else:
            chunks = ''
            async for chunk in streamer:
                if chunk is not None:
                    chunks += chunk
            yield chunks

        if self.lora_loaded:
            self.flush()

        logger.info('Request have been served successfully')

    def _load_weights(self):
        """
        Loads the `thinker` and `inference` model instances
        """
        # QLoRA config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )

        # Load the tokenizer
        logger.info('Started loading the tokenizer')
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            token=self.token
        )

        # Load the model
        logger.info('Started loading the model')
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            token=self.token,
            device_map=self.device,
            quantization_config=bnb_config
        )

        # Set padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def fine_tuning_handler(
            self,
            df: pd.DataFrame,
            user_id: str,
            knowledge_id: str,
            question_column_name: str,
            answer_column_name: str,
            file_data: dict
    ):
        try:
            # Pre-process the data
            documents = []
            for index, row in df.iterrows():
                question = row[question_column_name]
                answer = row[answer_column_name]

                # Skip any empty rows
                if not question or not answer:
                    continue

                # Fix any inconsistencies
                question = ftfy.fix_text(question)
                answer = ftfy.fix_text(answer)

                # Formulate the history
                history = [
                    {
                        "role": "user",
                        "content": question
                    },
                    {
                        "role": "assistant",
                        "content": answer
                    }
                ]

                # Apply the chat template
                formatted_history = self.tokenizer.apply_chat_template(history, tokenize=False)

                # Tokenize the text
                tokenized_data = self.tokenizer(
                    formatted_history,
                    padding="max_length",
                    truncation=True,
                    max_length=512
                )

                # Create labels (shifted input IDs for causal language modeling)
                tokenized_data["labels"] = tokenized_data["input_ids"].copy()

                # Ignore padding tokens for loss calculation
                tokenized_data["labels"][tokenized_data["labels"] == self.tokenizer.pad_token_id] = -100
                documents.append(tokenized_data)

            # Remove the previous dataframe
            del df
            gc.collect()

            # Check for documents are present
            if not documents:
                raise ValueError("No documents")

            # Convert to DataFrame
            df = pd.DataFrame(documents)

            # Split for train, eval
            train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

            # Convert to dataset supportable format
            train_df = Dataset.from_pandas(train_df)
            eval_df = Dataset.from_pandas(eval_df)

            # Calculate LoRA Hyperparameters
            hyperparameters = self.calculate_lora_hyperparameters(len(df))

            # Start training the model
            self.fine_tune(
                user_id=user_id,
                train_df=train_df,
                eval_df=eval_df,
                knowledge_id=knowledge_id,
                r=hyperparameters['r'],
                lora_alpha=hyperparameters['lora_alpha']
            )

            # Update the knowledge status
            file_data['status'] = 'Completed'
            _ = Knowledge_table.update_knowledge_data_by_id(
                id=knowledge_id,
                data=file_data
            )

        except Exception as e:
            logger.error(f"Fine-tuning error: {e}")

            # Update the knowledge status
            file_data['status'] = 'Failed'
            _ = Knowledge_table.update_knowledge_data_by_id(
                id=knowledge_id,
                data=file_data
            )

    def prepare_model(self, r: int, lora_alpha: int):
        """
            Prepare the base model for fine-tuning.

            Args:
                r (int): LoRA rank
                lora_alpha (int): LoRA scaling factor

            Returns:
                Tuple of prepared model and tokenizer
        """
        peft_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                'q_proj', 'k_proj', 'v_proj',
                'o_proj', 'gate_proj', 'up_proj', 'down_proj'
            ]
        )

        # Apply LoRA to the model
        self.model = get_peft_model(self.model, peft_config).to(self.device)

    def fine_tune(
            self,
            user_id: str,
            train_df: Any,
            eval_df: Any,
            r: int,
            lora_alpha: int,
            knowledge_id: str,
            fp16: bool = True,
            bf16: bool = False,
            fp16_full_eval: bool = True,
            num_epochs: int = 3,
            eval_steps: int = 100,
            optim: str = "adamw_bnb_8bit",
            logging_steps: int = 10,
            warmup_steps: int = 500,
            max_seq_len: int = 512,
            group_by_length: bool = False,
            learning_rate: float = 2e-4,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            report_to: str = 'none'
    ):
        """
        Fine-tune the model with the given dataset.

        Args:
            user_id (str): Unique ID of the user
            train_df (Any): Training dataset
            eval_df (Any): Evaluation dataset
            r (int): LoRA rank
            lora_alpha (int): LoRA scaling factor
            knowledge_id (str): Unique ID of the knowledge request
            fp16 (bool): Whether to use float16
            bf16 (bool): Whether to use bfloat16
            fp16_full_eval (bool): Whether to perform evaluation in float16
            num_epochs (int): Number of training epochs
            max_seq_len (int): Maximum sequence length from the dataset
            group_by_length (int): Whether to group the dataset by the length of the longest sequence
            eval_steps (int): Number of evaluation steps
            optim (str): Optimizer name
            logging_steps (int): Number of logging steps
            warmup_steps (int): Number of warmup steps
            learning_rate (float): Learning rate for training
            per_device_train_batch_size (int): Training batch size per device
            per_device_eval_batch_size (int): Evaluation batch size per device
            gradient_accumulation_steps (int): Gradient accumulation steps
            report_to (str): Report the training and evaluation results

        Yields:
            str: Progress updates during fine-tuning
        """
        if not self.inference_enabled:
            raise FineTuningDisabledError('A fine-tuning task is in progress!')

        # Disable inference during training
        self.inference_enabled = False

        # Unload and reset weights
        self.flush()

        # Prepare the data storage
        data_path = storage_manager.get_user_dir(user_id)

        # Define the adapter and logs path
        adapter_path = self.get_lora_path(
            data_path=data_path,
            knowledge_id=knowledge_id
        )
        logs_dir_path = self.get_logs_path(
            data_path=data_path,
            knowledge_id=knowledge_id
        )

        # Prepare model and tokenizer
        self.prepare_model(
            r=r,
            lora_alpha=lora_alpha
        )

        # Training arguments
        training_arguments = SFTConfig(
            output_dir=logs_dir_path,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            num_train_epochs=num_epochs,
            eval_strategy="steps",
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            warmup_steps=warmup_steps,
            logging_strategy="steps",
            learning_rate=learning_rate,
            fp16=fp16,
            bf16=bf16,
            fp16_full_eval=fp16_full_eval,
            group_by_length=group_by_length,
            max_seq_length=max_seq_len,
            report_to=report_to
        )

        # Initialize trainer
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_df,
            eval_dataset=eval_df,
            peft_config=None,  # Already applied
            tokenizer=self.tokenizer,
            args=training_arguments,
        )

        # Perform training
        trainer.train()

        # Save model
        os.makedirs(adapter_path, exist_ok=True)
        trainer.model.save_pretrained(adapter_path)

        # Reset the memory and weights
        self.flush()


model_service = ModelService()
