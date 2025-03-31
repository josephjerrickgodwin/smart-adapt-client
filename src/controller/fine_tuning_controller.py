import io
import logging
import os
import shutil
import uuid

import pandas as pd
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form, BackgroundTasks

from src.exception.fine_tuning_disabled_error import FineTuningDisabledError
from src.model.user_knowledge_model import UserKnowledgeModel
from src.service.database.knowledge import KnowledgeForm, Knowledge_table
from src.service.model_service import model_service
from src.service.storage_manager import storage_manager

log = logging.getLogger(__name__)
log.setLevel("INFO")

router = APIRouter(prefix="/api/v1", tags=["Fine Tuning"])


@router.post("/validate", status_code=status.HTTP_200_OK)
async def validate_user_knowledge(data: UserKnowledgeModel):
    try:
        user_role = data.user_role
        knowledge_ids = data.knowledge_ids

        if user_role == "admin":
            knowledge_bases = Knowledge_table.get_knowledge_bases()
        else:
            knowledge_bases = []
            for knowledge_id in knowledge_ids:
                knowledge = Knowledge_table.get_knowledge_by_id(knowledge_id)
                knowledge_bases.append(knowledge)

        return {
            "status": "Completed",
            "knowledge_bases": knowledge_bases
        }

    except Exception as e:
        log.error(f"User knowledge validate error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/fine-tune", status_code=status.HTTP_200_OK)
async def fine_tune(
        background_task: BackgroundTasks,
        user_id: str = Form(...),
        knowledge_id: str = Form(...),
        question_column_name: str = Form(...),
        answer_column_name: str = Form(...),
        file: UploadFile = File(...)
):
    try:
        # Read file content into BytesIO
        content = await file.read()
        byte_stream = io.BytesIO(content)

        # Convert CSV to Pandas DataFrame
        df = pd.read_csv(byte_stream)

        # Add file information
        file_data = {
            "filename": file.filename,
            "size": file.size,
            "status": "In Progress"
        }

        # Create the knowledge record
        knowledge_form = KnowledgeForm(
            name=str(uuid.uuid4()),
            description='Fine-tuning record',
            data=file_data,
            access_control=None
        )
        knowledge = Knowledge_table.insert_new_knowledge(
            user_id=user_id,
            knowledge_id=knowledge_id,
            form_data=knowledge_form
        )

        # Add fine-tuning as a background_task
        background_task.add_task(
            model_service.fine_tuning_handler,
            df=df,
            user_id=user_id,
            knowledge_id=knowledge_id,
            question_column_name=question_column_name,
            answer_column_name=answer_column_name,
            file_data=file_data
        )

        return {
            "id": knowledge.id,
            "status": "successful"
        }

    except FineTuningDisabledError as e:
        log.error(f"Fine-tuning error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    except ValueError as e:
        log.error(f"Fine-tuning error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    except Exception as e:
        log.error(f"Fine-tuning error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/adapter", status_code=status.HTTP_200_OK)
async def delete_lora_adapter(user_id: str, knowledge_id: str):
    try:
        # Check for model Knowledge
        userdata_dir = await storage_manager.get_user_dir(user_id)
        lora_path = model_service.get_lora_path(
            data_path=userdata_dir,
            knowledge_id=knowledge_id
        )
        logs_path = model_service.get_logs_path(
            data_path=userdata_dir,
            knowledge_id=knowledge_id
        )

        # Remove knowledge, if found
        if os.path.exists(lora_path) and len(os.listdir(lora_path)):
            shutil.rmtree(lora_path)
        if os.path.exists(logs_path) and len(os.listdir(logs_path)):
            shutil.rmtree(logs_path)

        # Remove the knowledge record
        knowledge = Knowledge_table.get_knowledge_by_id(id=knowledge_id)
        if knowledge is not None:
            _ = Knowledge_table.delete_knowledge_by_id(id=knowledge_id)

        return {"status": "successful"}

    except Exception as e:
        log.error(f"LoRA adapter removal error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
