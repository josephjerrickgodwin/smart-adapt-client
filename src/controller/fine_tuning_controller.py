import io
import logging
import os
import shutil

import pandas as pd

from fastapi import APIRouter, HTTPException, status, UploadFile, File

from src.exception.fine_tuning_disabled_error import FineTuningDisabledError
from src.service.model_service import model_service
from src.service.storage_manager import storage_manager

log = logging.getLogger(__name__)
log.setLevel("INFO")

router = APIRouter(prefix="/api/v1", tags=["Fine Tuning"])


@router.post("/fine-tune")
async def fine_tune(
        user_id: str,
        knowledge_id: str,
        question_column_name: str,
        answer_column_name: str,
        file: UploadFile = File(...)
):
    try:
        # Read file content into BytesIO
        content = await file.read()
        byte_stream = io.BytesIO(content)

        # Convert CSV to Pandas DataFrame
        df = pd.read_csv(byte_stream)

        await model_service.fine_tuning_handler(
            df=df,
            user_id=user_id,
            knowledge_id=knowledge_id,
            question_column_name=question_column_name,
            answer_column_name=answer_column_name
        )

        return {"status": "Successful"}

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


@router.delete("/adapter")
async def delete_lora_adapter(
        user_id: str,
        knowledge_id: str
):
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

    except Exception as e:
        log.error(f"LoRA adapter removal error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
