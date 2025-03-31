import logging

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from src.exception.inference_disabled_error import InferenceDisabledError
from src.model.completions_model import CompletionRequest
from src.service.model_service import model_service

log = logging.getLogger(__name__)
log.setLevel("INFO")

router = APIRouter(prefix="/api/v1", tags=["Chat Completions"])


@router.post("/rewrite", status_code=status.HTTP_200_OK)
async def rewrite_query(query: str, history: list):
    try:
        enhanced_query = model_service.rewrite_query(
            chat_query=query,
            history=history
        )
        return {"query": enhanced_query}

    except Exception as e:
        log.error(f"Query rewrite error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/completions", status_code=status.HTTP_200_OK)
async def start_completion(data: CompletionRequest):
    user_id = data.user_id
    messages = data.messages
    stream = data.stream
    knowledge_ids = data.knowledge_ids or []
    additional_kwargs = data.additional_kwargs or {}

    async def stream_content():
        async for chunk in model_service.start_completions(
                user_id=user_id,
                messages=messages,
                stream=stream,
                knowledge_ids=knowledge_ids,
                **additional_kwargs
        ):
            if chunk is not None:
                yield chunk
    try:
        return StreamingResponse(
            stream_content(),
            media_type="text/event-stream"
        )
    except InferenceDisabledError as e:
        log.error(f'Inference disabled error: {str(e)}')
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        log.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
