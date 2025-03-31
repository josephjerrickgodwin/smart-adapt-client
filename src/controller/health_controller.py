from fastapi import APIRouter, status

router = APIRouter(tags=["Health"])


@router.post("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "Server is up and running!"}
