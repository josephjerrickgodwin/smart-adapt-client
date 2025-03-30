from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.post("/health")
async def health_check():
    return {"status": "Server is up and running!"}
