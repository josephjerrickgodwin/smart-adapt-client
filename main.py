import logging
import sys

logging.basicConfig(stream=sys.stdout, level="INFO")
log = logging.getLogger(__name__)
log.setLevel("INFO")

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.controller.health_controller import router as health_router
from src.controller.fine_tuning_controller import router as fine_tuning_router
from src.controller.inference_controller import router as inference_router

app = FastAPI()

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(fine_tuning_router)
app.include_router(inference_router)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="localhost",
        port=8080,
        reload=False,
        log_level="debug"
    )
