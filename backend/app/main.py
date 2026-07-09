from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.health import router as health_router
from backend.app.api.predict import router as predict_router
from backend.app.ml.loader import load_inference_assets
from backend.app.services.telemetry import initialize_telemetry
from backend.app.api.feedback import router as feedback_router
from backend.app.services.feedback import initialize_feedback_storage
from backend.app.api.monitoring import router as monitoring_router
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.assets = load_inference_assets()
    initialize_telemetry()
    initialize_feedback_storage()
    yield

app = FastAPI(
    title="AksharaVision API",
    version="3.0.0",
    description=(
        "Kannada handwritten-character OCR with calibrated inference, "
        "quality routing, telemetry, and explainability."
    ),
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(health_router, prefix="/api")
app.include_router(predict_router, prefix="/api")
app.include_router(feedback_router, prefix="/api")
app.include_router(monitoring_router, prefix="/api")