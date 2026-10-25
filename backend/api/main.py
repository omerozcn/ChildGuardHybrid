import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import analyze, explain
from api.schemas import HealthResponse
from api.services import model_service

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 ChildGuard AI v4 başlatılıyor...")
    model_service.load()
    yield
    logger.info("🛑 ChildGuard AI kapatılıyor.")


app = FastAPI(title="ChildGuard AI", version="4.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS",
                            "http://localhost:3000,http://127.0.0.1:3000").split(","),
    allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"],
)

app.include_router(analyze.router, prefix="/api", tags=["Analiz"])
app.include_router(explain.router, prefix="/api", tags=["Açıklama"])


@app.get("/api/health", response_model=HealthResponse, tags=["Sistem"])
def health() -> HealthResponse:
    model = model_service.get()
    if model is None:
        return HealthResponse(status="degraded", error=model_service.error())
    return HealthResponse(
        status="ok",
        model_version=model.version,
        base_model=model.config.get("base_model"),
        uses_transformer=model.transformer is not None,
        device=model.transformer.device if model.transformer else "cpu",
    )


@app.get("/", tags=["Sistem"])
def root():
    return {"message": "ChildGuard AI v4 çalışıyor.", "docs": "/docs", "health": "/api/health"}
