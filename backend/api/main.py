import logging, os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import analyze, explain
from api.schemas import HealthResponse, HealthModel
from api.services import bert_service, xgb_service

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 ChildGuard AI v2.0 başlatılıyor...")
    bert_service.load_all_models()
    xgb_service.load_model()
    yield
    logger.info("🛑 ChildGuard AI kapatılıyor.")


app = FastAPI(title="ChildGuard AI v2.0", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS",
                            "http://localhost:3000,http://127.0.0.1:3000").split(","),
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

app.include_router(analyze.router, prefix="/api", tags=["Analiz"])
app.include_router(explain.router, prefix="/api", tags=["XAI"])


@app.get("/api/health", response_model=HealthResponse, tags=["Sistem"])
async def health():
    models = [
        HealthModel(name="BERT — Genel",     loaded=bert_service.is_loaded("bert_general"),   path=bert_service.model_path("bert_general")),
        HealthModel(name="BERT — Yaş Grubu", loaded=bert_service.is_loaded("bert_age_group"), path=bert_service.model_path("bert_age_group")),
        HealthModel(name="XGBoost",          loaded=xgb_service.is_loaded(),                  path=xgb_service.model_path()),
    ]
    return HealthResponse(status="ok" if all(m.loaded for m in models) else "degraded",
                          models=models)


@app.get("/", tags=["Sistem"])
async def root():
    return {"message": "ChildGuard AI v2.0 çalışıyor.", "docs": "/docs", "health": "/api/health"}
