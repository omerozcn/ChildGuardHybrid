import logging
from fastapi import APIRouter, HTTPException
from api.schemas import ExplainRequest, LimeRequest, ShapResponse, LimeResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/explain/shap", response_model=ShapResponse)
async def explain_shap(req: ExplainRequest) -> ShapResponse:
    try:
        from api.services import shap_service
        return ShapResponse(features=shap_service.explain(req.text, req.age_group))
    except ImportError:
        raise HTTPException(status_code=501, detail="SHAP servisi henüz aktif değil.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/explain/lime", response_model=LimeResponse)
async def explain_lime(req: LimeRequest) -> LimeResponse:
    try:
        from api.services import lime_service
        return LimeResponse(features=lime_service.explain(req.text, req.age_group, req.num_samples))
    except ImportError:
        raise HTTPException(status_code=501, detail="LIME servisi henüz aktif değil.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
