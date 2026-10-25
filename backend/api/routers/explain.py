from fastapi import APIRouter, HTTPException

from api.routers.analyze import _model
from api.schemas import ExplainRequest, ExplainResponse
from childguard.labels import LABELS

router = APIRouter()


@router.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest) -> ExplainResponse:
    if req.label not in LABELS:
        raise HTTPException(status_code=422, detail=f"Geçersiz etiket. Seçenekler: {LABELS}")
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Metin boş olamaz.")
    return ExplainResponse(**_model().explain(req.text, req.label))
