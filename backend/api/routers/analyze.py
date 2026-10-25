from fastapi import APIRouter, HTTPException

from api.schemas import (AnalyzeRequest, AnalyzeResponse, BatchAnalyzeRequest,
                         CategoryResult, RuleHit)
from api.services import model_service
from childguard.labels import DISPLAY_TR, LABELS, PRIMARY

router = APIRouter()


def _model():
    model = model_service.get()
    if model is None:
        raise HTTPException(status_code=503, detail=model_service.error() or "Model yüklü değil.")
    return model


def _to_response(result: dict, model) -> AnalyzeResponse:
    def cat(label):
        return CategoryResult(
            name=label, display_name=DISPLAY_TR[label],
            score=result["scores"][label], flagged=result["flags"][label],
            threshold=model.thresholds[label]["flag"],
        )

    return AnalyzeResponse(
        harmful=result["harmful"],
        risk_level=result["risk_level"],
        score=result["scores"][PRIMARY],
        categories=[cat(l) for l in LABELS if l not in (PRIMARY, "targets_minor")],
        targets_minor=cat("targets_minor"),
        rule_hits=[RuleHit(**h) for h in result["rule_hits"]],
        language_warning=result["language_warning"],
        model_version=model.version,
    )


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Metin boş olamaz.")
    model = _model()
    return _to_response(model.predict([req.text])[0], model)


@router.post("/analyze/batch", response_model=list[AnalyzeResponse])
def analyze_batch(req: BatchAnalyzeRequest) -> list[AnalyzeResponse]:
    texts = [t for t in req.texts if t.strip()]
    if not texts:
        raise HTTPException(status_code=422, detail="Boş olmayan en az bir metin gerekli.")
    model = _model()
    return [_to_response(r, model) for r in model.predict(texts)]
