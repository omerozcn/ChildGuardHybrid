import logging
from fastapi import APIRouter, HTTPException
from api.schemas import AnalyzeRequest, AnalyzeResponse
from api.services import bert_service, xgb_service, hybrid

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Metin boş olamaz.")

    bert_score = bert_service.predict(req.text)
    xgb_score  = xgb_service.predict(req.text, req.age_group)

    if bert_score is None and xgb_score is None:
        raise HTTPException(status_code=503, detail="Hiçbir model yüklü değil.")

    final, b, x = hybrid.compute(bert_score, xgb_score)

    age_result = None
    if final > hybrid.THRESHOLD:
        age_result = bert_service.predict_age_group(req.text)

    return AnalyzeResponse(
        label          = hybrid.label(final),
        final_score    = final,
        bert_score     = b,
        xgb_score      = x,
        age_group      = req.age_group,
        age_prediction = age_result,
    )
