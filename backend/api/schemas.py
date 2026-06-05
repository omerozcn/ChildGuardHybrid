from pydantic import BaseModel, Field
from typing import Literal, Optional


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    age_group: Literal["Younger", "Pre-Teen", "Teen"] = Field(
        ..., description="XGBoost feature için kullanıcı seçimi"
    )


class ExplainRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    age_group: Literal["Younger", "Pre-Teen", "Teen"]


class LimeRequest(ExplainRequest):
    num_samples: int = Field(default=500, ge=100, le=2000)


class AgePrediction(BaseModel):
    predicted: str
    scores: dict[str, float]


class AnalyzeResponse(BaseModel):
    label:          Literal["🚨 RİSKLİ / SİBER ZORBALIK", "✅ GÜVENLİ / TEMİZ"]
    final_score:    float
    bert_score:     float
    xgb_score:      float
    age_group:      str
    age_prediction: Optional[AgePrediction] = None


class FeatureWeight(BaseModel):
    token:  str
    weight: float


class ShapResponse(BaseModel):
    features: list[FeatureWeight]


class LimeResponse(BaseModel):
    features: list[FeatureWeight]


class HealthModel(BaseModel):
    name:   str
    loaded: bool
    path:   str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    models: list[HealthModel]
