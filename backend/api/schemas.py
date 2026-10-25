from typing import Literal, Optional

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class BatchAnalyzeRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64)


class CategoryResult(BaseModel):
    name: str
    display_name: str
    score: float
    flagged: bool
    threshold: float


class RuleHit(BaseModel):
    name: str
    description: str
    match: str


class AnalyzeResponse(BaseModel):
    harmful: bool
    risk_level: Literal["low", "medium", "high"]
    score: float
    categories: list[CategoryResult]
    targets_minor: CategoryResult
    rule_hits: list[RuleHit]
    language_warning: bool
    model_version: str


class ExplainRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    label: str = "toxicity"


class TokenWeight(BaseModel):
    token: str
    weight: float


class ExplainResponse(BaseModel):
    label: str
    base_score: float
    words: list[TokenWeight]
    ngrams: list[TokenWeight]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_version: Optional[str] = None
    base_model: Optional[str] = None
    uses_transformer: bool = False
    device: Optional[str] = None
    error: Optional[str] = None
