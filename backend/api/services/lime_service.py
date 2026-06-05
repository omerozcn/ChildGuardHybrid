# backend/api/services/lime_service.py
# BERT için LIME LimeTextExplainer — token ağırlıklarını döndürür

import logging
from lime.lime_text import LimeTextExplainer
from api.services import bert_service
from api.schemas import FeatureWeight

logger = logging.getLogger(__name__)

_explainer = None


def _get_explainer():
    global _explainer
    if _explainer is None:
        _explainer = LimeTextExplainer(class_names=["Güvenli", "Zararlı"])
    return _explainer


def explain(text: str, age_group: str, num_samples: int = 500) -> list[FeatureWeight]:
    if not bert_service.is_loaded("bert_general"):
        raise RuntimeError("BERT modeli yüklü değil.")

    predict_fn = bert_service.get_predict_fn()
    explainer  = _get_explainer()

    explanation = explainer.explain_instance(
        text,
        predict_fn,
        num_features=10,
        num_samples=num_samples,
        labels=[1],  # zararlı sınıfı açıkla
    )

    return [
        FeatureWeight(token=token, weight=round(float(weight), 6))
        for token, weight in explanation.as_list(label=1)
    ]
