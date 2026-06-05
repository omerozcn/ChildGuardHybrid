# backend/api/services/shap_service.py
# XGBoost için SHAP TreeExplainer — top-10 token ağırlığı döndürür

import logging
import numpy as np
import shap
from api.services import xgb_service
from api.schemas import FeatureWeight

logger = logging.getLogger(__name__)

_explainer = None


def _get_explainer():
    global _explainer
    if _explainer is None:
        model = xgb_service.get_raw_model()
        if model is None:
            raise RuntimeError("XGBoost modeli yüklü değil.")
        _explainer = shap.TreeExplainer(model)
    return _explainer


STOP_WORDS = {
    "the","a","an","is","it","in","of","to","and","or","that","this",
    "was","for","on","are","with","as","at","be","by","from","have",
    "has","had","not","but","we","you","he","she","they","i","my",
    "your","his","her","our","its","do","did","will","would","could",
    "should","may","might","about","up","out","so","if","me","him",
    "them","then","than","can","just","been","more","also","now",
}
META_FEATURES = {"text_len", "word_cnt"}
_SKIP = STOP_WORDS | META_FEATURES


def explain(text: str, age_group: str, top_n: int = 10) -> list[FeatureWeight]:
    X = xgb_service.get_features(text, age_group)
    if X is None:
        raise RuntimeError("XGBoost modeli yüklü değil.")

    explainer   = _get_explainer()
    X_dense     = X.toarray()
    shap_values = explainer.shap_values(X_dense)

    feature_names = xgb_service.get_feature_names()
    values        = shap_values[0] if shap_values.ndim > 1 else shap_values
    row           = X_dense[0]

    # Sadece input'ta non-zero olan feature'lar — metinde gerçekten geçen tokenlar
    present = [
        i for i in np.flatnonzero(row)
        if i < len(feature_names) and feature_names[i] not in _SKIP
    ]
    ordered = sorted(present, key=lambda i: abs(values[i]), reverse=True)[:top_n]

    return [
        FeatureWeight(
            token  = feature_names[i],
            weight = round(float(values[i]), 6),
        )
        for i in ordered
    ]
