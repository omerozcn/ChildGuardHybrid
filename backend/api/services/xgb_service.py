import os, re, logging
from typing import Optional
import numpy as np
import joblib
from scipy.sparse import hstack, csr_matrix

logger = logging.getLogger(__name__)
MODEL_BASE = os.getenv("MODEL_BASE_PATH", "./final_models")

XGB_PATH     = os.path.join(MODEL_BASE, "xgboost_model.pkl")
TFIDF_PATH   = os.path.join(MODEL_BASE, "tfidf_vectorizer.pkl")
AGE_COL_PATH = os.path.join(MODEL_BASE, "age_group_columns.pkl")

_xgb_model         = None
_vectorizer        = None
_age_group_columns = []
_xgb_loaded        = False


def _clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s!?.,]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_model() -> bool:
    global _xgb_model, _vectorizer, _age_group_columns, _xgb_loaded
    try:
        _xgb_model  = joblib.load(XGB_PATH)
        _vectorizer = joblib.load(TFIDF_PATH)
        if os.path.exists(AGE_COL_PATH):
            _age_group_columns = joblib.load(AGE_COL_PATH)
        # Eğer model yaş one-hot'suz eğitildiyse age_group_columns'u devre dışı bırak
        tfidf_dim = len(_vectorizer.get_feature_names_out())
        expected  = getattr(_xgb_model, "n_features_in_", None)
        if expected is not None and expected == tfidf_dim + 2:
            _age_group_columns = []
            logger.info("XGBoost: yaş one-hot devre dışı (model %d özellik bekliyor)", expected)
        _xgb_loaded = True
        logger.info("XGBoost yüklendi.")
        return True
    except Exception as exc:
        logger.error("XGBoost yüklenemedi: %s", exc)
        return False


def _build_features(text, age_group):
    cleaned = _clean(text)
    tfidf_v = _vectorizer.transform([cleaned])
    meta    = csr_matrix([[len(cleaned), len(cleaned.split())]])
    if _age_group_columns:
        age_vec = [1 if col == f"AgeGroup_{age_group}" else 0 for col in _age_group_columns]
        return hstack([tfidf_v, meta, csr_matrix([age_vec])])
    return hstack([tfidf_v, meta])


def predict(text: str, age_group: str) -> Optional[float]:
    if not _xgb_loaded:
        return None
    try:
        return float(_xgb_model.predict_proba(_build_features(text, age_group))[0][1])
    except Exception as exc:
        logger.error("XGBoost tahmin hatası: %s", exc)
        return None


def get_features(text, age_group):
    return _build_features(text, age_group) if _xgb_loaded else None


def get_feature_names():
    if not _xgb_loaded:
        return []
    names = _vectorizer.get_feature_names_out().tolist() + ["text_len", "word_cnt"]
    return names + _age_group_columns if _age_group_columns else names


def get_raw_model():
    return _xgb_model


def is_loaded() -> bool:
    return _xgb_loaded


def model_path() -> str:
    return XGB_PATH
