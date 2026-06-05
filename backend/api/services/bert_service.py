import os, re, logging
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib

logger = logging.getLogger(__name__)
MODEL_BASE = os.getenv("MODEL_BASE_PATH", "./final_models")

GENERAL_PATH   = os.path.join(MODEL_BASE, "bert_general")
AGE_GROUP_PATH = os.path.join(MODEL_BASE, "bert_age_group")
CONFIG_PATH    = os.path.join(MODEL_BASE, "bert_general_config.pkl")
LABEL_PATH     = os.path.join(MODEL_BASE, "age_label_names.pkl")

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 256

_tokenizer      = None
_general_model  = None
_age_model      = None
_threshold      = 0.45
_age_labels     = ["Younger", "Pre-Teen", "Teen"]
_general_loaded = False
_age_loaded     = False


def _clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s!?.,]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _encode(text):
    return _tokenizer(text, return_tensors="pt", truncation=True,
                      padding=True, max_length=MAX_LEN)


def load_all_models():
    global _tokenizer, _general_model, _age_model
    global _threshold, _age_labels, _general_loaded, _age_loaded
    status = {}

    if os.path.isdir(GENERAL_PATH):
        try:
            _tokenizer     = BertTokenizer.from_pretrained(GENERAL_PATH)
            _general_model = BertForSequenceClassification.from_pretrained(GENERAL_PATH)
            _general_model.to(DEVICE).eval()
            if os.path.exists(CONFIG_PATH):
                _threshold = joblib.load(CONFIG_PATH).get("threshold", 0.45)
            _general_loaded = True
            logger.info("bert_general yüklendi (thr=%.2f)", _threshold)
        except Exception as e:
            logger.error("bert_general yüklenemedi: %s", e)
    status["bert_general"] = _general_loaded

    if os.path.isdir(AGE_GROUP_PATH):
        try:
            _age_model = BertForSequenceClassification.from_pretrained(AGE_GROUP_PATH)
            _age_model.to(DEVICE).eval()
            if os.path.exists(LABEL_PATH):
                _age_labels = joblib.load(LABEL_PATH)
            _age_loaded = True
            logger.info("bert_age_group yüklendi")
        except Exception as e:
            logger.error("bert_age_group yüklenemedi: %s", e)
    status["bert_age_group"] = _age_loaded

    return status


def predict(text: str, age_group: str = None):
    if not _general_loaded:
        return None
    enc = _encode(_clean(text))
    with torch.no_grad():
        logits = _general_model(**{k: v.to(DEVICE) for k, v in enc.items()}).logits
    return float(torch.softmax(logits, dim=1)[0, 1].cpu())


def predict_age_group(text: str):
    if not _age_loaded:
        return None
    enc = _encode(_clean(text))
    with torch.no_grad():
        logits = _age_model(**{k: v.to(DEVICE) for k, v in enc.items()}).logits
    probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
    idx   = int(np.argmax(probs))
    return {
        "predicted": _age_labels[idx],
        "scores":    {_age_labels[i]: round(p, 4) for i, p in enumerate(probs)},
    }


def get_predict_fn(age_group=None):
    def _fn(texts):
        return np.array([[1 - (predict(t) or 0.0), predict(t) or 0.0] for t in texts])
    return _fn


def is_loaded(key="bert_general"):
    return _general_loaded if key == "bert_general" else _age_loaded


def model_path(key="bert_general"):
    return GENERAL_PATH if key == "bert_general" else AGE_GROUP_PATH
