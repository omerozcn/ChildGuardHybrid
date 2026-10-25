"""Uçtan uca tahmin hattı: transformer + sözcüksel model + yığınlama + kurallar.

    model = ChildGuardModel.load("final_models/v4")
    model.predict(["you are such a loser"])
    model.explain("you are such a loser")

Model klasörü:
    config.json          etiketler, eşikler, yığınlama katsayıları, metrikler
    lexical.joblib       TF-IDF (kelime + karakter) + etiket başına LR
    transformer/         (opsiyonel) Hugging Face modeli ve tokenizer
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from childguard.labels import LABELS, PRIMARY
from childguard.rules import apply_floors, find_hits
from childguard.text import for_lexical, for_transformer, looks_english


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class LexicalModel:
    """TF-IDF + etiket başına lojistik regresyon. joblib ile kaydedilir."""

    def __init__(self, vectorizer, models: dict):
        self.vectorizer = vectorizer
        self.models = models          # label -> sklearn LogisticRegression

    def transform(self, texts: list[str]):
        return self.vectorizer.transform([for_lexical(t) for t in texts])

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        X = self.transform(texts)
        out = np.zeros((len(texts), len(LABELS)))
        for j, label in enumerate(LABELS):
            m = self.models.get(label)
            out[:, j] = m.predict_proba(X)[:, 1] if m is not None else np.nan
        return out

    def contributions(self, text: str, label: str, top_n: int = 10) -> list[tuple[str, float]]:
        """Doğrusal modelde katkı = katsayı × özellik değeri (kesin, örneklemesiz)."""
        m = self.models.get(label)
        if m is None:
            return []
        x = self.transform([text]).tocoo()
        names = self.vectorizer.get_feature_names_out()
        contrib = [(names[j], float(m.coef_[0, j] * v)) for j, v in zip(x.col, x.data)]
        contrib.sort(key=lambda t: abs(t[1]), reverse=True)
        return contrib[:top_n]


class TransformerModel:
    def __init__(self, path: Path, device: str | None = None, max_len: int = 128):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(self.device).eval()
        self.max_len = max_len

    def predict_proba(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        torch = self.torch
        out = []
        for i in range(0, len(texts), batch_size):
            batch = [for_transformer(t) for t in texts[i:i + batch_size]]
            enc = self.tokenizer(batch, truncation=True, max_length=self.max_len,
                                 padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
            out.append(torch.sigmoid(logits).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(LABELS)))


@dataclass
class Stacker:
    """Etiket başına: p = sigmoid(a·logit(p_t) + b·logit(p_l) + c)."""
    weights: dict   # label -> {"transformer": a, "lexical": b, "bias": c}

    def combine(self, p_t: np.ndarray | None, p_l: np.ndarray) -> np.ndarray:
        p_l = np.nan_to_num(p_l, nan=0.5)    # sözcüksel modeli olmayan etiket
        out = np.zeros_like(p_l)
        for j, label in enumerate(LABELS):
            w = self.weights[label]
            z = w["bias"] + w["lexical"] * _logit(p_l[:, j])
            if p_t is not None and w.get("transformer"):
                z = z + w["transformer"] * _logit(p_t[:, j])
            out[:, j] = _sigmoid(z)
        return out


class ChildGuardModel:
    def __init__(self, config: dict, lexical: LexicalModel,
                 transformer: TransformerModel | None):
        self.config = config
        self.lexical = lexical
        self.transformer = transformer
        self.stacker = Stacker(config["stacker"])
        self.thresholds: dict = config["thresholds"]

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, model_dir: str | Path, device: str | None = None,
             use_transformer: bool = True) -> "ChildGuardModel":
        import joblib

        model_dir = Path(model_dir)
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        lexical = joblib.load(model_dir / "lexical.joblib")
        transformer = None
        tdir = model_dir / "transformer"
        if use_transformer and config.get("uses_transformer") and tdir.is_dir():
            transformer = TransformerModel(tdir, device, config.get("max_len", 128))
        return cls(config, lexical, transformer)

    @property
    def version(self) -> str:
        return self.config.get("version", "4.x")

    # ------------------------------------------------------------------
    def raw_proba(self, texts: list[str]) -> np.ndarray:
        p_l = self.lexical.predict_proba(texts)
        p_t = self.transformer.predict_proba(texts) if self.transformer else None
        return self.stacker.combine(p_t, p_l)

    def risk_level(self, p: float) -> str:
        th = self.thresholds[PRIMARY]
        if p >= th["high"]:
            return "high"
        if p >= th["flag"]:
            return "medium"
        return "low"

    def predict(self, texts: list[str]) -> list[dict]:
        probs = self.raw_proba(texts)
        results = []
        for text, row in zip(texts, probs):
            scores = {label: float(row[j]) for j, label in enumerate(LABELS)}
            hits = find_hits(text)
            scores = apply_floors(scores, hits)
            # Ciddi bir risk türü kendi eşiğini aşıyorsa genel skor da en az
            # işaretleme seviyesine çıkar (ör. tehdit var ama "toxicity" düşük
            # kalmışsa). Tek başına küfür ve "çocuğu hedef alıyor" bunu yapmaz.
            harmful_types = [l for l in LABELS
                             if l not in (PRIMARY, "targets_minor", "profanity")]
            for l in harmful_types:
                if scores[l] >= self.thresholds[l]["flag"]:
                    scores[PRIMARY] = max(scores[PRIMARY], self.thresholds[PRIMARY]["flag"])
            flags = {l: scores[l] >= self.thresholds[l]["flag"] for l in LABELS}
            results.append({
                "harmful": bool(flags[PRIMARY]),
                "risk_level": self.risk_level(scores[PRIMARY]),
                "scores": {l: round(v, 4) for l, v in scores.items()},
                "flags": flags,
                "rule_hits": [
                    {"name": h.name, "description": h.description, "match": h.match}
                    for h in hits
                ],
                "language_warning": not looks_english(text),
            })
        return results

    # ------------------------------------------------------------------
    def explain(self, text: str, label: str = PRIMARY, top_n: int = 10) -> dict:
        """İki açıklama döndürür:

        * words: kelime çıkarma (occlusion) — her kelime silindiğinde tam
          hattın skorunun ne kadar düştüğü. Deterministik, tüm modeli kapsar.
        * ngrams: sözcüksel modelin kesin doğrusal katkıları.
        """
        words = text.split()
        base = float(self.raw_proba([text])[0, LABELS.index(label)])
        variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
        word_scores = []
        if variants:
            probs = self.raw_proba(variants)[:, LABELS.index(label)]
            word_scores = [
                {"token": w, "weight": round(base - float(p), 6)}
                for w, p in zip(words, probs)
            ]
        ngrams = [{"token": t, "weight": round(v, 6)}
                  for t, v in self.lexical.contributions(text, label, top_n)]
        return {"label": label, "base_score": round(base, 4),
                "words": word_scores, "ngrams": ngrams}
