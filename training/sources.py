"""Veri kaynakları — her biri ortak şemaya çevrilir.

Çıktı sütunları: text, source, split, <LABELS...>
Etiket değerleri 0–1 arası yumuşak (soft) olasılıklardır; kaynağın bilgi
taşımadığı etiketler `MISSING` (-1) olur ve kayıp hesabında maskelenir.

Kaynaklar (hepsi Hugging Face'ten otomatik indirilir):
  * google/civil_comments          — CC0-1.0, ~2M yorum, 7 toksiklik puanı
  * ucberkeley-dlab/measuring-hate-speech — CC-BY-4.0, hedef yaş grubu bilgisi
  * toxigen/toxigen-data (annotated) — örtük nefret; yalnızca araştırma amaçlı
  * data/supplement/child_safety_supplement.csv — projeye özel, elle yazılmış
  * ChildGuard_Cleaned.csv — eski veri; etiket kalitesi sorunlu olduğu için
    varsayılan olarak eğitime girmez (bkz. docs/DATA_AUDIT.md)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from childguard.labels import LABELS, MISSING
from childguard.text import looks_english
from training import paths


def _frame(text, source, split, **labels) -> pd.DataFrame:
    df = pd.DataFrame({"text": text, "source": source, "split": split})
    for name in LABELS:
        df[name] = labels.get(name, MISSING)
    return df


def _hash_split(keys: pd.Series, val: float = 0.1, test: float = 0.1) -> pd.Series:
    """Anahtarın hash'ine göre deterministik bölme (aynı yorum hep aynı bölmede)."""
    def bucket(k):
        h = int(hashlib.md5(str(k).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return "test" if h < test else "val" if h < test + val else "train"
    return keys.map(bucket)


# --------------------------------------------------------------------------
# Civil Comments
# --------------------------------------------------------------------------
def load_civil_comments(max_pos: int | None, neg_ratio: float, seed: int = 42,
                        max_rare: int | None = None) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset("google/civil_comments")
    rng = np.random.default_rng(seed)
    parts = []
    for hf_split, split in [("train", "train"), ("validation", "val"), ("test", "test")]:
        df = ds[hf_split].to_pandas()
        tox = np.maximum(df["toxicity"], df["severe_toxicity"])
        # Nadir alt türlerin (tehdit, cinsel, küfür, kimlik) pozitifleri sınırsız
        # alınır; aksi halde rastgele örneklemede birkaç yüz örnekte kalıyorlar.
        rare_mask = ((df["threat"] >= 0.5) | (df["sexual_explicit"] >= 0.5)
                     | (df["obscene"] >= 0.5) | (df["identity_attack"] >= 0.5))
        rare = df[rare_mask]
        if max_rare and len(rare) > max_rare:              # yalnızca duman testinde
            rare = rare.sample(max_rare, random_state=seed)
        df = df[~rare_mask]
        tox = tox[~rare_mask]
        pos = df[tox >= 0.5]
        # Sınırda (0.2–0.5) örnekler zor negatif/belirsiz örneklerdir; bunları
        # tercihli alıyoruz ki model "biraz kaba" ile "zararlı"yı ayırsın.
        border = df[(tox >= 0.2) & (tox < 0.5)]
        neg = df[tox < 0.2]
        limit = max_pos if split == "train" else (max_pos // 10 if max_pos else None)
        if limit and len(pos) > limit:
            pos = pos.sample(limit, random_state=seed)
        n_neg = int(len(pos) * neg_ratio)
        n_border = min(len(border), n_neg // 3)
        border = border.sample(n_border, random_state=seed)
        neg = neg.sample(min(len(neg), n_neg - n_border), random_state=int(rng.integers(1e9)))
        df = pd.concat([rare, pos, border, neg])
        parts.append(_frame(
            df["text"].values, "civil_comments", split,
            toxicity=np.maximum(df["toxicity"], df["severe_toxicity"]).values,
            insult=df["insult"].values,
            threat=df["threat"].values,
            sexual=df["sexual_explicit"].values,
            identity_hate=df["identity_attack"].values,
            profanity=df["obscene"].values,
        ))
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------
# Measuring Hate Speech (Berkeley D-Lab)
# --------------------------------------------------------------------------
def load_measuring_hate_speech() -> pd.DataFrame:
    from datasets import load_dataset

    raw = load_dataset("ucberkeley-dlab/measuring-hate-speech", split="train").to_pandas()
    by_comment = raw["comment_id"]

    def agree_share(col: str, min_value: float) -> pd.Series:
        """Annotatörlerin kaçta kaçı "katılıyorum" dedi (yumuşak etiket).

        MHS maddeleri 0–4 onay ölçeğidir; 2 kararsız demektir. Ortalamayı
        4'e bölmek kararsız bir yorumu %50 pozitif sayardı, bu yüzden
        ≥ min_value verenlerin oranı kullanılır.
        """
        return (raw[col] >= min_value).astype(float).groupby(by_comment).mean()

    g = raw.groupby("comment_id")
    out = pd.DataFrame({"text": g["text"].first()})
    # hatespeech: 0 hayır, 1 belirsiz, 2 evet
    out["toxicity"] = agree_share("hatespeech", 2)
    out["identity_hate"] = out["toxicity"]
    out["insult"] = pd.concat([agree_share("insult", 3), agree_share("humiliate", 3)],
                              axis=1).max(axis=1)
    out["threat"] = agree_share("violence", 3)
    minor_cols = [c for c in ("target_age_children", "target_age_teenagers") if c in raw]
    if minor_cols:
        minor = raw[minor_cols].astype(float).max(axis=1)
        out["targets_minor"] = minor.groupby(raw["comment_id"]).mean()
    out = out.reset_index()
    out["split"] = _hash_split(out["comment_id"])
    return _frame(
        out["text"].values, "measuring_hate_speech", out["split"].values,
        **{c: out[c].values for c in LABELS if c in out},
    )


# --------------------------------------------------------------------------
# ToxiGen (annotated) — örtük nefret
# --------------------------------------------------------------------------
def load_toxigen() -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset("toxigen/toxigen-data", "annotated")
    parts = []
    for hf_split in ds:
        df = ds[hf_split].to_pandas()
        tox = (df["toxicity_human"].astype(float) - 1.0) / 4.0   # 1–5 -> 0–1
        if hf_split == "train":
            split = _hash_split(df["text"], val=0.1, test=0.0)
        else:
            split = pd.Series("test", index=df.index)
        lewd = df["lewd"].map({"contains lewd content": 1.0, "might contain lewd content": 0.5,
                               "disagreement": 0.5, "doesn't contain lewd content": 0.0})
        lewd = lewd.fillna(MISSING).values
        parts.append(_frame(
            df["text"].values, "toxigen", split.values,
            toxicity=tox.values, identity_hate=tox.values, sexual=lewd,
        ))
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------
# Projeye özel ek set ve eski ChildGuard verisi
# --------------------------------------------------------------------------
def load_supplement(path: Path = paths.SUPPLEMENT) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["split"] = _hash_split(df["text"], val=0.15, test=0.0)
    return _frame(df["text"].values, "supplement", df["split"].values,
                  **{c: df[c].astype(float).values for c in LABELS if c in df})


def load_childguard_legacy(path: Path = paths.LEGACY_CSV) -> pd.DataFrame:
    """Eski ChildGuard verisi — sadece `toxicity` etiketi, temizlenmiş haliyle."""
    df = pd.read_csv(path, encoding="cp1252", encoding_errors="replace")
    df = df.dropna(subset=["text", "actual_class"])
    text = df["text"].astype(str).str.strip()
    wiki = text.str.contains(r"\b(?:wikipedia|utc|talk page|wp\w+|classnotice|aligncenter)\b",
                             case=False, regex=True)
    # Aksanları silinmiş Hırvatça/Sırpça satırlar ASCII testinden geçer; bu yüzden
    # yaygın İngilizce kelime oranına bakılır. Bu, kök haline getirilip stop-word'leri
    # atılmış satırları da eler — model gerçek kullanımda ham metin görür.
    english = text.map(looks_english)
    words = text.str.split().str.len()
    keep = ~wiki & english & words.between(4, 200)
    df, text = df[keep], text[keep]
    split = _hash_split(text)
    return _frame(text.values, "childguard_legacy", split.values,
                  toxicity=df["actual_class"].astype(float).values)
