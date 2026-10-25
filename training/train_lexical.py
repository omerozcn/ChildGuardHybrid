"""Sözcüksel model: kelime (1–2) + karakter (2–5) TF-IDF, etiket başına LR.

Karakter n-gramları "st00pid", "l o s e r" gibi gizlenmiş yazımları yakalar.
Yumuşak etiketler, her satırı iki kez (pozitif ağırlık p, negatif ağırlık
1−p) vererek doğrudan kullanılır. Etiketler paralel eğitilir.

    python -m training.train_lexical
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import FeatureUnion

from childguard.labels import LABELS
from childguard.pipeline import LexicalModel
from childguard.text import for_lexical
from training import paths


def build_vectorizer(max_word: int, max_char: int) -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=max_word,
                                 sublinear_tf=True, token_pattern=r"(?u)\b\w[\w']*\b")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5,
                                 max_features=max_char, sublinear_tf=True)),
    ])


def fit_soft_lr(X, p: np.ndarray, C: float) -> LogisticRegression:
    X2 = vstack([X, X])
    y2 = np.concatenate([np.ones(len(p)), np.zeros(len(p))])
    w2 = np.concatenate([p, 1.0 - p])
    keep = w2 > 1e-6
    lr = LogisticRegression(C=C, max_iter=2000, solver="liblinear")
    lr.fit(X2[keep], y2[keep], sample_weight=w2[keep])
    return lr


def _fit_label(label, Xtr, Xva, train, val, Cs):
    tr_mask = (train[label] >= 0).values
    va_mask = (val[label] >= 0).values
    y_va = (val.loc[va_mask, label] >= 0.5).astype(int).values
    if tr_mask.sum() < 50 or (train.loc[tr_mask, label] >= 0.5).sum() < 5:
        return label, None, None, None, int(tr_mask.sum())
    best = (None, -1.0, None)
    for C in Cs:
        m = fit_soft_lr(Xtr[tr_mask], train.loc[tr_mask, label].values, C)
        score = (average_precision_score(y_va, m.predict_proba(Xva[va_mask])[:, 1])
                 if 0 < y_va.sum() < len(y_va) else 0.0)
        if score > best[1]:
            best = (m, score, C)
    return label, best[0], best[1], best[2], int(tr_mask.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-word", type=int, default=150_000)
    ap.add_argument("--max-char", type=int, default=200_000)
    ap.add_argument("--max-rows", type=int, default=250_000,
                    help="Eğitim satırı üst sınırı (sözcüksel model için yeterli; süreyi sınırlar)")
    ap.add_argument("--C", type=float, nargs="+", default=[1.0, 4.0],
                    help="Doğrulama setinde denenecek C değerleri")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    data = paths.data_dir()
    train = pd.read_parquet(data / "train.parquet")
    val = pd.read_parquet(data / "val.parquet")
    if len(train) > args.max_rows:
        train = train.sample(args.max_rows, random_state=42).reset_index(drop=True)
    print(f"train={len(train):,} val={len(val):,}", flush=True)

    t0 = time.time()
    vec = build_vectorizer(args.max_word, args.max_char)
    if len(train) < 20_000:                    # duman testi / küçük veri
        vec.set_params(word__min_df=1, char__min_df=2)
    Xtr = vec.fit_transform([for_lexical(t) for t in train["text"]]).tocsr()
    Xva = vec.transform([for_lexical(t) for t in val["text"]]).tocsr()
    print(f"vektörleştirme: {Xtr.shape[1]:,} özellik, {time.time() - t0:.0f}s", flush=True)

    # liblinear GIL'i bıraktığı için thread'ler gerçekten paralel çalışır ve
    # büyük seyrek matrisler kopyalanmaz.
    results = Parallel(n_jobs=args.jobs, prefer="threads")(
        delayed(_fit_label)(label, Xtr, Xva, train, val, args.C) for label in LABELS)

    models = {}
    for label, m, score, C, n in results:
        if m is None:
            print(f"  {label:14s} atlandı (yetersiz veri, n={n:,})")
            continue
        models[label] = m
        print(f"  {label:14s} C={C:<4} val PR-AUC={score:.3f} (eğitim n={n:,})")

    out = paths.model_dir() if args.out is None else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(LexicalModel(vec, models), out / "lexical.joblib", compress=3)
    print(f"✅ Kaydedildi: {out / 'lexical.joblib'}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
