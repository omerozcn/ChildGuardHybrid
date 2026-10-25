"""Yığınlama (stacking), kalibrasyon ve eşik seçimi — yalnızca DOĞRULAMA setinde.

Mevcut projedeki hatanın (eşik ve Optuna'nın test setinde seçilmesi) tersine,
burada bütün ayarlar validation üzerinde yapılır; test seti `evaluate.py`
tarafından en sonda bir kez kullanılır.

Eşik politikası:
  * flag: Güvenlik açısından kritik etiketlerde (toxicity, threat, sexual)
    hedef recall'u (varsayılan 0.90) sağlayan en yüksek eşik — ancak o
    noktada precision ≥ `--min-precision` (0.40) ise. Değilse recall'u
    ağırlıklı tutan F2'yi en yüksek yapan eşik. (v4.0'da tehdit eşiği bu
    kontrol olmadığı için 0.026'ya inmiş, "kill/die" geçen masum cümleler
    alarm vermişti.) Diğer etiketlerde F1'i en yüksek yapan eşik.
  * high: precision ≥ 0.90 sağlayan en düşük eşik ("yüksek risk").

    python -m training.calibrate
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from childguard import __version__
from childguard.labels import LABELS
from childguard.pipeline import Stacker, TransformerModel, _logit
from training import paths

SAFETY_CRITICAL = {"toxicity", "threat", "sexual"}


def _f_beta(prec: np.ndarray, rec: np.ndarray, beta: float) -> np.ndarray:
    b2 = beta * beta
    return (1 + b2) * prec * rec / np.clip(b2 * prec + rec, 1e-9, None)


def pick_thresholds(y: np.ndarray, p: np.ndarray, label: str, target_recall: float,
                    min_precision: float = 0.40) -> dict:
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]               # thr ile aynı uzunluk
    policy = "f1"
    if label in SAFETY_CRITICAL:
        ok = np.where(rec >= target_recall)[0]
        i = ok[-1] if len(ok) else int(np.argmax(rec))
        policy = f"recall≥{target_recall}"
        if prec[i] < min_precision:
            i = int(np.argmax(_f_beta(prec, rec, 2.0)))
            policy = f"f2 (recall≥{target_recall} noktasında precision {prec[ok[-1] if len(ok) else i]:.2f} < {min_precision})"
    else:
        i = int(np.argmax(_f_beta(prec, rec, 1.0)))
    flag = float(thr[i])
    hi = np.where((prec >= 0.90) & (thr >= flag))[0]
    high = float(thr[hi[0]]) if len(hi) else max(flag, 0.9)
    return {"flag": round(flag, 4), "high": round(high, 4),
            "val_precision": round(float(prec[i]), 4), "val_recall": round(float(rec[i]), 4),
            "policy": policy}


RAISE_CANDIDATES = ["insult", "threat", "sexual", "identity_hate"]


def choose_raise_rules(probs: np.ndarray, y_harmful: np.ndarray, thresholds: dict) -> dict:
    """Hangi risk türü, hangi eşikte genel "zararlı" kararını tetiklesin?

    Her tür için {yok, yüksek risk eşiği, işaretleme eşiği} seçenekleri
    doğrulama setinde, genel kararın F1'ine göre tek tek (açgözlü) seçilir.
    v4.1'de bu kural elle "yüksek risk" olarak sabitlenmişti; cinsel içerik
    gibi türlerde kayba yol açıyordu.
    """
    tox = probs[:, LABELS.index("toxicity")] >= thresholds["toxicity"]["flag"]

    def f1(pred):
        tp = (pred & y_harmful).sum()
        return 2 * tp / max(pred.sum() + y_harmful.sum(), 1)

    rules, decision = {}, tox.copy()
    for label in RAISE_CANDIDATES:
        p = probs[:, LABELS.index(label)]
        best_mode, best_pred, best_score = "none", decision, f1(decision)
        for mode in ("high", "flag"):
            pred = decision | (p >= thresholds[label][mode])
            score = f1(pred)
            if score > best_score + 1e-4:
                best_mode, best_pred, best_score = mode, pred, score
        rules[label], decision = best_mode, best_pred
    return rules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--target-recall", type=float, default=0.90)
    ap.add_argument("--min-precision", type=float, default=0.40,
                    help="Güvenlik etiketlerinde hedef recall noktasında kabul edilen en düşük precision")
    ap.add_argument("--no-transformer", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model_dir) if args.model_dir else paths.model_dir()
    val = pd.read_parquet(paths.data_dir() / "val.parquet")
    texts = val["text"].tolist()
    lexical = joblib.load(model_dir / "lexical.joblib")
    p_l = lexical.predict_proba(texts)

    p_t, meta = None, {}
    tdir = model_dir / "transformer"
    if not args.no_transformer and tdir.is_dir():
        meta = json.loads((tdir / "training_meta.json").read_text(encoding="utf-8"))
        tm = TransformerModel(tdir, max_len=meta.get("max_len", 128))
        print(f"Transformer tahminleri ({len(texts):,} satır, {tm.device})...")
        p_t = tm.predict_proba(texts, batch_size=64)

    stacker, thresholds, metrics = {}, {}, {}
    for j, label in enumerate(LABELS):
        known = (val[label] >= 0).values & ~np.isnan(p_l[:, j])
        y = (val.loc[known, label] >= 0.5).astype(int).values
        if known.sum() < 30 or y.sum() < 5 or y.sum() == len(y):
            # Kalibre edecek veri yok: modelin ham çıktısını kullan.
            stacker[label] = {"transformer": 1.0 if p_t is not None else 0.0,
                              "lexical": 0.0 if p_t is not None else 1.0, "bias": 0.0}
            thresholds[label] = {"flag": 0.5, "high": 0.9, "val_precision": None,
                                 "val_recall": None}
            print(f"  {label:14s} kalibrasyon atlandı (yetersiz doğrulama verisi)")
            continue
        feats = [_logit(np.nan_to_num(p_l[known, j], nan=0.5))]
        if p_t is not None:
            feats.insert(0, _logit(p_t[known, j]))
        Z = np.column_stack(feats)
        lr = LogisticRegression(C=1.0).fit(Z, y)
        coef = lr.coef_[0]
        stacker[label] = {
            "transformer": float(coef[0]) if p_t is not None else 0.0,
            "lexical": float(coef[-1]),
            "bias": float(lr.intercept_[0]),
        }
        p = lr.predict_proba(Z)[:, 1]
        thresholds[label] = pick_thresholds(y, p, label, args.target_recall, args.min_precision)
        metrics[label] = {
            "val_pr_auc": round(float(average_precision_score(y, p)), 4),
            "val_roc_auc": round(float(roc_auc_score(y, p)), 4),
            "val_pr_auc_lexical": round(float(average_precision_score(y, p_l[known, j])), 4),
            **({"val_pr_auc_transformer":
                round(float(average_precision_score(y, p_t[known, j])), 4)}
               if p_t is not None else {}),
            "n": int(known.sum()), "positives": int(y.sum()),
        }
        t = thresholds[label]
        print(f"  {label:14s} PR-AUC={metrics[label]['val_pr_auc']:.3f} "
              f"flag={t['flag']:.3f} (P={t['val_precision']:.2f} R={t['val_recall']:.2f}) "
              f"high={t['high']:.3f} · {t['policy']}")

    # Genel kararı yükseltme kuralı: bütün doğrulama satırlarında, kalibre
    # edilmiş olasılıklarla seçilir.
    full_probs = Stacker(stacker).combine(p_t, p_l)
    y_harmful = (val["toxicity"] >= 0.5).values
    raise_rules = choose_raise_rules(full_probs, y_harmful, thresholds)
    print(f"  Genel kararı yükselten türler: {raise_rules}")

    config = {
        "version": __version__,
        "raise_overall": raise_rules,
        "trained_on": date.today().isoformat(),
        "labels": LABELS,
        "uses_transformer": p_t is not None,
        "base_model": meta.get("base_model"),
        "max_len": meta.get("max_len", 128),
        "target_recall": args.target_recall,
        "min_precision": args.min_precision,
        "stacker": stacker,
        "thresholds": thresholds,
        "val_metrics": metrics,
    }
    cfg_path = model_dir / "config.json"
    if cfg_path.exists():                                   # önceki kalibrasyonu sakla
        (model_dir / "config.prev.json").write_text(cfg_path.read_text(encoding="utf-8"),
                                                    encoding="utf-8")
    cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"✅ Yazıldı: {model_dir / 'config.json'}")


if __name__ == "__main__":
    main()
