"""Son değerlendirme: test seti (bir kez), altın set ve eski ChildGuard verisi.

İsteğe bağlı olarak eski v3 modeliyle (bert_general, hibrit eşik 0.45)
altın set üzerinde karşılaştırma yapar.

    python -m training.evaluate --compare-v3

Çıktı: reports/eval_report.md ve reports/eval_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from childguard.labels import DISPLAY_TR, LABELS, PRIMARY
from childguard.pipeline import ChildGuardModel
from training import paths


def label_metrics(df: pd.DataFrame, results: list[dict], thresholds: dict) -> dict:
    out = {}
    for label in LABELS:
        known = (df[label] >= 0).values
        y = (df.loc[known, label] >= 0.5).astype(int).values
        if known.sum() == 0 or y.sum() == 0 or y.sum() == len(y):
            continue
        p = np.array([r["scores"][label] for r, k in zip(results, known) if k])
        pred = np.array([r["flags"][label] for r, k in zip(results, known) if k]).astype(int)
        out[label] = {
            "n": int(known.sum()), "positives": int(y.sum()),
            "pr_auc": round(float(average_precision_score(y, p)), 4),
            "roc_auc": round(float(roc_auc_score(y, p)), 4),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
            "threshold": thresholds[label]["flag"],
        }
    return out


def gold_by_category(gold: pd.DataFrame, results: list[dict]) -> dict:
    out = {}
    harmful = [r["harmful"] for r in results]
    for cat, g in gold.groupby("category"):
        y = (g[PRIMARY] >= 0.5).values
        pred = np.array([harmful[i] for i in g.index])
        out[cat] = {
            "n": int(len(g)),
            "harmful_examples": int(y.sum()),
            "caught": int((pred & y).sum()),
            "false_alarms": int((pred & ~y).sum()),
            "recall": round(float((pred & y).sum() / y.sum()), 3) if y.sum() else None,
            "false_alarm_rate": round(float((pred & ~y).sum() / (~y).sum()), 3) if (~y).sum() else None,
        }
    return out


def v3_baseline(gold: pd.DataFrame) -> dict | None:
    """Eski sistemin altın setteki toxicity başarısı (sadece bert_general).

    v3 hibritinde XGBoost yalnızca skoru ≥0.65 ise katılıyordu ve test
    raporlarına göre çoğu zaman katılmıyordu; burada BERT bileşeni ölçülür.
    """
    path = paths.V3_BERT
    if not path.is_dir():
        return None
    import re
    import torch
    from transformers import BertForSequenceClassification, BertTokenizer

    def clean(t):
        t = str(t).lower()
        t = re.sub(r"http\S+|www\S+", "", t)
        t = re.sub(r"@\w+", "", t)
        t = re.sub(r"[^a-zA-Z0-9\s!?.,]", "", t)
        return re.sub(r"\s+", " ", t).strip()

    tok = BertTokenizer.from_pretrained(path)
    model = BertForSequenceClassification.from_pretrained(path).eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(gold), 32):
            enc = tok([clean(t) for t in gold["text"][i:i + 32]], truncation=True,
                      max_length=256, padding=True, return_tensors="pt")
            probs.extend(torch.softmax(model(**enc).logits, dim=1)[:, 1].tolist())
    p = np.array(probs)
    y = (gold[PRIMARY] >= 0.5).astype(int).values
    pred = (p > 0.45).astype(int)
    per_cat = {}
    for cat, g in gold.groupby("category"):
        yy, pp = y[g.index], pred[g.index]
        per_cat[cat] = {
            "recall": round(float((pp & yy).sum() / yy.sum()), 3) if yy.sum() else None,
            "false_alarms": int((pp & (1 - yy)).sum()),
        }
    return {
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "by_category": per_cat,
    }


def to_markdown(report: dict) -> str:
    lines = [f"# ChildGuard AI v{report['version']} — Değerlendirme Raporu", ""]
    lines += [f"Model: `{report['model']}` · Eşikler doğrulama setinde seçildi "
              f"(hedef recall {report['target_recall']}).", ""]

    def table(title, metrics):
        lines.extend([f"## {title}", "",
                      "| Etiket | n | Pozitif | PR-AUC | ROC-AUC | Precision | Recall | F1 | Eşik |",
                      "|---|---|---|---|---|---|---|---|---|"])
        for l, m in metrics.items():
            lines.append(f"| {DISPLAY_TR[l]} | {m['n']} | {m['positives']} | {m['pr_auc']} | "
                         f"{m['roc_auc']} | {m['precision']} | {m['recall']} | {m['f1']} | "
                         f"{m['threshold']} |")
        lines.append("")

    table("Test seti (eğitimde ve ayarlamada hiç kullanılmadı)", report["test"])
    table("Altın set (elle yazılmış gerçekçi mesajlar)", report["gold"])
    lines += ["## Altın set — kategori bazında", "",
              "| Kategori | n | Zararlı | Yakalanan | Yanlış alarm | Recall | Yanlış alarm oranı |",
              "|---|---|---|---|---|---|---|"]
    for c, m in report["gold_by_category"].items():
        lines.append(f"| {c} | {m['n']} | {m['harmful_examples']} | {m['caught']} | "
                     f"{m['false_alarms']} | {m['recall']} | {m['false_alarm_rate']} |")
    lines.append("")
    if report.get("v3_baseline"):
        b, g = report["v3_baseline"], report["gold"].get(PRIMARY, {})
        lines += ["## Eski sistemle karşılaştırma (altın set, zararlı/zararsız)", "",
                  "| Sistem | PR-AUC | Precision | Recall | F1 |", "|---|---|---|---|---|",
                  f"| v3 (bert_general, eşik 0.45) | {b['pr_auc']} | {b['precision']} | "
                  f"{b['recall']} | {b['f1']} |",
                  f"| v{report['version']} | {g.get('pr_auc')} | {g.get('precision')} | "
                  f"{g.get('recall')} | {g.get('f1')} |", "",
                  "| Kategori | v3 recall | v4 recall | v3 yanlış alarm | v4 yanlış alarm |",
                  "|---|---|---|---|---|"]
        for c, m in report["gold_by_category"].items():
            o = b["by_category"].get(c, {})
            lines.append(f"| {c} | {o.get('recall')} | {m['recall']} | "
                         f"{o.get('false_alarms')} | {m['false_alarms']} |")
        lines.append("")
    if report.get("legacy"):
        table("Eski ChildGuard verisi (sadece bilgi amaçlı; etiket kalitesi düşük)",
              report["legacy"])
    return "\n".join(lines)


def make_plots(test: pd.DataFrame, test_res: list[dict], report: dict, out: Path) -> list[str]:
    """PR eğrileri ve altın set kategori grafiği. matplotlib yoksa sessizce atlanır."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve
    except ImportError:
        return []
    files = []
    fig, ax = plt.subplots(figsize=(7, 5))
    for label in LABELS:
        known = (test[label] >= 0).values
        y = (test.loc[known, label] >= 0.5).astype(int).values
        if y.sum() == 0 or y.sum() == len(y):
            continue
        p = np.array([r["scores"][label] for r, k in zip(test_res, known) if k])
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=f"{label} (AP={average_precision_score(y, p):.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Test seti — precision/recall eğrileri")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "pr_curves.png", dpi=130)
    plt.close(fig)
    files.append("pr_curves.png")

    cats = list(report["gold_by_category"])
    v4 = [report["gold_by_category"][c]["recall"] for c in cats]
    if any(v is not None for v in v4):
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(cats))
        base = report.get("v3_baseline")
        w = 0.4 if base else 0.6
        if base:
            v3 = [base["by_category"].get(c, {}).get("recall") for c in cats]
            ax.bar(x - w / 2, [v or 0 for v in v3], w, label="v3", color="#a8a29e")
        ax.bar(x + (w / 2 if base else 0), [v or 0 for v in v4], w, label="v4", color="#dc2626")
        ax.set_xticks(x, cats, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Yakalama oranı (recall)")
        ax.set_title("Altın set — kategori bazında zararlı mesaj yakalama")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "gold_recall_by_category.png", dpi=130)
        plt.close(fig)
        files.append("gold_recall_by_category.png")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--compare-v3", action="store_true")
    ap.add_argument("--legacy-sample", type=int, default=5000)
    args = ap.parse_args()

    model = ChildGuardModel.load(args.model_dir or paths.model_dir())
    th = model.thresholds
    report = {"version": model.version, "model": model.config.get("base_model") or "lexical-only",
              "target_recall": model.config.get("target_recall")}
    out = paths.report_dir()
    out.mkdir(parents=True, exist_ok=True)

    test = pd.read_parquet(paths.data_dir() / "test.parquet")
    print(f"Test seti: {len(test):,}", flush=True)
    test_res = model.predict(test["text"].tolist())
    report["test"] = label_metrics(test, test_res, th)

    gold = pd.read_csv(paths.GOLD)
    gold_res = model.predict(gold["text"].tolist())
    report["gold"] = label_metrics(gold, gold_res, th)
    report["gold_by_category"] = gold_by_category(gold, gold_res)
    misses = [{"text": t, "category": c, "score": r["scores"][PRIMARY]}
              for t, c, y, r in zip(gold["text"], gold["category"], gold[PRIMARY], gold_res)
              if (y >= 0.5) != r["harmful"]]
    report["gold_errors"] = misses
    report["warnings"] = []

    # İsteğe bağlı bölümler: hata verirlerse rapor yine de yazılır.
    if args.legacy_sample and not paths.is_lfs_pointer(paths.LEGACY_CSV):
        try:
            from training.sources import load_childguard_legacy
            leg = load_childguard_legacy()
            leg = leg[leg["split"] == "test"].sample(min(args.legacy_sample, len(leg)),
                                                     random_state=0).reset_index(drop=True)
            report["legacy"] = label_metrics(leg, model.predict(leg["text"].tolist()), th)
        except Exception as exc:
            report["warnings"].append(f"Eski veri değerlendirilemedi: {exc}")
    elif args.legacy_sample:
        report["warnings"].append("Eski ChildGuard CSV'si indirilmemiş (LFS); atlandı.")

    if args.compare_v3:
        if paths.is_lfs_pointer(paths.V3_BERT):
            report["warnings"].append("v3 modeli indirilmemiş (LFS); karşılaştırma atlandı.")
        else:
            try:
                print("Eski v3 modeli değerlendiriliyor...", flush=True)
                report["v3_baseline"] = v3_baseline(gold)
            except Exception as exc:
                report["warnings"].append(f"v3 karşılaştırması başarısız: {exc}")

    md = to_markdown(report)
    try:
        for f in make_plots(test, test_res, report, out):
            md += f"\n![{f}]({f})\n"
    except Exception as exc:
        report["warnings"].append(f"Grafikler çizilemedi: {exc}")
    if misses:
        md += "\n## Altın sette hatalı kararlar\n\n| Kategori | Skor | Metin |\n|---|---|---|\n"
        md += "\n".join(f"| {m['category']} | {m['score']:.2f} | {m['text']} |" for m in misses)
        md += "\n"
    if report["warnings"]:
        md += "\n## Uyarılar\n\n" + "\n".join(f"- {w}" for w in report["warnings"]) + "\n"
    (out / "eval_report.md").write_text(md, encoding="utf-8")
    (out / "eval_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    print(md)
    print(f"✅ Rapor: {out / 'eval_report.md'}")


if __name__ == "__main__":
    main()
