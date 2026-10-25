"""Tüm hattı sırayla çalıştırır: veri → sözcüksel → transformer → kalibrasyon → değerlendirme.

    # 1) Önce duman testi (birkaç dakika, ayrı klasörde, gerçek modelle):
    python -m training.run_all --smoke
    # 2) Geçerse tam eğitim (tamamlanan aşamalar atlanır; kesilirse aynı komutla devam):
    python -m training.run_all --profile full

    # GPU'suz hızlı deneme:
    python -m training.run_all --profile small --lexical-only

Çalışma klasörü `CG_WORK` ortam değişkeninden okunur (Colab'da Drive).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from training import paths


def run(*args: str, env: dict | None = None):
    cmd = [sys.executable, "-u", "-m", *args]
    print(f"\n{'=' * 70}\n$ {' '.join(cmd[2:])}\n{'=' * 70}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True, env=env)
    print(f"⏱  {(time.time() - t0) / 60:.1f} dk", flush=True)


def smoke(args) -> None:
    base = paths._work() or paths.ROOT / "data"
    work = base / "smoke"
    env = {**os.environ, "CG_WORK": str(work), "PYTHONIOENCODING": "utf-8"}
    print(f"🔥 DUMAN TESTİ — çalışma klasörü: {work}")
    t0 = time.time()
    run("training.prepare_data", "--profile", "smoke", env=env)
    run("training.train_lexical", "--C", "1.0", env=env)
    if not args.lexical_only:
        extra = ["--model", args.model] if args.model else []
        run("training.train_transformer", "--max-train", "4000", "--max-steps", "30",
            "--fresh", *extra, env=env)
    run("training.calibrate", *(["--no-transformer"] if args.lexical_only else []), env=env)
    run("training.evaluate", "--compare-v3", "--legacy-sample", "500", env=env)

    # Son kontrol: modeli tıpkı backend gibi yükleyip tahmin yap.
    os.environ["CG_WORK"] = str(work)
    from childguard.pipeline import ChildGuardModel
    model = ChildGuardModel.load(paths.model_dir())
    r = model.predict(["kys nobody likes you", "great job on your project!"])
    assert r[0]["harmful"] and r[0]["rule_hits"], "Kural katmanı çalışmadı"
    e = model.explain("you are a loser")
    assert len(e["words"]) == 4, "Açıklama üretilemedi"
    for f in ["config.json", "lexical.joblib"]:
        assert (paths.model_dir() / f).exists(), f"{f} yazılmadı"
    assert (paths.report_dir() / "eval_report.md").exists(), "Rapor yazılmadı"

    meta_path = paths.model_dir() / "transformer" / "training_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sps = meta.get("train_samples_per_second")
        print(f"\nModel: {meta['base_model']} · eğitim hızı ≈ {sps:.0f} örnek/s" if sps else "")
    print(f"\n✅ SMOKE OK — tüm aşamalar çalıştı ({(time.time() - t0) / 60:.1f} dk). "
          f"Tam eğitime geçebilirsiniz.")


def estimate(train_rows: int, epochs: float) -> None:
    meta_path = (paths._work() or paths.ROOT / "data") / "smoke" / "models" / "v4" / \
        "transformer" / "training_meta.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    sps = meta.get("train_samples_per_second")
    if sps:
        val_rows = json.loads((paths.data_dir() / "stats.json").read_text(
            encoding="utf-8"))["val"]["rows"]
        train_min = train_rows * epochs / sps / 60
        # Epoch başına ~3 ara değerlendirme yapılır.
        eval_min = 3 * epochs * val_rows * meta.get("eval_seconds_per_1k", 0) / 1000 / 60
        print(f"⏳ Tahmini transformer süresi: ~{train_min + eval_min:.0f} dk "
              f"(eğitim ~{train_min:.0f} + değerlendirme ~{eval_min:.0f}; "
              f"{train_rows:,} satır × {epochs} epoch, ~{sps:.0f} örnek/s)", flush=True)


def full(args) -> None:
    data, models = paths.data_dir(), paths.model_dir()
    print(f"📁 Veri: {data}\n📁 Model: {models}\n📁 Checkpoint: {paths.checkpoint_dir()}")

    stats = data / "stats.json"
    have_data = stats.exists() and json.loads(stats.read_text(encoding="utf-8")).get(
        "profile") == args.profile
    if args.redo or not have_data:
        run("training.prepare_data", "--profile", args.profile,
            *(["--no-toxigen"] if args.no_toxigen else []))
    else:
        print(f"✓ Veri hazır ({args.profile}), atlanıyor.")
    st = json.loads(stats.read_text(encoding="utf-8"))
    estimate(st["train"]["rows"], float(args.epochs))

    if args.redo or not (models / "lexical.joblib").exists():
        run("training.train_lexical")
    else:
        print("✓ Sözcüksel model hazır, atlanıyor.")

    if not args.lexical_only:
        if args.redo or not (models / "transformer" / "training_meta.json").exists():
            extra = ["--model", args.model] if args.model else []
            run("training.train_transformer", "--epochs", str(args.epochs), *extra,
                *(["--fresh"] if args.redo else []))
        else:
            print("✓ Transformer hazır, atlanıyor.")

    run("training.calibrate", *(["--no-transformer"] if args.lexical_only else []))
    run("training.evaluate", "--compare-v3")
    print(f"\n✅ Bitti. Model: {models}\n   Rapor: {paths.report_dir() / 'eval_report.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Birkaç dakikalık uçtan uca deneme")
    ap.add_argument("--profile", choices=["small", "full"], default="full")
    ap.add_argument("--model", default=None, help="Varsayılan: GPU'ya göre otomatik")
    ap.add_argument("--epochs", default="2")
    ap.add_argument("--lexical-only", action="store_true")
    ap.add_argument("--no-toxigen", action="store_true")
    ap.add_argument("--redo", action="store_true", help="Tamamlanmış aşamaları da baştan yap")
    args = ap.parse_args()
    smoke(args) if args.smoke else full(args)


if __name__ == "__main__":
    main()
