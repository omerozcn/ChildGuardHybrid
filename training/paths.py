"""Eğitim hattının bütün dosya yolları.

`CG_WORK` ortam değişkeni ayarlıysa her şey onun altına yazılır (Colab'da
Google Drive klasörü). Ayarlı değilse depo içindeki varsayılanlar kullanılır.

    CG_WORK/
      data/          birleştirilmiş train/val/test parquet
      checkpoints/   transformer checkpoint'leri (devam ettirme için)
      models/v4/     son model (config.json, lexical.joblib, transformer/)
      reports/       değerlendirme raporu ve grafikler
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _work() -> Path | None:
    w = os.environ.get("CG_WORK")
    return Path(w) if w else None


def data_dir() -> Path:
    w = _work()
    return w / "data" if w else ROOT / "data" / "processed"


def checkpoint_dir() -> Path:
    w = _work()
    return w / "checkpoints" if w else ROOT / "data" / "checkpoints"


def model_dir() -> Path:
    w = _work()
    return w / "models" / "v4" if w else ROOT / "final_models" / "v4"


def report_dir() -> Path:
    w = _work()
    return w / "reports" if w else ROOT / "reports"


GOLD = ROOT / "data" / "gold" / "gold_test.csv"
SUPPLEMENT = ROOT / "data" / "supplement" / "child_safety_supplement.csv"
LEGACY_CSV = ROOT / "data" / "raw" / "ChildGuard_Cleaned.csv"
V3_BERT = ROOT / "final_models" / "bert_general"


def is_lfs_pointer(path: Path) -> bool:
    """Git LFS dosyası indirilmemişse yerinde küçük bir metin işaretçisi kalır."""
    try:
        if path.is_dir():
            path = next(path.glob("*.safetensors"), path / "_none_")
        if not path.exists():
            return True
        if path.stat().st_size > 1024:
            return False
        return path.read_bytes()[:40].startswith(b"version https://git-lfs")
    except OSError:
        return True
