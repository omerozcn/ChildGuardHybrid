import logging
import os
import sys
from pathlib import Path

# Çekirdek paket (childguard/) depo kökünde; Docker'da PYTHONPATH ile gelir.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from childguard.pipeline import ChildGuardModel  # noqa: E402

logger = logging.getLogger(__name__)
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "final_models" / "v4"))

_model: ChildGuardModel | None = None
_error: str | None = None


def load() -> bool:
    global _model, _error
    try:
        _model = ChildGuardModel.load(MODEL_DIR, device=os.getenv("DEVICE"))
        _error = None
        logger.info("Model yüklendi: v%s (%s)", _model.version,
                    _model.config.get("base_model") or "lexical-only")
        return True
    except FileNotFoundError as exc:
        _error = (f"Model bulunamadı ({MODEL_DIR}). Önce eğitimi çalıştırın: "
                  f"python -m training.run_all")
        logger.error("%s — %s", _error, exc)
    except Exception as exc:
        _error = f"Model yüklenemedi: {exc}"
        logger.exception(_error)
    _model = None
    return False


def get() -> ChildGuardModel | None:
    return _model


def error() -> str | None:
    return _error
