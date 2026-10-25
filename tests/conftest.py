import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from childguard.labels import LABELS  # noqa: E402
from childguard.pipeline import ChildGuardModel, LexicalModel  # noqa: E402


@pytest.fixture(autouse=True)
def isolate_from_real_work_dirs(monkeypatch):
    """Testler gerçek çalışma klasörlerine (Colab'da Drive ve yerel checkpoint
    klasörü) asla yazmamalı; ortamdan gelen yol ayarlarını her testte temizle."""
    for var in ("CG_WORK", "CG_CKPT"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(scope="session")
def tiny_model() -> ChildGuardModel:
    """Ek set üzerinde saniyeler içinde eğitilen küçük, yalnızca sözcüksel model."""
    from training.train_lexical import build_vectorizer, fit_soft_lr
    from childguard.text import for_lexical

    df = pd.read_csv(ROOT / "data" / "supplement" / "child_safety_supplement.csv")
    vec = build_vectorizer(5000, 5000)
    vec.set_params(word__min_df=1, char__min_df=1)
    X = vec.fit_transform([for_lexical(t) for t in df["text"]])
    models = {}
    for label in LABELS:
        p = df[label].astype(float).values
        if (p >= 0.5).sum() >= 2:
            models[label] = fit_soft_lr(X, p, C=4.0)
    config = {
        "version": "test",
        "uses_transformer": False,
        "stacker": {l: {"transformer": 0.0, "lexical": 1.0, "bias": 0.0} for l in LABELS},
        "thresholds": {l: {"flag": 0.5, "high": 0.8} for l in LABELS},
    }
    return ChildGuardModel(config, LexicalModel(vec, models), None)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
