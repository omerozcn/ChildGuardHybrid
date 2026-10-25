import numpy as np

from childguard.labels import LABELS
from childguard.pipeline import Stacker


def test_predict_shape_and_fields(tiny_model):
    out = tiny_model.predict(["you are such a pathetic loser", "great job on your project"])
    assert len(out) == 2
    for r in out:
        assert set(r["scores"]) == set(LABELS)
        assert r["risk_level"] in {"low", "medium", "high"}
        assert all(0.0 <= v <= 1.0 for v in r["scores"].values())


def test_rule_hit_forces_threat_and_harmful(tiny_model):
    r = tiny_model.predict(["kys"])[0]
    assert r["flags"]["threat"]
    assert r["harmful"]
    assert r["rule_hits"][0]["name"] == "self_harm_kys"


def test_language_warning(tiny_model):
    r = tiny_model.predict(["sen çok kötü birisin ve herkes senden nefret ediyor"])[0]
    assert r["language_warning"]


def test_explain_returns_word_and_ngram_weights(tiny_model):
    e = tiny_model.explain("you are a pathetic loser")
    assert [w["token"] for w in e["words"]] == ["you", "are", "a", "pathetic", "loser"]
    assert e["ngrams"]


def test_stacker_handles_missing_lexical_label():
    s = Stacker({l: {"transformer": 1.0, "lexical": 1.0, "bias": 0.0} for l in LABELS})
    p_l = np.full((1, len(LABELS)), np.nan)
    p_t = np.full((1, len(LABELS)), 0.8)
    out = s.combine(p_t, p_l)
    assert np.allclose(out, 0.8)
