"""Eğitim hattının testleri — Colab'da saatler harcamadan önce hataları yakalamak için.

* Veri kaynakları, gerçek Hugging Face sütun biçimlerini taklit eden sahte
  verilerle test edilir (ağ gerekmez).
* Uçtan uca test, minik rastgele bir RoBERTa ile bütün aşamaları (veri →
  sözcüksel → transformer → kalibrasyon → değerlendirme → yükleme) CPU'da
  saniyeler içinde çalıştırır.
"""

import json
import sys

import numpy as np
import pandas as pd
import pytest

from childguard.labels import LABELS, MISSING
from training import paths, prepare_data, sources

TINY_MODEL = "hf-internal-testing/tiny-random-RobertaForSequenceClassification"

TOXIC = ["you are a pathetic loser", "shut up you idiot", "kill yourself freak",
         "you stupid fat pig", "nobody likes you, go away", "i will hurt you",
         "you are disgusting trash", "what a dumb clown"]
CLEAN = ["have a nice day", "see you at practice", "great job on the test",
         "the movie was fun", "thanks for the help", "i love this song",
         "let's play later", "happy birthday friend"]


# --------------------------------------------------------------------------
# Sahte Hugging Face veri setleri
# --------------------------------------------------------------------------
class FakeSplit:
    def __init__(self, df):
        self.df = df

    def to_pandas(self):
        return self.df.copy()


class FakeDict(dict):
    pass


def fake_civil(n=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        toxic = i % 2 == 0
        text = (TOXIC if toxic else CLEAN)[i % 8] + f" #{seed}-{i}"
        t = rng.uniform(0.5, 1.0) if toxic else rng.uniform(0.0, 0.3)
        rows.append({"text": text, "toxicity": t, "severe_toxicity": t / 3,
                     "obscene": rng.uniform(0, 1) if toxic else 0.0,
                     "threat": 0.8 if "kill" in text or "hurt" in text else 0.0,
                     "insult": t, "identity_attack": 0.6 if i % 10 == 0 else 0.0,
                     "sexual_explicit": 0.7 if i % 12 == 0 else 0.0})
    return pd.DataFrame(rows)


def fake_mhs():
    rows = []
    for cid in range(40):
        text = (TOXIC if cid % 2 else CLEAN)[cid % 8] + f" mhs{cid}"
        for ann in range(3):
            toxic = cid % 2 == 1
            rows.append({"comment_id": cid, "annotator_id": ann, "text": text,
                         "hatespeech": 2 if toxic and ann < 2 else 0,
                         "insult": 4 if toxic else 2, "humiliate": 3 if toxic else 1,
                         "violence": 3 if toxic and cid % 4 == 1 else 0,
                         "target_age_children": cid % 5 == 0,
                         "target_age_teenagers": False})
    return pd.DataFrame(rows)


def fake_toxigen(n=30):
    lewd = ["doesn't contain lewd content", "contains lewd content",
            "might contain lewd content", "disagreement"]
    return pd.DataFrame({
        "text": [f"group statement {i}" for i in range(n)],
        "toxicity_human": [5.0 if i % 3 == 0 else 1.0 for i in range(n)],
        "lewd": [lewd[i % 4] for i in range(n)],
    })


@pytest.fixture
def fake_hf(monkeypatch):
    def load_dataset(name, config=None, split=None, **kw):
        if name == "google/civil_comments":
            return FakeDict(train=FakeSplit(fake_civil(80, 0)),
                            validation=FakeSplit(fake_civil(30, 1)),
                            test=FakeSplit(fake_civil(30, 2)))
        if name == "ucberkeley-dlab/measuring-hate-speech":
            return FakeSplit(fake_mhs())
        if name == "toxigen/toxigen-data":
            return FakeDict(train=FakeSplit(fake_toxigen(30)), test=FakeSplit(fake_toxigen(9)))
        raise AssertionError(name)

    import datasets
    monkeypatch.setattr(datasets, "load_dataset", load_dataset)


# --------------------------------------------------------------------------
# Kaynaklar
# --------------------------------------------------------------------------
def test_civil_comments_schema(fake_hf):
    df = sources.load_civil_comments(max_pos=10, neg_ratio=1.0)
    assert set(df["split"]) == {"train", "val", "test"}
    assert (df["targets_minor"] == MISSING).all()
    for c in LABELS[:-1]:
        assert df[c].between(0, 1).all(), c


def test_mhs_uses_agreement_share_not_mean(fake_hf):
    df = sources.load_measuring_hate_speech()
    assert len(df) == 40                              # annotatör başına değil, yorum başına
    toxic = df[df["text"].str.contains("mhs1$")].iloc[0]
    assert toxic["toxicity"] == pytest.approx(2 / 3)   # 3 annotatörden 2'si "evet"
    clean = df[df["text"].str.contains("mhs0$")].iloc[0]
    assert clean["insult"] == 0.0                     # "2 = kararsız" pozitif sayılmaz
    assert clean["targets_minor"] == 1.0
    assert (df["sexual"] == MISSING).all() and (df["profanity"] == MISSING).all()


def test_toxigen_text_lewd_column(fake_hf):
    df = sources.load_toxigen()
    assert set(df["sexual"].unique()) <= {0.0, 0.5, 1.0}
    assert set(df.loc[df["split"] == "test", "text"]) == {f"group statement {i}" for i in range(9)}
    assert df["toxicity"].between(0, 1).all()


def test_supplement_and_gold_are_valid():
    for path in (paths.SUPPLEMENT, paths.GOLD):
        df = pd.read_csv(path)
        assert set(LABELS) <= set(df.columns), path
        assert df[LABELS].apply(lambda s: s.between(0, 1)).all().all(), path
        assert df["text"].str.strip().str.len().gt(0).all(), path
        assert not df["text"].duplicated().any(), path


def test_legacy_loader_filters_noise(tmp_path):
    csv = tmp_path / "legacy.csv"
    pd.DataFrame({
        "text": ["you are an idiot and a loser", "see wikipedia talk page 12:00 utc please",
                 "ok", "ne kuim zato imaju potrebu da su tako glasni šđč"],
        "actual_class": [1.0, 0.0, 0.0, 1.0], "predicted_class": [1, 0, 0, 1],
        "Age_Group": ["Unknown"] * 4,
    }).to_csv(csv, index=False, encoding="cp1252", errors="replace")
    df = sources.load_childguard_legacy(csv)
    assert df["text"].tolist() == ["you are an idiot and a loser"]


# --------------------------------------------------------------------------
# Birleştirme
# --------------------------------------------------------------------------
def test_build_dedups_across_splits_and_validates(fake_hf, monkeypatch):
    monkeypatch.setitem(prepare_data.PROFILES, "tiny", {"max_pos": 20, "neg_ratio": 1.0})
    df = prepare_data.build("tiny", use_toxigen=True, use_legacy=False)
    prepare_data.validate(df)
    assert not df["key"].duplicated().any()
    assert set(df["source"]) == {"civil_comments", "measuring_hate_speech", "toxigen", "supplement"}


def test_validate_rejects_missing_label():
    df = pd.DataFrame({"text": ["a b"] * 3, "split": ["train", "val", "test"],
                       **{c: [0.0] * 3 for c in LABELS}})
    with pytest.raises(AssertionError):
        prepare_data.validate(df)


# --------------------------------------------------------------------------
# Uçtan uca: tüm aşamalar minik modelle
# --------------------------------------------------------------------------
@pytest.fixture
def work(tmp_path, monkeypatch, fake_hf):
    monkeypatch.setenv("CG_WORK", str(tmp_path))
    monkeypatch.setitem(prepare_data.PROFILES, "tiny", {"max_pos": 30, "neg_ratio": 1.0})
    return tmp_path


def _main(module, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", *argv])
    module.main()


def test_end_to_end_pipeline(work, monkeypatch):
    pytest.importorskip("torch")
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(TINY_MODEL)
    except Exception as exc:  # ağ yoksa
        pytest.skip(f"Minik model indirilemedi: {exc}")

    from training import calibrate, evaluate, train_lexical, train_transformer

    _main(prepare_data, ["--profile", "tiny"], monkeypatch)
    assert (work / "data" / "train.parquet").exists()

    _main(train_lexical, ["--C", "1.0", "--jobs", "2"], monkeypatch)
    assert (work / "models" / "v4" / "lexical.joblib").exists()

    _main(train_transformer, ["--preset", "none", "--model", TINY_MODEL, "--bs", "4",
                              "--grad-accum", "1", "--max-steps", "4", "--fresh"], monkeypatch)
    tdir = work / "models" / "v4" / "transformer"
    meta = json.loads((tdir / "training_meta.json").read_text())
    assert meta["base_model"] == TINY_MODEL

    # Aynı komut tekrar çalışınca checkpoint'ten devam etmeli (çökmeden).
    _main(train_transformer, ["--preset", "none", "--model", TINY_MODEL, "--bs", "4",
                              "--grad-accum", "1", "--max-steps", "4"], monkeypatch)
    # Farklı ayarla devam etmeye çalışmak açık bir hatayla durmalı.
    with pytest.raises(SystemExit):
        _main(train_transformer, ["--preset", "none", "--model", TINY_MODEL, "--bs", "2",
                                  "--grad-accum", "1", "--max-steps", "4"], monkeypatch)

    _main(calibrate, [], monkeypatch)
    config = json.loads((work / "models" / "v4" / "config.json").read_text())
    assert config["uses_transformer"] and set(config["thresholds"]) == set(LABELS)
    for th in config["thresholds"].values():
        assert 0.0 <= th["flag"] <= 1.0

    _main(evaluate, ["--legacy-sample", "0"], monkeypatch)
    report = (work / "reports" / "eval_report.md").read_text(encoding="utf-8")
    assert "Altın set" in report

    from childguard.pipeline import ChildGuardModel
    model = ChildGuardModel.load(work / "models" / "v4", device="cpu")
    assert model.transformer is not None
    r = model.predict(["kys", "hello friend"])
    assert r[0]["harmful"] and set(r[1]["scores"]) == set(LABELS)
    assert len(model.explain("you are a loser")["words"]) == 4


def test_lexical_only_pipeline(work, monkeypatch):
    from training import calibrate, evaluate, train_lexical

    _main(prepare_data, ["--profile", "tiny", "--no-toxigen"], monkeypatch)
    _main(train_lexical, ["--C", "1.0", "--jobs", "1"], monkeypatch)
    _main(calibrate, ["--no-transformer"], monkeypatch)
    _main(evaluate, ["--legacy-sample", "0", "--compare-v3"], monkeypatch)
    report = json.loads((work / "reports" / "eval_report.json").read_text(encoding="utf-8"))
    assert report["model"] == "lexical-only"
    assert "gold_by_category" in report
