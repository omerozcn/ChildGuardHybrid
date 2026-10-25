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
    df = prepare_data.build("tiny", use_toxigen=True, use_legacy=False, supplement_repeat=1)
    prepare_data.validate(df)
    assert not df["key"].duplicated().any()
    assert set(df["source"]) == {"civil_comments", "measuring_hate_speech", "toxigen",
                                 "supplement", "generated"}


def test_own_data_repeated_only_in_train(fake_hf, monkeypatch):
    monkeypatch.setitem(prepare_data.PROFILES, "tiny", {"max_pos": 20, "neg_ratio": 1.0})
    once = prepare_data.build("tiny", True, False, supplement_repeat=1)
    thrice = prepare_data.build("tiny", True, False, supplement_repeat=3)
    own = lambda d, s: ((d["split"] == s) & d["source"].isin(["supplement", "generated"])).sum()
    assert own(thrice, "train") == 3 * own(once, "train")
    assert own(thrice, "val") == own(once, "val") and own(once, "val") > 0
    assert (thrice["source"] != "generated").sum() == (once["source"] != "generated").sum() \
        + 2 * ((once["split"] == "train") & (once["source"] == "supplement")).sum()


def test_profanity_only_comments_are_not_harmful():
    df = pd.DataFrame({
        "text": ["fuck this homework", "fuck you loser", "you are dumb", "nice day"],
        "toxicity": [0.8, 0.9, 0.7, 0.0], "severe_toxicity": [0.1, 0.3, 0.0, 0.0],
        "obscene": [0.9, 0.9, 0.0, 0.0], "insult": [0.1, 0.8, 0.7, 0.0],
        "threat": [0.0] * 4, "identity_attack": [0.0] * 4, "sexual_explicit": [0.0] * 4,
    })
    tox = sources.profanity_adjusted_toxicity(df)
    assert tox[0] < 0.5                      # tek başına küfür → zararsız
    assert tox[1] == pytest.approx(0.9)      # küfür + hakaret → zararlı kalır
    assert tox[2] == pytest.approx(0.7)      # küfürsüz hakaret değişmez
    assert tox[3] == 0.0


def test_generated_data_is_reproducible_and_valid():
    from training import augment

    df = augment.generate()
    saved = pd.read_csv(paths.GENERATED)
    assert df["text"].tolist() == saved["text"].tolist(), \
        "generated.csv güncel değil: python -m training.augment"
    assert df[LABELS].apply(lambda s: s.between(0, 1)).all().all()
    assert not df["text"].str.lower().duplicated().any()
    hard_neg = {"self_deprecation", "victim_report", "idiom", "profanity_only", "support"}
    assert hard_neg <= set(df["category"])
    assert (df.loc[df["category"].isin(hard_neg), "toxicity"] < 0.5).all()
    assert (df.loc[df["category"] == "profanity_only", "profanity"] == 1).all()


def test_choose_raise_rules_on_validation():
    from training.calibrate import choose_raise_rules

    rng = np.random.default_rng(1)
    n = 4000
    y = rng.random(n) < 0.3
    probs = np.zeros((n, len(LABELS)))
    probs[:, LABELS.index("toxicity")] = np.where(y, rng.uniform(0.2, 1, n), rng.uniform(0, 0.4, n))
    # Cinsel içerik: zararlı satırların bir kısmında belirgin, toxicity'nin kaçırdıklarını yakalar.
    sexual = np.where(y & (rng.random(n) < 0.3), rng.uniform(0.6, 1, n), rng.uniform(0, 0.1, n))
    probs[:, LABELS.index("sexual")] = sexual
    # Tehdit: gürültü — zararsız satırlarda da yüksek.
    probs[:, LABELS.index("threat")] = rng.uniform(0, 1, n)
    th = {l: {"flag": 0.5, "high": 0.9} for l in LABELS}
    th["toxicity"] = {"flag": 0.5, "high": 0.9}
    rules = choose_raise_rules(probs, y, th)
    assert rules["sexual"] in ("flag", "high")
    assert rules["threat"] == "none"


def test_run_all_retrains_stale_models_and_resumes(tmp_path, monkeypatch):
    """Drive'da eski veri sürümüyle eğitilmiş modeller varken neyin yeniden çalıştığı."""
    from training import run_all
    from training.prepare_data import DATA_VERSION

    monkeypatch.setenv("CG_WORK", str(tmp_path))
    monkeypatch.setenv("CG_CKPT", str(tmp_path / "ckpt"))
    data, models = paths.data_dir(), paths.model_dir()
    calls = []

    def fake_run(*args, env=None):
        calls.append(args)
        if args[0] == "training.prepare_data":
            data.mkdir(parents=True, exist_ok=True)
            (data / "stats.json").write_text(json.dumps(
                {"profile": "full", "data_version": DATA_VERSION, "train": {"rows": 10},
                 "val": {"rows": 5}}))
        elif args[0] == "training.train_lexical":
            (models / "lexical.joblib").write_text("x")
        elif args[0] == "training.train_transformer":
            (models / "transformer").mkdir(parents=True, exist_ok=True)
            (models / "transformer" / "training_meta.json").write_text("{}")

    monkeypatch.setattr(run_all, "run", fake_run)
    args = type("A", (), {"profile": "full", "redo": False, "lexical_only": False,
                          "no_toxigen": False, "model": None, "epochs": "2"})()

    # 1) v4.0 durumu: eski veri + damgasız bitmiş modeller → hepsi yeniden, transformer sıfırdan
    data.mkdir(parents=True)
    (data / "stats.json").write_text(json.dumps({"profile": "full", "train": {"rows": 9}}))
    (models / "transformer").mkdir(parents=True)
    (models / "lexical.joblib").write_text("old")
    (models / "transformer" / "training_meta.json").write_text("{}")
    run_all.full(args)
    names = [c[0] for c in calls]
    assert names == ["training.prepare_data", "training.train_lexical",
                     "training.train_transformer", "training.calibrate", "training.evaluate"]
    assert "--fresh" in calls[2]
    assert (models / "transformer" / "training_meta.stale.json").exists()

    # 2) Her şey güncel → yalnızca kalibrasyon ve değerlendirme
    calls.clear()
    run_all.full(args)
    assert [c[0] for c in calls] == ["training.calibrate", "training.evaluate"]

    # 3) Transformer yarıda kaldı (meta yok, damga yok) → --fresh OLMADAN devam
    calls.clear()
    (models / "transformer" / "training_meta.json").unlink()
    (models / "transformer.data_version").unlink()
    run_all.full(args)
    assert [c[0] for c in calls] == ["training.train_transformer", "training.calibrate",
                                     "training.evaluate"]
    assert "--fresh" not in calls[0]


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
    monkeypatch.setenv("CG_CKPT", str(tmp_path / "ckpt"))
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
    assert list((work / "ckpt").glob("checkpoint-*")), "checkpoint test klasörüne yazılmalı"
    tdir = work / "models" / "v4" / "transformer"
    meta = json.loads((tdir / "training_meta.json").read_text())
    assert meta["base_model"] == TINY_MODEL
    assert meta["train_samples_per_second"] > 0       # süre tahmini için saf adım hızı

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


def test_threshold_policy_falls_back_when_precision_too_low():
    from training.calibrate import pick_thresholds

    rng = np.random.default_rng(0)
    # Nadir ve zor ayrışan etiket: %90 recall ancak çok düşük precision'la mümkün.
    y = np.r_[np.ones(100), np.zeros(5000)].astype(int)
    p = np.r_[rng.beta(4, 2, 100), rng.beta(1.5, 4, 5000)]
    naive = pick_thresholds(y, p, "threat", target_recall=0.90, min_precision=0.0)
    t = pick_thresholds(y, p, "threat", target_recall=0.90, min_precision=0.40)
    assert naive["policy"].startswith("recall") and naive["val_precision"] < 0.40
    assert t["policy"].startswith("f2")
    assert t["flag"] >= naive["flag"] and t["val_precision"] >= naive["val_precision"]

    # Kolay ayrışan etiket: recall hedefi yüksek precision'la tutturulur, politika değişmez.
    p_easy = np.r_[rng.beta(8, 2, 100), rng.beta(2, 8, 5000)]
    t2 = pick_thresholds(y, p_easy, "threat", target_recall=0.90, min_precision=0.40)
    assert t2["policy"].startswith("recall") and t2["val_recall"] >= 0.90
    assert t2["high"] >= t2["flag"]


def test_corrupt_cached_weights_trigger_one_redownload():
    pytest.importorskip("torch")
    from training.train_transformer import load_with_retry

    calls = []

    def loader(name, **kw):
        calls.append(kw.get("force_download", False))
        if not kw.get("force_download"):
            raise RuntimeError("Error while deserializing header: header too large")
        return "model"

    assert load_with_retry(loader, "roberta-large") == "model"
    assert calls == [False, True]

    def unrelated(name, **kw):
        raise ValueError("num_labels uyuşmuyor")

    with pytest.raises(ValueError):
        load_with_retry(unrelated, "roberta-large")


def test_checkpoint_dir_can_live_outside_work(tmp_path, monkeypatch):
    monkeypatch.setenv("CG_WORK", str(tmp_path / "drive"))
    assert paths.checkpoint_dir() == tmp_path / "drive" / "checkpoints"
    monkeypatch.setenv("CG_CKPT", str(tmp_path / "local_ckpt"))
    assert paths.checkpoint_dir() == tmp_path / "local_ckpt"
    assert paths.model_dir() == tmp_path / "drive" / "models" / "v4"


def test_lexical_only_pipeline(work, monkeypatch):
    from training import calibrate, evaluate, train_lexical

    _main(prepare_data, ["--profile", "tiny", "--no-toxigen"], monkeypatch)
    _main(train_lexical, ["--C", "1.0", "--jobs", "1"], monkeypatch)
    _main(calibrate, ["--no-transformer"], monkeypatch)
    _main(evaluate, ["--legacy-sample", "0", "--compare-v3"], monkeypatch)
    report = json.loads((work / "reports" / "eval_report.json").read_text(encoding="utf-8"))
    assert report["model"] == "lexical-only"
    assert "gold_by_category" in report
