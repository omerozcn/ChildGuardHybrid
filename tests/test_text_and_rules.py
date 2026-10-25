import pandas as pd
import pytest

from childguard.rules import apply_floors, find_hits
from childguard.text import for_lexical, for_transformer, looks_english

from conftest import ROOT


@pytest.mark.parametrize("raw, expected", [
    ("ur such a l0ser", "ur such a loser"),
    ("st00pid", "stoopid"),
    ("sooooo dumbbbb", "soo dumbb"),
    ("f.u.c.k off", "fuck off"),
    ("you're stupid!!", "you're stupid!!"),
    ("@mike see http://x.com", "@user see url"),
    ("score was 2024 to 13", "score was 2024 to 13"),
])
def test_for_lexical(raw, expected):
    assert for_lexical(raw) == expected


def test_for_transformer_keeps_case_and_punctuation():
    assert for_transformer("You are SO Smart!!!") == "You are SO Smart!!"


def test_turkish_and_slavic_text_flagged_as_non_english():
    assert not looks_english("sen çok aptalsın ve kimse seni sevmiyor")
    assert not looks_english("ne kuim zato imaju potrebu da su tako glasni")
    assert looks_english("you are such a loser and everyone knows it")


@pytest.mark.parametrize("text", [
    "kys loser", "K Y S", "just k1ll urself already", "you should just die",
    "nobody would miss you if you were gone", "the world would be better without you",
    "you are not ok, kill yourself", "I'm gonna beat you up after school",
    "go jump off a bridge", "slit your wrists", "i will kill you",
])
def test_rules_catch_severe_messages(text):
    assert find_hits(text), text


@pytest.mark.parametrize("text", [
    "be careful not to hurt yourself on the slide",
    "please don't kill yourself over this test",
    "let's go hang out later", "I'm gonna beat you at this game lol",
    "I just died laughing", "my keys are in the car",
    "let's go jump off the diving board", "this song kills",
])
def test_rules_ignore_benign_messages(text):
    assert not find_hits(text), text


def test_apply_floors_only_raises():
    hits = find_hits("kys")
    out = apply_floors({"toxicity": 0.95, "threat": 0.1, "insult": 0.2}, hits)
    assert out["toxicity"] == 0.95          # model daha yüksekse korunur
    assert out["threat"] >= 0.9             # kural tabanı uygulanır
    assert out["insult"] == 0.2             # ilgisiz etiket değişmez


def test_rules_have_no_false_alarms_on_benign_gold_examples():
    gold = pd.read_csv(ROOT / "data" / "gold" / "gold_test.csv")
    benign = gold[gold["toxicity"] < 0.5]
    fired = [t for t in benign["text"] if find_hits(t)]
    assert fired == []


def test_gold_and_supplement_do_not_overlap():
    gold = pd.read_csv(ROOT / "data" / "gold" / "gold_test.csv")
    sup = pd.read_csv(ROOT / "data" / "supplement" / "child_safety_supplement.csv")
    norm = lambda s: {" ".join(for_lexical(t).split()) for t in s}
    assert not (norm(gold["text"]) & norm(sup["text"]))
