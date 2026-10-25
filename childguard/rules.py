"""Yüksek kesinlikli kural katmanı.

Çocuk güvenliğinde en kritik kalıplar (kendine zarar teşviki, doğrudan
tehdit) için modelden bağımsız, deterministik bir güvenlik ağı. Bir kural
eşleşirse ilgili etiketin olasılığı en az `floor` değerine çekilir; model
daha yüksek bir değer verdiyse o korunur. Eşleşmeler yanıtta açıkça
raporlanır, böylece karar açıklanabilir kalır.

Kurallar `text.for_lexical` çıktısı üzerinde çalışır (küçük harf, gizleme
çözülmüş), bu yüzden "K Y S", "k1ll urself" gibi yazımları da yakalar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from childguard.text import for_lexical


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern
    labels: tuple[str, ...]
    floor: float
    description: str


def _r(p: str) -> re.Pattern:
    return re.compile(p, re.IGNORECASE)


_YOU = r"(you|u|ya)"
_YOURSELF = r"(yourself|urself|ur self|your self|yoself|yourselves)"

RULES: list[Rule] = [
    Rule("self_harm_kys", _r(r"\bk\s?y\s?s\b"),
         ("toxicity", "threat"), 0.9, "Kendini öldür kısaltması (kys)"),
    Rule("self_harm_direct",
         _r(rf"\b(kill|hang|end|off|unalive|hurt|cut|shoot|drown)\s+{_YOURSELF}\b"),
         ("toxicity", "threat"), 0.9, "Doğrudan kendine zarar verme çağrısı"),
    Rule("self_harm_go_die",
         _r(r"\b(go|just|pls|please|plz)\s+(die|jump off (a |the )?(bridge|cliff|building|roof))\b"),
         ("toxicity", "threat"), 0.85, "Ölmesini isteme"),
    Rule("self_harm_methods",
         _r(r"\b(slit|cut)\s+(your|ur)\s+(wrists?|throat)\b|\bdrink\s+bleach\b"),
         ("toxicity", "threat"), 0.9, "Kendine zarar yöntemi önerme"),
    Rule("self_harm_implicit",
         _r(rf"\b(no ?one|nobody)\s+(would|will)\s+(miss|care if)\s+{_YOU}\b"
            rf"|\b(world|everyone)\s+(would be|is|will be)\s+better\s+(off\s+)?without\s+{_YOU}\b"
            rf"|\b{_YOU}\s+should\s+(just\s+)?(die|not exist|disappear forever)\b"),
         ("toxicity", "threat", "insult"), 0.8, "Örtük kendine zarar teşviki"),
    Rule("direct_threat",
         _r(rf"\bi('ll| will| am going to| am gonna| gonna|'?m gonna|'?m going to)\s+"
            rf"((kill|hurt|stab|shoot)\s+{_YOU}\b|beat\s+{_YOU}\s+up\b"
            rf"|find\s+{_YOU}\s+and\s+(kill|hurt))"),
         ("toxicity", "threat"), 0.7, "Doğrudan şiddet tehdidi"),
]

# Eşleşmeden hemen önce bu kelimeler varsa kural tetiklenmez:
# "be careful not to hurt yourself", "don't kill yourself over it".
_NEGATION = re.compile(
    r"\b(don'?t|do not|dont|never|not|careful|stop|didn'?t|won'?t|wouldn'?t|"
    r"shouldn'?t|please don'?t|nobody should|no one should)\s+(\w+\s+)?$")
# Aktarılan söz: "if someone tells you to hurt yourself, report them",
# "why do people say kys online?"
_REPORTED = re.compile(
    r"\b(tell|tells|told|telling|ask|asks|asked)\s+(you|u|someone|people|anyone|me)\s+to\s+$"
    r"|\b(say|says|said|saying|type|types|typed|write|writes|wrote)\s+[\"']?$")


@dataclass(frozen=True)
class RuleHit:
    name: str
    description: str
    labels: tuple[str, ...]
    floor: float
    match: str


def find_hits(text: str) -> list[RuleHit]:
    norm = for_lexical(text)
    hits = []
    for rule in RULES:
        for m in rule.pattern.finditer(norm):
            before = norm[max(0, m.start() - 40):m.start()]
            if _NEGATION.search(before) or _REPORTED.search(before):
                continue
            hits.append(RuleHit(rule.name, rule.description, rule.labels,
                                rule.floor, m.group(0)))
            break
    return hits


def apply_floors(probs: dict[str, float], hits: list[RuleHit]) -> dict[str, float]:
    out = dict(probs)
    for hit in hits:
        for label in hit.labels:
            if label in out:
                out[label] = max(out[label], hit.floor)
    return out
