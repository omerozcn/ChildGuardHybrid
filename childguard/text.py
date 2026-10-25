"""Metin normalizasyonu — eğitim ve çalışma zamanı için tek kaynak.

İki görünüm üretilir:
  * `for_transformer`: büyük/küçük harf ve noktalama korunur, emojiler
    metne çevrilir, URL ve kullanıcı adları sabit belirteçlere dönüşür.
  * `for_lexical`: ek olarak küçük harfe çevrilir ve gizleme ("st*pid",
    "k1ll", "stuuupid") çözülür; karakter n-gram modeli içindir.
"""

from __future__ import annotations

import re
import unicodedata

try:
    import emoji as _emoji
except ImportError:  # emoji opsiyonel; yoksa emojiler olduğu gibi kalır
    _emoji = None

_URL = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_USER = re.compile(r"(?<!\w)@\w+")
_REPEAT = re.compile(r"(.)\1{2,}")
_SPACE = re.compile(r"\s+")
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
                       "7": "t", "@": "a", "$": "s", "!": "i"})
# Kelime içindeki gizleme karakterleri: "st*pid", "f.u.c.k", "k-i-l-l"
_INNER_MASK = re.compile(r"(?<=\w)[*#._\-](?=\w)")
_ASCII_WORD = re.compile(r"[a-z]+")
# Belirtecin sonundaki noktalama ("stupid!!", "you?") leet çevirisine girmez.
_EDGE = re.compile(r"^(.*?)([^\w@$]*)$")

_EN_COMMON = {
    "the", "you", "and", "to", "a", "is", "are", "i", "it", "of", "in", "that",
    "this", "for", "on", "your", "my", "me", "be", "with", "so", "not", "no",
    "just", "have", "do", "what", "like", "all", "go", "get", "she", "he",
    "they", "we", "was", "at", "if", "but", "can", "will", "one", "up", "out",
    "u", "ur", "lol", "dont", "im", "youre",
}


def base_clean(text: str) -> str:
    """Her iki görünümün ortak adımları."""
    text = unicodedata.normalize("NFKC", str(text))
    text = _URL.sub(" URL ", text)
    text = _USER.sub(" @user ", text)
    if _emoji is not None:
        text = _emoji.demojize(text, delimiters=(" :", ": "))
    text = _REPEAT.sub(r"\1\1", text)          # "stuuupid" -> "stuupid"
    return _SPACE.sub(" ", text).strip()


def for_transformer(text: str) -> str:
    return base_clean(text)


def for_lexical(text: str) -> str:
    text = base_clean(text).lower()
    text = _INNER_MASK.sub("", text)
    # Rakam/simge gizlemesini yalnızca harflerle çevrili belirteçlerde çöz,
    # böylece "2024" gibi sayılar bozulmaz.
    tokens = []
    for tok in text.split(" "):
        if tok != "@user" and any(c.isalpha() for c in tok):
            core, tail = _EDGE.match(tok).groups()
            tok = core.translate(_LEET) + tail
        tokens.append(tok)
    return " ".join(tokens)


def english_ratio(text: str) -> float:
    """Yaygın İngilizce kelimelerin oranı (0–1). Dil uyarısı için kaba bir ölçü."""
    words = _ASCII_WORD.findall(text.lower())
    if not words:
        return 0.0
    non_ascii = sum(1 for c in text if ord(c) > 127 and c.isalpha())
    alpha = sum(1 for c in text if c.isalpha()) or 1
    if non_ascii / alpha > 0.2:
        return 0.0
    return sum(w in _EN_COMMON for w in words) / len(words)


def looks_english(text: str) -> bool:
    words = _ASCII_WORD.findall(text.lower())
    if len(words) < 3:            # çok kısa metinde karar verme
        return True
    return english_ratio(text) >= 0.08
