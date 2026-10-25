"""Etiket şeması.

Her mesaj için çok etiketli (multi-label) olasılıklar üretilir. `toxicity`
genel "zararlı mı?" kararını taşır; diğerleri risk türleridir.
`targets_minor` mesajın bir çocuğu/genci hedef alıp almadığını gösteren
yardımcı (deneysel) sinyaldir.
"""

LABELS: list[str] = [
    "toxicity",       # genel zararlılık
    "insult",         # hakaret, aşağılama, küçümseme
    "threat",         # şiddet tehdidi ve kendine zarar teşviki
    "sexual",         # cinsel içerik
    "identity_hate",  # kimliğe (ırk, din, cinsiyet...) yönelik nefret
    "profanity",      # küfür / müstehcen dil
    "targets_minor",  # hedef bir çocuk veya genç (deneysel)
]

# Ana karar etiketi; risk seviyesi bu etiketin olasılığından türetilir.
PRIMARY = "toxicity"

# Arayüzde gösterilecek Türkçe adlar.
DISPLAY_TR: dict[str, str] = {
    "toxicity": "Zararlı içerik",
    "insult": "Hakaret / aşağılama",
    "threat": "Tehdit / kendine zarar",
    "sexual": "Cinsel içerik",
    "identity_hate": "Kimlik nefreti",
    "profanity": "Küfür",
    "targets_minor": "Çocuğu hedef alıyor",
}

# Veri kaynağında hiç bilgi taşımayan etiketler için kullanılan değer.
# Kayıp fonksiyonu bu hücreleri maskeler (öğrenmeye katmaz).
MISSING = -1.0
