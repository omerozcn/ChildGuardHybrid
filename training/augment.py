"""Çocuk güvenliği için şablon tabanlı ek eğitim verisi (v4.2).

v4.1 değerlendirmesinde modelin zayıf kaldığı türleri hedefler:
  * örtük küçümseme / çocuklaştırma ("aww the toddler wants attention")
  * dışlama
  * grooming ve cinsel davet (açık içerik yok; gizlilik, buluşma, fotoğraf isteği)
  * örtük kendine zarar teşviki
ve aşırı alarm veren zor negatifleri:
  * kendini kötüleme ("i'm such a dork, i lost my charger")
  * mağdurun anlattığı olaylar ("a kid in class called me fat and i told my teacher")
  * deyimler ("that trick was sick", "you crushed it on stage")
  * tek başına küfürlü ünlemler (proje kararı: zararsız, yalnızca küfür)
  * destek mesajları

Ek ifadeler (lol, emoji, noktalama, büyük harf) pozitif ve negatif kategorilere
AYNI dağılımla eklenir; model bunları etiket ipucu olarak öğrenemez.

Altın test setiyle (data/gold) birebir örtüşme testle engellenir. Şablonlar
altın setteki kalıplardan bilerek farklı yazılmıştır; yine de aynı kişi
tarafından yazıldıkları için altın set sonuçları iyimser olabilir — bağımsız
bir test seti önerilir (docs/RESEARCH.md).

    python -m training.augment          # data/supplement/generated.csv üretir
"""

from __future__ import annotations

import itertools
import random

import pandas as pd

from childguard.labels import LABELS
from training import paths

OUT = paths.ROOT / "data" / "supplement" / "generated.csv"


def L(**kw) -> dict:
    return {l: float(kw.get(l, 0.0)) for l in LABELS}


# (kategori, etiketler, en fazla örnek, [(çerçeve, {yuva: seçenekler})])
CATEGORIES: list[tuple[str, dict, int, list[tuple[str, dict]]]] = [
    ("implicit_infantilizing", L(toxicity=0.9, insult=0.9, targets_minor=1), 1100, [
        ("{opener} the {child} {action}", {
            "opener": ["aww", "awww,", "oh look,", "omg", "wow,", "lmao"],
            "child": ["baby", "little kid", "toddler", "fifth grader", "middle schooler",
                      "little one", "preschooler", "kiddo", "tiny human", "smol child"],
            "action": ["has feelings again", "is trying to talk to the grown ups",
                       "thinks they're smart", "is having a meltdown", "wants attention",
                       "is mad because they lost", "is typing in all caps again",
                       "needs their blankie", "wants a juice box",
                       "found the keyboard", "learned to type"]}),
        ("aren't you supposed to be {place}{tail}", {
            "place": ["in bed", "doing your spelling worksheet", "at daycare",
                      "playing with your blocks", "coloring", "with your babysitter"],
            "tail": ["", " little guy", " kid", " right now", " buddy"]}),
        ("run back to {kidplace}{tail}", {
            "kidplace": ["your sandbox", "the kids table", "your toy cars",
                         "your coloring book", "your playpen", "story time"],
            "tail": ["", ", nobody asked", ", kid", " lil bro", " now"]}),
        ("{praise} for a {kidword}", {
            "praise": ["cute take", "nice try", "big words", "adorable argument",
                       "impressive spelling", "good effort", "so smart", "very brave post"],
            "kidword": ["ten year old", "kid", "baby", "seventh grader", "toddler", "child",
                        "little boy", "little girl", "middle schooler"]}),
        ("{nobody} {cares} what a {kidword} {says}", {
            "nobody": ["nobody", "literally nobody", "no one here"],
            "cares": ["cares", "asked", "wants to hear"],
            "kidword": ["nine year old", "kid", "baby", "sixth grader", "child", "tween"],
            "says": ["thinks", "says", "has to say", "posts"]}),
        ("{ask} {helper} {task}", {
            "ask": ["maybe ask", "go ask", "should we wait for"],
            "helper": ["your mommy", "your daddy", "your big sister", "your teacher",
                       "your nanny"],
            "task": ["to help you read this", "to tie your shoes first",
                     "to explain the big words", "to type it for you",
                     "to tuck you in"]}),
        ("{intro} you {babyverb}", {
            "intro": ["lol", "bro", "dude", "honestly", "omg", ""],
            "babyverb": ["type like a toddler", "argue like a baby", "sound like a preschooler",
                         "cry like a little kid", "act like you're five",
                         "write like a second grader", "throw tantrums like a baby"]}),
    ]),
    ("exclusion", L(toxicity=1, insult=0.9), 600, [
        ("{nobody} {you_where}{tail}", {
            "nobody": ["nobody wants", "no one wants", "we don't want", "none of us want"],
            "you_where": ["you at our table", "you in the group", "you at the party",
                          "you in this chat", "you around",
                          "to be your partner", "you in our game", "you on the bus seat"],
            "tail": ["", ", just leave", ", go away", ", get lost", ", seriously", " ever"]}),
        ("{who} {made} without you {why}", {
            "who": ["we", "everyone", "the whole class", "all of us"],
            "made": ["made a new group chat", "planned the sleepover", "started a new club",
                     "went to the movies", "made the team photo"],
            "why": ["because you're annoying", "because you ruin everything", "lol",
                    "because nobody likes you", "on purpose", "and it was better"]}),
        ("you always {alone}, {gee}", {
            "alone": ["sit by yourself", "eat by yourself", "get picked last",
                      "have nobody to talk to", "walk home alone", "end up without a partner"],
            "gee": ["wonder why", "no surprise", "makes sense", "lol",
                    "everyone knows why", "and you deserve it"]}),
    ]),
    ("grooming_secrecy", L(toxicity=1, sexual=0.6, threat=0.2, targets_minor=0.5), 450, [
        ("{who} can't know {thing}{tail}", {
            "who": ["your mom", "your parents", "your dad", "your teacher", "your friends",
                    "nobody at home"],
            "thing": ["we talk", "about us", "about my video calls with you",
                      "about the gifts i send you", "where you go after school",
                      "that i'm older"],
            "tail": ["", " 😉", ", got it?", ", they wouldn't understand",
                     ", they'd ruin everything"]}),
        ("{lets} keep {this} {secret}", {
            "lets": ["let's", "we should", "promise to", "you have to"],
            "this": ["this", "our talks", "what we do", "our friendship"],
            "secret": ["our little secret", "between us", "a secret from your parents",
                       "hidden from everyone"]}),
        ("delete {msgs} {after}", {
            "msgs": ["our messages", "this chat", "my pics", "our conversation"],
            "after": ["after you read them", "so your parents don't see",
                      "before your mom checks your phone", "every night", "right away ok"]}),
        ("you're so {mature} for your age{tail}", {
            "mature": ["mature", "grown up", "pretty", "special"],
            "tail": [", i like that", ", we should meet", ", can we video chat?",
                     ", not like other kids", " 😘"]}),
        ("{wanna} meet {where} {alone}", {
            "wanna": ["wanna", "do you want to", "we should", "come and"],
            "where": ["up after school", "at the park", "at my place", "somewhere quiet",
                      "near your school"],
            "alone": ["without telling your parents", "alone", "just us two",
                      "and don't bring anyone"]}),
    ]),
    ("sexual_solicitation", L(toxicity=1, sexual=1, targets_minor=0.3), 450, [
        ("{send} me a {pic} {ofyou}{tail}", {
            "send": ["send", "can you send", "just send", "dm"],
            "pic": ["pic", "photo", "selfie", "video"],
            "ofyou": ["of you in bed", "without clothes", "of your body",
                      "of you when you're alone", "in your pajamas",
                      "of you in your room alone"],
            "tail": ["", " please", " 😏", ", nobody will see", " i won't show anyone"]}),
        ("{turn} on your camera and {act}", {
            "turn": ["turn", "put", "switch"],
            "act": ["show me your body", "get in bed", "lock your door first",
                    "sit closer to me"]}),
        ("{wanna} {sexact}{tail}", {
            "wanna": ["wanna", "do you want to", "we could"],
            "sexact": ["do something naughty", "send spicy pics", "get freaky on cam",
                       "talk dirty", "play a grown up game"],
            "tail": ["", " tonight", ", it's fun", " 😉", " with me"]}),
    ]),
    ("implicit_self_harm", L(toxicity=1, threat=1, insult=0.7), 400, [
        ("{maybe} you should {sh}{tail}", {
            "maybe": ["maybe", "honestly", "seriously", "tbh", "imo"],
            "sh": ["just give up on life", "stop existing", "disappear for good",
                   "go to sleep and never wake up", "not wake up tomorrow",
                   "take a long walk off a short bridge"],
            "tail": ["", ", nobody would notice", ", do everyone a favor", " lol",
                     ", it'd be better"]}),
        ("{who} would {react} if you {gone}", {
            "who": ["no one", "nobody", "not a single person"],
            "react": ["notice", "care", "cry", "even remember you", "come looking"],
            "gone": ["were gone", "vanished", "died", "never came back"]}),
        ("{everyone} {better} without you {tail}", {
            "everyone": ["everyone's life", "your family", "the class", "this world"],
            "better": ["would be better", "would be happier", "would improve"],
            "tail": ["", "honestly", "just saying", "think about it"]}),
    ]),
    ("self_deprecation", L(toxicity=0.05), 350, [
        ("{lost} {item} again, {selfroast}{tail}", {
            "lost": ["forgot", "lost", "dropped", "broke"],
            "item": ["my keys at home", "my homework on the bus", "my phone charger",
                     "my water bottle", "my jacket at practice", "my headphones"],
            "selfroast": ["i'm hopeless", "what a dork i am", "classic me",
                          "my brain is gone", "i'm the worst at remembering stuff",
                          "dummy me"],
            "tail": ["", " 😅", " haha", " smh"]}),
        ("{my} brain is {mush} today{tail}", {
            "my": ["my", "ugh my", "lol my"],
            "mush": ["mush", "fried", "broken", "on vacation", "potato mode"],
            "tail": ["", ", i can't do math", " lol", " 😩"]}),
        ("i'm so {bad} at {thing}{tail}", {
            "bad": ["bad", "terrible", "awful", "trash"],
            "thing": ["drawing", "math", "this game", "volleyball", "spelling", "cooking"],
            "tail": ["", " lol", " but i'm trying", ", help", " 😂"]}),
    ]),
    ("victim_report", L(toxicity=0.1), 300, [
        ("{who} called me {word} {today} and {after}", {
            "who": ["a kid in my class", "my brother", "someone online", "a girl at school",
                    "this boy on the bus"],
            "word": ["a loser", "fat", "ugly", "a crybaby", "a freak", "weird"],
            "today": ["today", "again", "at recess", "in the group chat"],
            "after": ["i told my teacher", "it hurt", "i don't know what to do",
                      "i want it to stop", "i blocked them"]}),
        ("{someone} keeps {bully} me{tail}", {
            "someone": ["someone", "a boy in my class", "an older kid", "a girl online"],
            "bully": ["making fun of", "pushing", "laughing at", "messaging mean stuff to",
                      "following"],
            "tail": ["", ", what should i do?", " and i'm scared", ", can you help?"]}),
    ]),
    ("idiom", L(toxicity=0.0), 600, [
        ("that {thing} was {slang}{tail}", {
            "thing": ["goal", "dance", "drawing", "song", "magic trick", "concert",
                      "bike jump", "science fair project"],
            "slang": ["sick", "insane", "crazy good", "fire", "nuts", "killer", "wild"],
            "tail": ["", "!", " bro", " 🔥", " lol"]}),
        ("you {crushed} {where}{tail}", {
            "crushed": ["crushed it", "slayed", "smashed it", "nailed it", "owned it"],
            "where": ["at the recital", "in the match", "on that quiz", "on stage", "today",
                      "at tryouts"],
            "tail": ["", "!", " 👏", ", proud of you", " fr"]}),
        ("{he} is an absolute {beast} at {sport}{tail}", {
            "he": ["she", "my cousin", "our goalie", "my sister"],
            "beast": ["machine", "monster", "legend", "wizard"],
            "sport": ["soccer", "chess", "swimming", "fortnite", "math", "piano"],
            "tail": ["", "!", " lol", " no cap"]}),
        ("{this} {murdered} {victim}{tail}", {
            "this": ["the final exam", "that hike", "leg day", "the spelling bee"],
            "murdered": ["absolutely destroyed", "wrecked", "murdered"],
            "victim": ["me", "my whole class", "my legs", "everyone"],
            "tail": ["", " 😂", " lmao", " haha"]}),
        ("i'm {dying} to {want}{tail}", {
            "dying": ["dying", "dead set on trying", "literally dying"],
            "want": ["see the new movie", "try that pizza place", "go to the beach",
                     "play the new game"],
            "tail": ["", "!", " 😭", " lol"]}),
    ]),
    ("profanity_only", L(toxicity=0.15, profanity=1), 400, [
        ("{exclaim} {situation}{tail}", {
            "exclaim": ["holy crap", "damn", "what the hell", "oh shit", "fuck yeah",
                        "shit", "hell yeah", "fucking finally"],
            "situation": ["we won", "my phone died", "that test was hard",
                          "i'm late again", "it's snowing", "the wifi is down",
                          "summer break starts tomorrow", "the bus left without me"],
            "tail": ["", "!", " lol", " 😂", " bro"]}),
    ]),
    ("support", L(toxicity=0.0), 250, [
        ("it's not your fault when {they} {hurt} you{tail}", {
            "they": ["someone", "anyone", "those kids", "people online"],
            "hurt": ["tease", "bully", "threaten", "say mean things to"],
            "tail": ["", ", you can always talk to me", ", come sit with us",
                     ", let's tell a grown up together"]}),
        ("{you} {matter}{tail}", {
            "you": ["you", "hey, you", "remember you"],
            "matter": ["matter", "are not alone", "deserve kindness", "are important to us"],
            "tail": ["", " ❤️", ", always", ", talk to us anytime"]}),
    ]),
]

# Bütün kategorilere aynı olasılıklarla uygulanır (etiket sızıntısı olmasın).
_SUFFIX = ["", "", "", "", " lol", " lmao", " 😂", " 😭", "!!", "...", " fr", " tbh"]


def _render(frame: str, slots: dict) -> list[str]:
    keys = list(slots)
    out = []
    for combo in itertools.product(*(slots[k] for k in keys)):
        text = frame.format(**dict(zip(keys, combo)))
        out.append(" ".join(text.split()))
    return out


def _vary(text: str, rng: random.Random) -> str:
    r = rng.random()
    if r < 0.25:
        text = text[:1].upper() + text[1:]
    elif r < 0.30:
        text = text.upper()
    return text + rng.choice(_SUFFIX)


def generate(seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows, seen = [], set()
    for category, labels, limit, frames in CATEGORIES:
        pool = [t for frame, slots in frames for t in _render(frame, slots)]
        rng.shuffle(pool)
        kept = 0
        for base in pool:
            text = _vary(base, rng)
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append({"text": text, "category": category, **labels})
            kept += 1
            if kept >= limit:
                break
    return pd.DataFrame(rows, columns=["text", "category", *LABELS])


def main():
    df = generate()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df["category"].value_counts().to_string())
    print(f"✅ {len(df):,} satır → {OUT}")


if __name__ == "__main__":
    main()
