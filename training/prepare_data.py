"""Tüm kaynakları birleştirip eğitim/doğrulama/test tablolarını üretir.

Kullanım:
    python -m training.prepare_data --profile smoke   # duman testi (birkaç dakika)
    python -m training.prepare_data --profile small   # yerel GPU / hızlı deneme
    python -m training.prepare_data --profile full    # Colab / tam eğitim

Çıktı: <data_dir>/{train,val,test}.parquet ve <data_dir>/stats.json
"""

from __future__ import annotations

import argparse
import hashlib
import json

import pandas as pd

from childguard.labels import LABELS, MISSING
from childguard.text import for_lexical
from training import paths, sources

# Veri tanımı değiştikçe artırılır; run_all eski veriyi ve ona bağlı modelleri
# bu damgaya bakarak yeniden üretir. 4.2: küfür etiketi + üretilmiş ek veri.
DATA_VERSION = "4.2"

PROFILES = {
    # max_pos: Civil Comments eğitim kümesinden alınacak en fazla pozitif örnek
    # max_rows: bölme başına satır üst sınırı (yalnızca duman testinde)
    "smoke": {"max_pos": 1_000, "neg_ratio": 1.5, "max_rare": 1_000,
              "max_rows": {"train": 4_000, "val": 1_500, "test": 1_500}},
    "small": {"max_pos": 15_000, "neg_ratio": 2.0},
    "full": {"max_pos": 120_000, "neg_ratio": 2.5},
}


def _key(text: str) -> str:
    return hashlib.md5(" ".join(for_lexical(text).split()).encode()).hexdigest()


def build(profile: str, use_toxigen: bool, use_legacy: bool,
          supplement_repeat: int = 3) -> pd.DataFrame:
    cfg = PROFILES[profile]
    frames = []
    print("• Civil Comments...", flush=True)
    frames.append(sources.load_civil_comments(cfg["max_pos"], cfg["neg_ratio"],
                                              max_rare=cfg.get("max_rare")))
    print("• Measuring Hate Speech...", flush=True)
    frames.append(sources.load_measuring_hate_speech())
    if use_toxigen:
        print("• ToxiGen (annotated)...", flush=True)
        frames.append(sources.load_toxigen())
    print("• Çocuk güvenliği ek seti + üretilmiş veri...", flush=True)
    frames.append(sources.load_supplement())
    frames.append(sources.load_generated())
    if use_legacy:
        print("• Eski ChildGuard verisi (sadece toxicity)...", flush=True)
        frames.append(sources.load_childguard_legacy())

    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.split().str.len().between(2, 400)]

    # Aynı metin farklı bölmelere düşmesin: test > val > train önceliği.
    df["key"] = df["text"].map(_key)
    order = {"test": 0, "val": 1, "train": 2}
    df["_o"] = df["split"].map(order)
    df = df.sort_values("_o").drop_duplicates("key", keep="first").drop(columns="_o")

    for c in LABELS:
        df[c] = df[c].astype(float).where(df[c] >= 0, MISSING).clip(upper=1.0)

    # Projeye özel veriler 470 bin satırın içinde kaybolmasın: eğitim bölmesinde
    # tekrarla (doğrulama/test tekrarlanmaz, metrikler etkilenmez).
    if supplement_repeat > 1:
        own = df[(df["split"] == "train") & df["source"].isin(["supplement", "generated"])]
        df = pd.concat([df] + [own] * (supplement_repeat - 1), ignore_index=True)

    limits = cfg.get("max_rows")
    if limits:
        parts = []
        for split, g in df.groupby("split"):
            n = limits.get(split)
            # Ek setin tamamı kalsın; kalanı örnekle.
            keep = g[g["source"] == "supplement"]
            rest = g.drop(keep.index)
            if n and len(rest) > n:
                rest = rest.sample(n, random_state=42)
            parts.append(pd.concat([keep, rest]))
        df = pd.concat(parts)
    return df.reset_index(drop=True)


def stats(df: pd.DataFrame) -> dict:
    out = {}
    for split, g in df.groupby("split"):
        s = {"rows": len(g), "by_source": g["source"].value_counts().to_dict(), "labels": {}}
        for c in LABELS:
            known = g[g[c] >= 0][c]
            s["labels"][c] = {"known": int(len(known)), "positive@0.5": int((known >= 0.5).sum())}
        out[split] = s
    return out


def validate(df: pd.DataFrame) -> None:
    """Eğitime geçmeden önce verinin kullanılabilir olduğunu doğrula."""
    for split in ("train", "val", "test"):
        assert (df["split"] == split).any(), f"'{split}' bölmesi boş"
    train = df[df["split"] == "train"]
    for c in LABELS:
        known = train[c] >= 0
        assert known.any(), f"'{c}' etiketi için hiç eğitim verisi yok"
        assert (train.loc[known, c] >= 0.5).sum() >= 5, f"'{c}' için pozitif örnek çok az"
    assert df["text"].str.len().gt(0).all(), "Boş metin var"
    assert not df[LABELS].isna().any().any(), "Etiketlerde NaN kaldı"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=PROFILES, default="small")
    ap.add_argument("--no-toxigen", action="store_true",
                    help="ToxiGen'i dışarıda bırak (lisansı yalnızca araştırma amaçlı)")
    ap.add_argument("--supplement-repeat", type=int, default=3,
                    help="Projeye özel verilerin eğitimde kaç kez tekrarlanacağı")
    ap.add_argument("--with-legacy", action="store_true",
                    help="Eski ChildGuard verisini eğitime kat (önerilmez)")
    args = ap.parse_args()

    df = build(args.profile, not args.no_toxigen, args.with_legacy, args.supplement_repeat)
    validate(df)
    out = paths.data_dir()
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        part = df[df["split"] == split].drop(columns=["split"])
        part.to_parquet(out / f"{split}.parquet", index=False)
    st = stats(df)
    st["profile"] = args.profile
    st["data_version"] = DATA_VERSION
    (out / "stats.json").write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")
    for split in ("train", "val", "test"):
        labels = {k: v["positive@0.5"] for k, v in st[split]["labels"].items()}
        print(f"  {split:5s} {st[split]['rows']:>8,} satır · pozitifler: {labels}")
    print(f"✅ Yazıldı: {out}")


if __name__ == "__main__":
    main()
