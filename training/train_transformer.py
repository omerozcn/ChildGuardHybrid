"""Transformer modelini çok etiketli olarak ince ayarlar.

Eksik etiketler (bir kaynağın bilgi taşımadığı hücreler) kayıpta maskelenir.
Nadir etiketler için pos_weight kullanılır (yakalama oranını artırır).

Uzun eğitimler için güvenceler:
  * Bellek ön testi: eğitimden önce en kötü durum (tam batch × max_len) ile
    bir adım atılır; bellek yetmezse saniyeler içinde net hata verir.
  * Devam ettirme: checkpoint'ler `paths.checkpoint_dir()` altına yazılır
    (Colab'da Drive). Aynı komut tekrar çalıştırılırsa kaldığı yerden sürer.
  * Uyumsuzluk koruması: farklı model/veri ile eski checkpoint'ten devam
    edilmesi engellenir (`--fresh` ile sıfırdan başlanır).

    python -m training.train_transformer                 # GPU'ya göre otomatik ayar
    python -m training.train_transformer --model roberta-base --bs 32 --epochs 2
"""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback, Trainer,
                          TrainerCallback, TrainingArguments, set_seed)

from childguard.labels import LABELS
from childguard.text import for_transformer
from training import paths


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.enc = tokenizer([for_transformer(t) for t in df["text"]],
                             truncation=True, max_length=max_len)
        self.labels = df[LABELS].astype("float32").values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def masked_bce(logits: torch.Tensor, labels: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    mask = (labels >= 0).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.float(), labels.clamp(min=0), reduction="none",
        pos_weight=pos_weight.to(logits.device))
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


class MaskedBCETrainer(Trainer):
    def __init__(self, *a, pos_weight: torch.Tensor, **kw):
        super().__init__(*a, **kw)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = masked_bce(outputs.logits, labels, self.pos_weight)
        return (loss, outputs) if return_outputs else loss


class StepTimer(TrainerCallback):
    """Yalnızca eğitim adımlarının süresini ölçer.

    HF'nin `train_runtime` değeri ara değerlendirme ve checkpoint kaydetmeyi
    de içerir; büyük modellerde kayıt süresi baskın olduğu için süre
    tahmini katlarca şişer.
    """

    def __init__(self):
        self.seconds, self.steps, self._t = 0.0, 0, None

    def on_step_begin(self, args, state, control, **kw):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t = time.perf_counter()

    def on_step_end(self, args, state, control, **kw):
        if self._t is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self.steps > 0:                       # ilk adım (ısınma) sayılmaz
            self.seconds += time.perf_counter() - self._t
        self.steps += 1
        self._t = None


def compute_metrics(pred):
    logits, labels = pred
    probs = 1 / (1 + np.exp(-logits.astype(np.float64)))
    out, aps = {}, []
    for j, label in enumerate(LABELS):
        known = labels[:, j] >= 0
        y = (labels[known, j] >= 0.5).astype(int)
        if known.sum() and 0 < y.sum() < len(y):
            ap = average_precision_score(y, probs[known, j])
            out[f"pr_auc_{label}"] = ap
            aps.append(ap)
    out["pr_auc_macro"] = float(np.mean(aps)) if aps else 0.0
    return out


def load_with_retry(loader, name: str, **kw):
    """Önbellekteki ağırlık dosyası bozuksa (ör. ağ diskinde yarım/bozuk okunmuşsa)
    modeli bir kez zorla yeniden indirir."""
    try:
        return loader(name, **kw)
    except Exception as exc:
        msg = str(exc).lower()
        if not any(s in msg for s in ("header too large", "deserializ", "safetensor",
                                      "invalid load key", "unexpected eof", "corrupt")):
            raise
        print(f"⚠️  Önbellekteki model dosyası okunamadı ({exc.__class__.__name__}: {exc}). "
              f"Yeniden indiriliyor...", flush=True)
        return loader(name, force_download=True, **kw)


def make_training_args(**kw) -> TrainingArguments:
    """transformers 4.x ve 5.x ile uyumlu TrainingArguments.

    Benzer uzunluktaki örnekleri aynı batch'e koymak dolguyu azaltır
    (~1.5–2x hız). 4.x'te `group_by_length=True`, 5.x'te
    `train_sampling_strategy="group_by_length"` ile açılır.
    """
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "train_sampling_strategy" in params:
        kw["train_sampling_strategy"] = "group_by_length"
    elif "group_by_length" in params:
        kw["group_by_length"] = True
    if "eval_strategy" not in params:                       # çok eski 4.x
        kw["evaluation_strategy"] = kw.pop("eval_strategy")
    return TrainingArguments(**kw)


def gpu_preset() -> dict:
    """GPU belleğine göre makul varsayılanlar.

    ≥30 GB (A100): roberta-large · 20–30 GB (L4/A10): roberta-large, küçük batch
    ≤16 GB (T4, yerel kartlar): roberta-base
    """
    if not torch.cuda.is_available():
        return {"model": "distilroberta-base", "bs": 16, "grad_accum": 1, "lr": 3e-5}
    mem = torch.cuda.get_device_properties(0).total_memory / 2**30
    if mem >= 30:
        p = {"model": "roberta-large", "bs": 32, "grad_accum": 1, "lr": 1e-5}
    elif mem >= 20:
        p = {"model": "roberta-large", "bs": 16, "grad_accum": 2, "lr": 1e-5}
    elif mem >= 12:
        p = {"model": "roberta-base", "bs": 32, "grad_accum": 1, "lr": 2e-5}
    else:
        p = {"model": "roberta-base", "bs": 16, "grad_accum": 2, "lr": 2e-5}
    print(f"GPU: {torch.cuda.get_device_name(0)} ({mem:.0f} GB) → {p}", flush=True)
    return p


def memory_probe(model, bs: int, max_len: int, bf16: bool, pos_weight: torch.Tensor) -> float:
    """En kötü durum batch'iyle bir ileri+geri adım; tepe belleği (GB) döndürür."""
    if not torch.cuda.is_available():
        return 0.0
    model.cuda().train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(5, 1000, (bs, max_len), device="cuda")
    labels = torch.rand(bs, len(LABELS), device="cuda")
    dtype = torch.bfloat16 if bf16 else torch.float16
    try:
        with torch.autocast("cuda", dtype=dtype):
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
        masked_bce(logits, labels, pos_weight).backward()
    except torch.OutOfMemoryError:
        raise SystemExit(f"❌ Bellek yetmedi (batch={bs}, max_len={max_len}). "
                         f"--bs değerini yarıya indirip --grad-accum değerini iki katına çıkarın.")
    model.zero_grad(set_to_none=True)
    # AdamW, parametre başına 2 adet fp32 durum tutar (8 bayt); ilk adımda eklenir.
    optim = sum(p.numel() for p in model.parameters()) * 8 / 2**30
    peak = torch.cuda.max_memory_allocated() / 2**30 + optim
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    del ids, labels, logits
    torch.cuda.empty_cache()
    print(f"Bellek ön testi: tahmini tepe {peak:.1f} GB / {total:.0f} GB "
          f"(batch={bs}, max_len={max_len}, optimizer dahil) ✓", flush=True)
    if peak > total:
        raise SystemExit(f"❌ Tahmini bellek ({peak:.1f} GB) GPU'yu aşıyor. "
                         f"--bs değerini yarıya indirip --grad-accum değerini iki katına çıkarın.")
    if peak > 0.9 * total:
        print("⚠️  Bellek sınıra çok yakın; optimizer durumları eklenince taşabilir.", flush=True)
    return peak


def find_resume(ckpt_dir: Path, signature: dict, fresh: bool) -> bool:
    sig_file = ckpt_dir / "run_signature.json"
    has_ckpt = any(ckpt_dir.glob("checkpoint-*"))
    if fresh and ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
        has_ckpt = False
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if has_ckpt:
        old = json.loads(sig_file.read_text()) if sig_file.exists() else {}
        if old != signature:
            raise SystemExit(
                "❌ Checkpoint klasöründe farklı bir eğitimin kayıtları var:\n"
                f"   eski: {old}\n   yeni: {signature}\n"
                "   Sıfırdan başlamak için --fresh ekleyin.")
        print(f"↻ Kaldığı yerden devam ediliyor: {ckpt_dir}", flush=True)
        return True
    sig_file.write_text(json.dumps(signature))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["auto", "none"], default="auto",
                    help="auto: GPU'ya göre model/batch/lr seç; elle verilen değerler önceliklidir")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-train", type=int, default=None,
                    help="Eğitim satırı üst sınırı (hızlı deneme için)")
    ap.add_argument("--max-steps", type=int, default=-1, help="Duman testi için adım sınırı")
    ap.add_argument("--fresh", action="store_true", help="Eski checkpoint'leri silip sıfırdan başla")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    preset = gpu_preset() if args.preset == "auto" else {
        "model": "roberta-base", "bs": 16, "grad_accum": 2, "lr": 2e-5}
    for k in ("model", "bs", "grad_accum", "lr"):
        if getattr(args, k) is None:
            setattr(args, k, preset[k])
    set_seed(args.seed)

    data = paths.data_dir()
    train = pd.read_parquet(data / "train.parquet")
    val = pd.read_parquet(data / "val.parquet")
    if args.max_train and len(train) > args.max_train:
        # Nadir etiketlerin pozitiflerini koru, kalanı rastgele örnekle.
        rare = train[(train[["threat", "sexual", "identity_hate", "targets_minor"]] >= 0.5).any(axis=1)]
        if len(rare) > args.max_train // 2:
            rare = rare.sample(args.max_train // 2, random_state=args.seed)
        rest = train.drop(rare.index).sample(args.max_train - len(rare), random_state=args.seed)
        train = pd.concat([rare, rest]).sample(frac=1, random_state=args.seed)
    print(f"train={len(train):,} val={len(val):,} model={args.model}", flush=True)

    pos_w = []
    for label in LABELS:
        known = train[label] >= 0
        pos = (train.loc[known, label] >= 0.5).sum()
        neg = known.sum() - pos
        pos_w.append(float(np.clip(neg / max(pos, 1), 1.0, 8.0)))
    pos_weight = torch.tensor(pos_w)
    print("pos_weight:", dict(zip(LABELS, [round(w, 2) for w in pos_w])), flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_with_retry(
        AutoModelForSequenceClassification.from_pretrained,
        args.model, num_labels=len(LABELS), problem_type="multi_label_classification",
        id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)},
        # Hazır sınıflandırma başlığı olan modellerde (ör. toksisite modelleri)
        # başlık 7 etikete göre yeniden başlatılır.
        ignore_mismatched_sizes=True)

    use_cuda = torch.cuda.is_available()
    bf16 = use_cuda and torch.cuda.is_bf16_supported()     # A100/L4: bf16, T4: fp16
    memory_probe(model, args.bs, args.max_len, bf16, pos_weight)

    steps_per_epoch = max(1, len(train) // (args.bs * args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.epochs)
    eval_steps = max(100, steps_per_epoch // 3)    # epoch başına ~3 değerlendirme
    if args.max_steps > 0:
        eval_steps = max(5, args.max_steps // 2)
    ckpt_dir = paths.checkpoint_dir()
    signature = {"model": args.model, "train_rows": len(train), "val_rows": len(val),
                 "bs": args.bs, "grad_accum": args.grad_accum, "max_len": args.max_len,
                 "epochs": args.epochs, "max_steps": args.max_steps}
    resume = find_resume(ckpt_dir, signature, args.fresh)
    print(f"Toplam adım ≈ {total_steps:,} · değerlendirme her {eval_steps} adımda", flush=True)

    targs = make_training_args(
        output_dir=str(ckpt_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=int(0.06 * total_steps),
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=1,            # HF en iyi checkpoint'i ayrıca korur
        load_best_model_at_end=True,
        metric_for_best_model="pr_auc_macro",
        greater_is_better=True,
        bf16=bf16,
        fp16=use_cuda and not bf16,
        max_steps=args.max_steps,
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=2 if use_cuda else 0,
        seed=args.seed,
    )
    t0 = time.time()
    print("Belirteçlere ayrılıyor...", flush=True)
    train_ds = TextDataset(train, tok, args.max_len)
    val_ds = TextDataset(val, tok, args.max_len)
    timer = StepTimer()
    trainer = MaskedBCETrainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), timer],
        pos_weight=pos_weight,
    )
    trainer.train(resume_from_checkpoint=True if resume else None)
    eval_secs = sum(h.get("eval_runtime", 0) for h in trainer.state.log_history)
    pure_sps = ((timer.steps - 1) * args.bs * args.grad_accum / timer.seconds
                if timer.steps > 1 and timer.seconds > 0 else 0.0)
    print(f"Saf eğitim hızı: ~{pure_sps:.0f} örnek/s", flush=True)
    metrics = trainer.evaluate()

    out = paths.model_dir() / "transformer"
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(out)
    tok.save_pretrained(out)
    (out / "training_meta.json").write_text(json.dumps({
        "base_model": args.model, "max_len": args.max_len, "train_rows": len(train),
        "pos_weight": dict(zip(LABELS, pos_w)), "val_metrics": metrics,
        "minutes": round((time.time() - t0) / 60, 1),
        "train_samples_per_second": round(pure_sps, 1),
        "eval_seconds_per_1k": round(1000 * eval_secs / max(
            len(val) * sum("eval_runtime" in h for h in trainer.state.log_history), 1), 2),
        "bs": args.bs, "grad_accum": args.grad_accum, "epochs": args.epochs,
    }, indent=2), encoding="utf-8")
    print(f"✅ Kaydedildi: {out}  ({(time.time() - t0) / 60:.1f} dk)")
    print(json.dumps({k: round(v, 4) for k, v in metrics.items() if "pr_auc" in k}, indent=2))


if __name__ == "__main__":
    main()
