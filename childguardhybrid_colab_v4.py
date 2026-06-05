# -*- coding: utf-8 -*-
"""ChildGuard AI v3.0 — Veri Temizleme + İki Aşamalı Eğitim (Google Colab)
W&B ile profesyonel experiment tracking
"""

# ============================================================
# HÜCRE 1 — Kurulum
# ============================================================
# @title Kurulum
!pip install -q transformers datasets accelerate joblib seaborn \
             xgboost optuna shap ipywidgets wandb

# ============================================================
# HÜCRE 2 — W&B + Drive Bağlantısı
# ============================================================
# @title W&B ve Drive

import wandb
wandb.login()

from google.colab import drive
drive.mount('/content/drive')

REPO_DIR       = "/content/drive/MyDrive/ChildGuard/ChildGuard_repo"
DRIVE_SAVE_DIR = "/content/drive/MyDrive/ChildGuard/final_models_v3"
WANDB_PROJECT  = "childguard-v3"

import os
for f in ["ChildGuard.csv", "lexical_chilhate.csv", "contextual_childhate.csv"]:
    path = f"{REPO_DIR}/{f}"
    print(f"{'✅' if os.path.exists(path) else '❌'} {f}")

# ============================================================
# HÜCRE 3 — Import'lar ve Ayarlar
# ============================================================
# @title Import'lar

import os, re, shutil, time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_recall_fscore_support, classification_report,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import shap
from tqdm.auto import tqdm

TEXT           = "text"
TARGET         = "actual_class"
AGE_GROUP_COL  = "Age_Group"
MAX_LEN        = 128
BATCH_SIZE     = 16
EPOCHS         = 3
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AGE_LABEL_NAMES = ["Younger", "Pre-Teen", "Teen"]

os.makedirs("final_models", exist_ok=True)
print("✅ Ortam hazır.")
print(f"   PyTorch : {torch.__version__}")
print(f"   Device  : {DEVICE}")
print(f"   XGBoost : {xgb.__version__}")

# ============================================================
# HÜCRE 4 — Veri Temizleme ve Birleştirme
# ============================================================
# @title Veri Temizleme

print("Veriler yükleniyor...")
df_main = pd.read_csv(f"{REPO_DIR}/ChildGuard.csv",           encoding="cp1252")
df_lex  = pd.read_csv(f"{REPO_DIR}/lexical_chilhate.csv",     encoding="cp1252")
df_ctx  = pd.read_csv(f"{REPO_DIR}/contextual_childhate.csv", encoding="cp1252")

df_lex["Age_Group"] = "Unknown"
df_ctx["Age_Group"] = "Unknown"
df_ctx = df_ctx.drop(columns=["contextual_score"], errors="ignore")

df = pd.concat([df_main, df_lex, df_ctx], ignore_index=True)
print(f"Birleşik toplam   : {len(df):,}")

before = len(df)
df = df.dropna(subset=[TEXT, TARGET])
df = df[df[TEXT].str.strip() != ""]
df = df[df[TEXT].str.len() >= 10]
df = df[df[TEXT].str.len() <= 1000]
df = df.drop_duplicates(subset=[TEXT])
df = df[[TEXT, TARGET, AGE_GROUP_COL]]
df[TARGET] = df[TARGET].astype(float)

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s!?.,]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df[TEXT] = df[TEXT].apply(clean_text)
df = df[df[TEXT].str.strip() != ""]
df = df[df[TEXT].str.len() >= 10]

print(f"Temizleme sonrası : {len(df):,}  (düşen: {before - len(df):,})")
print(f"\nActual class:\n{df[TARGET].value_counts()}")
print(f"\nAge Group:\n{df[AGE_GROUP_COL].value_counts()}")

cleaned_path = "/content/drive/MyDrive/ChildGuard/ChildGuard_v3_Cleaned.csv"
df.to_csv(cleaned_path, index=False)
print(f"\n✅ Kaydedildi: {cleaned_path}")

# ============================================================
# HÜCRE 5 — Tokenizer ve Dataset Sınıfları
# ============================================================
# @title Tokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class BinaryDataset(Dataset):
    def __init__(self, texts, labels):
        self.enc    = tokenizer(texts, truncation=True, padding=True,
                                max_length=MAX_LEN, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {
            "input_ids":      self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
            "labels":         self.labels[idx],
        }

class MultiClassDataset(Dataset):
    def __init__(self, texts, labels):
        self.enc    = tokenizer(texts, truncation=True, padding=True,
                                max_length=MAX_LEN, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {
            "input_ids":      self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
            "labels":         self.labels[idx],
        }

def get_bert_probs(model, texts):
    model.eval()
    ds     = BinaryDataset(texts, [0] * len(texts))
    loader = DataLoader(ds, batch_size=BATCH_SIZE)
    probs  = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
            ).logits
            probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.array(probs)

def train_bert_model(train_loader, val_loader, num_labels, save_path, run_name):
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels)
    model.to(DEVICE)

    optimizer   = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    wandb.config.update({
        f"{run_name}_lr":         2e-5,
        f"{run_name}_batch_size": BATCH_SIZE,
        f"{run_name}_epochs":     EPOCHS,
        f"{run_name}_num_labels": num_labels,
    }, allow_val_change=True)

    best_val_acc, best_state = 0.0, None
    epoch_metrics = []

    for epoch in range(EPOCHS):
        t0 = time.time()
        model.train()
        running_loss, correct_tr, total_tr = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"[{run_name}] Ep {epoch+1}/{EPOCHS} Train",
                    unit="batch", leave=True)
        for step, batch in enumerate(pbar, 1):
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=batch["labels"].to(DEVICE),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()

            running_loss += out.loss.item()
            preds         = out.logits.argmax(dim=1).cpu()
            correct_tr   += (preds == batch["labels"]).sum().item()
            total_tr     += len(batch["labels"])
            pbar.set_postfix(
                loss=f"{running_loss/step:.4f}",
                acc=f"{correct_tr/total_tr:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.1e}",
            )

        model.eval()
        correct_v, total_v, val_loss = 0, 0, 0.0
        all_preds_v, all_labels_v = [], []

        vbar = tqdm(val_loader, desc=f"[{run_name}] Ep {epoch+1}/{EPOCHS} Val  ",
                    unit="batch", leave=True)
        with torch.no_grad():
            for batch in vbar:
                out_v = model(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                    labels=batch["labels"].to(DEVICE),
                )
                val_loss  += out_v.loss.item()
                preds_v    = out_v.logits.argmax(dim=1).cpu()
                correct_v += (preds_v == batch["labels"]).sum().item()
                total_v   += len(batch["labels"])
                all_preds_v.extend(preds_v.numpy())
                all_labels_v.extend(batch["labels"].numpy())
                vbar.set_postfix(
                    val_loss=f"{val_loss/(vbar.n+1):.4f}",
                    val_acc=f"{correct_v/total_v:.4f}",
                )

        val_acc   = correct_v / total_v
        train_acc = correct_tr / total_tr
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss   = val_loss / len(val_loader)
        epoch_time     = time.time() - t0

        if num_labels == 2:
            p, r, f1, _ = precision_recall_fscore_support(
                all_labels_v, all_preds_v, average="binary")
        else:
            p, r, f1, _ = precision_recall_fscore_support(
                all_labels_v, all_preds_v, average="macro")

        flag = " ✅ en iyi" if val_acc > best_val_acc else ""
        print(f"  → Epoch {epoch+1} | train_acc={train_acc:.4f} | "
              f"val_acc={val_acc:.4f} | val_f1={f1:.4f} | "
              f"time={epoch_time:.0f}s{flag}\n")

        wandb.log({
            f"{run_name}/train_loss":    avg_train_loss,
            f"{run_name}/train_acc":     train_acc,
            f"{run_name}/val_loss":      avg_val_loss,
            f"{run_name}/val_acc":       val_acc,
            f"{run_name}/val_f1":        f1,
            f"{run_name}/val_precision": p,
            f"{run_name}/val_recall":    r,
            f"{run_name}/epoch_time_s":  epoch_time,
            "epoch": epoch + 1,
        })

        epoch_metrics.append({
            "epoch":      epoch + 1,
            "train_loss": round(avg_train_loss, 4),
            "train_acc":  round(train_acc, 4),
            "val_loss":   round(avg_val_loss, 4),
            "val_acc":    round(val_acc, 4),
            "val_f1":     round(f1, 4),
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"  💾 Kaydedildi → {save_path}  (best val_acc={best_val_acc:.4f})")
    return model, epoch_metrics

print("✅ Tokenizer ve yardımcı sınıflar hazır.")

# ============================================================
# HÜCRE 6 — W&B Run Başlat
# ============================================================
# @title W&B Run Başlat

run = wandb.init(
    project=WANDB_PROJECT,
    name=f"childguard-v3-{datetime.now().strftime('%Y%m%d-%H%M')}",
    config={
        "dataset":    "ChildGuard_v3_Cleaned",
        "bert_model": "bert-base-uncased",
        "batch_size": BATCH_SIZE,
        "epochs":     EPOCHS,
        "max_len":    MAX_LEN,
        "device":     str(DEVICE),
    }
)
print(f"✅ W&B run başladı: {run.url}")

# ============================================================
# HÜCRE 7 — AŞAMA 1: Genel BERT
# ⚠️  ~45–60 dk
# ============================================================
# @title Aşama 1 — Genel BERT

print("=" * 60)
print(" AŞAMA 1: Genel Zararlı İçerik Tespiti")
print("=" * 60)

X_all = df[TEXT].tolist()
y_all = df[TARGET].astype(int).tolist()

X_tr, X_te, y_tr, y_te = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
X_tr, X_va, y_tr, y_va = train_test_split(
    X_tr, y_tr, test_size=0.1, random_state=42, stratify=y_tr)

wandb.log({
    "data/total_samples":  len(df),
    "data/train_samples":  len(X_tr),
    "data/val_samples":    len(X_va),
    "data/test_samples":   len(X_te),
    "data/harmful_ratio":  sum(y_all) / len(y_all),
})

df_tr  = pd.DataFrame({"text": X_tr, "label": y_tr})
df_maj = df_tr[df_tr.label == 0]
df_min = df_tr[df_tr.label == 1]
df_min_up = resample(df_min, replace=True, n_samples=len(df_maj), random_state=42)
df_up = pd.concat([df_maj, df_min_up]).sample(frac=1, random_state=42)

train_loader = DataLoader(
    BinaryDataset(df_up["text"].tolist(), df_up["label"].tolist()),
    batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(BinaryDataset(X_va, y_va), batch_size=BATCH_SIZE)

bert_general, bert_general_metrics = train_bert_model(
    train_loader, val_loader,
    num_labels=2,
    save_path="final_models/bert_general",
    run_name="bert_general",
)

all_probs  = get_bert_probs(bert_general, X_te)
all_labels = np.array(y_te)

best_f1, best_thr, best_preds = -1, 0.5, None
for thr in np.linspace(0.3, 0.7, 20):
    preds = (all_probs >= thr).astype(int)
    f1    = precision_recall_fscore_support(all_labels, preds, average="binary")[2]
    if f1 > best_f1:
        best_f1, best_thr, best_preds = f1, thr, preds

joblib.dump({"threshold": best_thr}, "final_models/bert_general_config.pkl")

p_te, r_te, f1_te, _ = precision_recall_fscore_support(all_labels, best_preds, average="binary")
acc_te = accuracy_score(all_labels, best_preds)

wandb.log({
    "bert_general/test_accuracy":  acc_te,
    "bert_general/test_f1":        f1_te,
    "bert_general/test_precision": p_te,
    "bert_general/test_recall":    r_te,
    "bert_general/best_threshold": best_thr,
})

print(f"\n  Threshold: {best_thr:.2f} | Test F1: {f1_te:.4f} | Acc: {acc_te:.4f}")
print(classification_report(all_labels, best_preds, target_names=["Güvenli", "Zararlı"]))

cm = confusion_matrix(all_labels, best_preds)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Güvenli", "Zararlı"],
            yticklabels=["Güvenli", "Zararlı"], ax=ax)
ax.set_title(f"BERT Genel (thr={best_thr:.2f})")
wandb.log({"bert_general/confusion_matrix": wandb.Image(fig)})
plt.tight_layout(); plt.show()

# ============================================================
# HÜCRE 8 — AŞAMA 1: XGBoost
# ⚠️  ~20–30 dk
# ============================================================
# @title Aşama 1 — XGBoost

print("=" * 60)
print(" AŞAMA 1: XGBoost")
print("=" * 60)

df_xgb = df.copy()
df_xgb["text_len"] = df_xgb[TEXT].str.len()
df_xgb["word_cnt"] = df_xgb[TEXT].str.split().str.len()

age_dummies       = pd.get_dummies(df_xgb[AGE_GROUP_COL], prefix="AgeGroup")
AGE_GROUP_COLUMNS = list(age_dummies.columns)
joblib.dump(AGE_GROUP_COLUMNS, "final_models/age_group_columns.pkl")

X_meta = df_xgb[["text_len", "word_cnt"]]
y_xgb  = df_xgb[TARGET].astype(int)
texts  = df_xgb[TEXT]

(X_m_tr, X_m_te,
 y_tr_x, y_te_x,
 t_tr,   t_te,
 age_tr, age_te) = train_test_split(
    X_meta, y_xgb, texts, age_dummies,
    test_size=0.2, random_state=42, stratify=y_xgb,
)

vectorizer = TfidfVectorizer(max_features=5000, min_df=5)
X_tr_tfidf = vectorizer.fit_transform(t_tr)
X_te_tfidf = vectorizer.transform(t_te)

age_te     = age_te.reindex(columns=age_tr.columns, fill_value=0)
X_tr_final = hstack([X_tr_tfidf, csr_matrix(X_m_tr.values), csr_matrix(age_tr.values)])
X_te_final = hstack([X_te_tfidf, csr_matrix(X_m_te.values), csr_matrix(age_te.values)])

neg = (y_tr_x == 0).sum(); pos = (y_tr_x == 1).sum()
scale_pos_weight = neg / pos
print(f"scale_pos_weight = {scale_pos_weight:.3f}")

def xgb_objective(trial):
    params = {
        "n_estimators":     trial.suggest_int  ("n_estimators",     100, 800, step=50),
        "max_depth":        trial.suggest_int  ("max_depth",          3,  10),
        "learning_rate":    trial.suggest_float("learning_rate",    0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample",         0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree",  0.5, 1.0),
        "min_child_weight": trial.suggest_int  ("min_child_weight",    1,  10),
        "scale_pos_weight": scale_pos_weight,
        "eval_metric":      "logloss",
        "random_state":     42,
        "tree_method":      "hist",
        "device":           "cpu",
    }
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_tr_final, y_tr_x, verbose=False)
    preds = clf.predict(X_te_final)
    f1 = precision_recall_fscore_support(y_te_x, preds, average="binary")[2]
    wandb.log({"xgb/trial_f1": f1, "xgb/trial": trial.number})
    return f1

print("Optuna araması (50 deneme)...")
study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(xgb_objective, n_trials=50, show_progress_bar=True)
print(f"✅ En iyi F1: {study.best_value:.4f}")

best_xgb_params = {
    **study.best_params,
    "scale_pos_weight": scale_pos_weight,
    "eval_metric": "logloss",
    "random_state": 42,
    "tree_method": "hist",
    "device": "cpu",
}
best_xgb = xgb.XGBClassifier(**best_xgb_params)
best_xgb.fit(X_tr_final, y_tr_x, verbose=False)

joblib.dump(best_xgb,   "final_models/xgboost_model.pkl")
joblib.dump(vectorizer, "final_models/tfidf_vectorizer.pkl")

y_pred_xgb = best_xgb.predict(X_te_final)
p_x, r_x, f1_x, _ = precision_recall_fscore_support(y_te_x, y_pred_xgb, average="binary")
acc_x = accuracy_score(y_te_x, y_pred_xgb)

wandb.log({
    "xgb/test_accuracy":  acc_x,
    "xgb/test_f1":        f1_x,
    "xgb/test_precision": p_x,
    "xgb/test_recall":    r_x,
})

print(classification_report(y_te_x, y_pred_xgb, target_names=["Güvenli", "Zararlı"]))

cm_xgb = confusion_matrix(y_te_x, y_pred_xgb)
fig2, ax2 = plt.subplots(figsize=(5, 4))
sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Güvenli", "Zararlı"],
            yticklabels=["Güvenli", "Zararlı"], ax=ax2)
ax2.set_title("XGBoost — Genel")
wandb.log({"xgb/confusion_matrix": wandb.Image(fig2)})
plt.tight_layout(); plt.show()

# ============================================================
# HÜCRE 9 — SHAP
# ============================================================
# @title SHAP

feature_names = (
    vectorizer.get_feature_names_out().tolist()
    + ["text_len", "word_cnt"]
    + AGE_GROUP_COLUMNS
)
explainer   = shap.TreeExplainer(best_xgb)
X_sample    = X_te_final.toarray()[:200]
shap_values = explainer.shap_values(X_sample)

fig3, ax3 = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                  max_display=15, show=False)
wandb.log({"xgb/shap_summary": wandb.Image(fig3)})
plt.tight_layout(); plt.show()
print("✅ SHAP hazır.")

# ============================================================
# HÜCRE 10 — AŞAMA 2: Yaş Grubu BERT
# ⚠️  ~20–30 dk
# ============================================================
# @title Aşama 2 — Yaş Grubu

print("=" * 60)
print(" AŞAMA 2: Yaş Grubu Sınıflandırması")
print("=" * 60)

df_age = df[
    (~df[AGE_GROUP_COL].str.contains("Unknown", case=False, na=False)) &
    (df[TARGET] == 1.0)
].copy()

def normalize_age_group(val):
    v = str(val).lower()
    if "under 11" in v or "younger" in v: return 0
    if "11" in v or "pre" in v:           return 1
    if "13" in v or "teen" in v:          return 2
    return -1

df_age["age_label"] = df_age[AGE_GROUP_COL].apply(normalize_age_group)
df_age = df_age[df_age["age_label"] >= 0]

print(f"Yaş grubu: {len(df_age):,} zararlı örnek")
print(df_age["age_label"].value_counts().rename({0:"Younger",1:"Pre-Teen",2:"Teen"}))

X_age = df_age[TEXT].tolist()
y_age = df_age["age_label"].tolist()

X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(
    X_age, y_age, test_size=0.2, random_state=42, stratify=y_age)
X_tr_a, X_va_a, y_tr_a, y_va_a = train_test_split(
    X_tr_a, y_tr_a, test_size=0.1, random_state=42, stratify=y_tr_a)

df_tr_a  = pd.DataFrame({"text": X_tr_a, "label": y_tr_a})
max_size = df_tr_a["label"].value_counts().max()
balanced = pd.concat([
    resample(df_tr_a[df_tr_a.label == cls],
             replace=True, n_samples=max_size, random_state=42)
    for cls in [0, 1, 2]
]).sample(frac=1, random_state=42)

train_loader_a = DataLoader(
    MultiClassDataset(balanced["text"].tolist(), balanced["label"].tolist()),
    batch_size=BATCH_SIZE, shuffle=True)
val_loader_a = DataLoader(MultiClassDataset(X_va_a, y_va_a), batch_size=BATCH_SIZE)

bert_age, bert_age_metrics = train_bert_model(
    train_loader_a, val_loader_a,
    num_labels=3,
    save_path="final_models/bert_age_group",
    run_name="bert_age",
)

bert_age.eval()
all_preds_a, all_labels_a = [], []
with torch.no_grad():
    for batch in DataLoader(MultiClassDataset(X_te_a, y_te_a), batch_size=BATCH_SIZE):
        preds = bert_age(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE),
        ).logits.argmax(dim=1).cpu().numpy()
        all_preds_a.extend(preds)
        all_labels_a.extend(batch["labels"].numpy())

p_a, r_a, f1_a, _ = precision_recall_fscore_support(
    all_labels_a, all_preds_a, average="macro")
acc_a = accuracy_score(all_labels_a, all_preds_a)

wandb.log({
    "bert_age/test_accuracy":  acc_a,
    "bert_age/test_f1_macro":  f1_a,
    "bert_age/test_precision": p_a,
    "bert_age/test_recall":    r_a,
})

print(classification_report(all_labels_a, all_preds_a, target_names=AGE_LABEL_NAMES))

cm_age = confusion_matrix(all_labels_a, all_preds_a)
fig4, ax4 = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_age, annot=True, fmt="d", cmap="Purples",
            xticklabels=AGE_LABEL_NAMES, yticklabels=AGE_LABEL_NAMES, ax=ax4)
ax4.set_title("Yaş Grubu Sınıflandırıcı")
wandb.log({"bert_age/confusion_matrix": wandb.Image(fig4)})
plt.tight_layout(); plt.show()

joblib.dump(AGE_LABEL_NAMES, "final_models/age_label_names.pkl")

# ============================================================
# HÜCRE 11 — Özet + ZIP + Drive + W&B Finish
# ============================================================
# @title Özet ve İndir

wandb.summary.update({
    "bert_general_test_acc":  acc_te,
    "bert_general_test_f1":   f1_te,
    "xgb_test_acc":           acc_x,
    "xgb_test_f1":            f1_x,
    "bert_age_test_acc":      acc_a,
    "bert_age_test_f1_macro": f1_a,
    "best_threshold":         best_thr,
    "dataset_size":           len(df),
})

print("\n=== ÖZET ===")
print(f"  BERT Genel → acc={acc_te:.4f}  F1={f1_te:.4f}  thr={best_thr:.2f}")
print(f"  XGBoost    → acc={acc_x:.4f}  F1={f1_x:.4f}")
print(f"  BERT Yaş   → acc={acc_a:.4f}  F1={f1_a:.4f} (macro)")

shutil.make_archive("ChildGuard_v3_Bundle", "zip", "final_models")
os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
shutil.copy("ChildGuard_v3_Bundle.zip", f"{DRIVE_SAVE_DIR}/ChildGuard_v3_Bundle.zip")
print(f"\n✅ Drive'a kaydedildi: {DRIVE_SAVE_DIR}/ChildGuard_v3_Bundle.zip")

wandb.finish()
print("✅ W&B run tamamlandı.")

from google.colab import files
files.download("ChildGuard_v3_Bundle.zip")
