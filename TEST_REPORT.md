# ChildGuard AI v2.0 — Test Raporu

Tarih: 2026-04-14  
Backend: `http://localhost:8000`  
Frontend: `http://localhost:3000`

---

## 1. GET /api/health

**Süre:** 0.001s | **Status:** 200

```json
{
  "status": "ok",
  "models": [
    { "name": "BERT — Genel",     "loaded": true, "path": "../final_models/bert_general" },
    { "name": "BERT — Yaş Grubu", "loaded": true, "path": "../final_models/bert_age_group" },
    { "name": "XGBoost",          "loaded": true, "path": "../final_models/xgboost_model.pkl" }
  ]
}
```

**Sonuç:** PASS — 3 model yüklü.

---

## 2. POST /api/analyze

### 2a. Zararlı Metin

**Input:**
```json
{
  "text": "You are stupid and ugly, everyone hates you, just kill yourself loser.",
  "age_group": "Teen"
}
```

**Süre:** 0.116s | **Status:** 200

```json
{
  "label": "🚨 RİSKLİ / SİBER ZORBALIK",
  "final_score": 0.9954,
  "bert_score": 0.9961,
  "xgb_score": 0.9944,
  "age_group": "Teen",
  "age_prediction": {
    "predicted": "Teen",
    "scores": { "Younger": 0.0011, "Pre-Teen": 0.0011, "Teen": 0.9978 }
  }
}
```

**Sonuç:** PASS — Yüksek güven (%99.5), yaş tahmini `Teen` (%99.8).

---

### 2b. Güvenli Metin

**Input:**
```json
{
  "text": "I love playing basketball with my friends after school, it is so much fun!",
  "age_group": "Younger"
}
```

**Süre:** 0.059s | **Status:** 200

```json
{
  "label": "✅ GÜVENLİ / TEMİZ",
  "final_score": 0.1358,
  "bert_score": 0.0001,
  "xgb_score": 0.3394,
  "age_group": "Younger",
  "age_prediction": null
}
```

**Sonuç:** PASS — Düşük risk (%13.6), yaş tahmini tetiklenmedi (eşik altı).

---

## 3. POST /api/explain/shap

**Input:**
```json
{
  "text": "You are stupid and ugly, everyone hates you, just kill yourself loser.",
  "age_group": "Teen"
}
```

**Süre:** 0.021s | **Status:** 200

```json
{
  "features": [
    { "token": "trannies",  "weight":  2.567 },
    { "token": "bihday",    "weight": -2.052 },
    { "token": "tranny",    "weight":  1.834 },
    { "token": "8220",      "weight": -0.896 },
    { "token": "fathersday","weight": -0.852 },
    { "token": "fuck",      "weight": -0.812 },
    { "token": "sandbox",   "weight": -0.800 },
    { "token": "reddit",    "weight":  0.793 },
    { "token": "licker",    "weight":  0.692 },
    { "token": "committee", "weight": -0.633 }
  ]
}
```

**Sonuç:** PASS — SHAP çalışıyor. Not: XGBoost global vocab'a dayandığından giriş metninin tokenleri yerine eğitim veri kümesinden ağırlıklı tokenler görünüyor.

---

## 4. POST /api/explain/lime

**Input:**
```json
{
  "text": "You are stupid and ugly, everyone hates you, just kill yourself loser.",
  "age_group": "Teen",
  "num_samples": 300
}
```

**Süre:** 2.741s | **Status:** 200

```json
{
  "features": [
    { "token": "stupid",   "weight":  0.0951 },
    { "token": "loser",    "weight":  0.0865 },
    { "token": "ugly",     "weight":  0.0837 },
    { "token": "kill",     "weight":  0.0704 },
    { "token": "everyone", "weight": -0.0300 },
    { "token": "hates",    "weight":  0.0226 },
    { "token": "You",      "weight": -0.0164 },
    { "token": "yourself", "weight":  0.0158 },
    { "token": "and",      "weight": -0.0147 },
    { "token": "just",     "weight":  0.0134 }
  ]
}
```

**Sonuç:** PASS — LIME çalışıyor. `stupid`, `loser`, `ugly`, `kill` en yüksek pozitif ağırlıklar — giriş metniyle tutarlı.

---

## 5. Frontend

**URL:** `http://localhost:3000`  
**Süre:** 0.028s | **Status:** 200

**Sonuç:** PASS — Next.js 16 + Tailwind + shadcn/ui çalışıyor.

---

## Özet

| Test | Status | Süre |
|---|---|---|
| GET /api/health | PASS | 0.001s |
| POST /api/analyze (zararlı) | PASS | 0.116s |
| POST /api/analyze (güvenli) | PASS | 0.059s |
| POST /api/explain/shap | PASS | 0.021s |
| POST /api/explain/lime | PASS | 2.741s |
| Frontend http://localhost:3000 | PASS | 0.028s |

**6/6 PASS**

### Dikkat

- SHAP global TF-IDF vocab tokenlarını döndürüyor; giriş metnindeki kelimeler doğrudan çıkmıyor. Bu beklenen davranış.
- LIME süresi `num_samples`'a doğrusal bağlı; default 500'de ~4-5s beklenir.
