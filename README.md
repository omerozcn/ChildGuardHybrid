# ChildGuard AI v3.0

Çevrim içi metinlerde çocuklara yönelik zararlı içerik tespiti yapan iki aşamalı hibrit yapay zekâ sistemi.

> Afyon Kocatepe Üniversitesi — Veri Madenciliği dersi final projesi.

---

## Genel Bakış

ChildGuard AI, sosyal medya ve mesajlaşma platformlarındaki metinleri iki aşamada analiz eder:

1. **Aşama 1 — Zararlı mı?** BERT ve XGBoost modellerinin hibrit skoruyla risk tespiti.
2. **Aşama 2 — Kime yönelik?** Zararlı içerikse hedef yaş grubunu sınıflandırma (Younger / Pre-Teen / Teen).

Sistem ayrıca tahminin neden o şekilde verildiğini açıklayan **SHAP** (XGBoost) ve **LIME** (BERT) tabanlı XAI çıktıları üretir.

---

## Mimari

```
┌─────────────┐     POST /api/analyze     ┌──────────────────┐
│  Frontend   │ ────────────────────────► │  FastAPI Backend │
│ (Next.js 16)│                            │                  │
│             │ ◄──── analiz + skor ──────│  ┌────────────┐  │
│             │                            │  │ BERT genel │  │
│             │     POST /api/explain     │  └────────────┘  │
│             │ ────────────────────────► │  ┌────────────┐  │
│             │                            │  │ XGBoost    │  │
│             │ ◄──── SHAP / LIME ────────│  └────────────┘  │
└─────────────┘                            │  ┌────────────┐  │
                                           │  │ BERT yaş   │  │
                                           │  └────────────┘  │
                                           └──────────────────┘
```

**Hibrit skor:**

```
final_score = 0.6 * bert_score + 0.4 * xgb_score
```

Eşik `bert_general_config.pkl` dosyasından okunur (saklanan değer: 0.32). Skor eşiğin üzerindeyse içerik **RİSKLİ** olarak işaretlenir, sonrasında yaş grubu modeli çalıştırılır.

---

## Teknolojiler

| Katman | Stack |
|---|---|
| Backend | FastAPI, Uvicorn, Pydantic |
| ML | PyTorch, HuggingFace Transformers (`bert-base-uncased`), XGBoost, scikit-learn TF-IDF |
| XAI | SHAP (TreeExplainer), LIME (LimeTextExplainer) |
| Frontend | Next.js 16, TypeScript, Tailwind CSS, shadcn/ui, Recharts |
| Deploy | Docker Compose |
| Experiment Tracking | Weights & Biases (`childguard-v3`) |

---

## Klasör Yapısı

```
ChildGuard_v3_Final/
├── backend/
│   ├── api/
│   │   ├── main.py              # FastAPI app + CORS + lifespan
│   │   ├── schemas.py           # Pydantic modelleri
│   │   ├── routers/
│   │   │   ├── analyze.py       # POST /api/analyze
│   │   │   └── explain.py       # POST /api/explain/{shap,lime}
│   │   └── services/
│   │       ├── bert_service.py  # PyTorch BERT yükleme + tahmin
│   │       ├── xgb_service.py   # XGBoost + TF-IDF + feature engineering
│   │       ├── hybrid.py        # Hibrit skor birleştirme
│   │       ├── shap_service.py  # SHAP TreeExplainer
│   │       └── lime_service.py  # LIME LimeTextExplainer
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/                 # Next.js app router
│   │   ├── components/          # Navbar, AnalyzeForm, ResultCard, XAIPanel, HistoryPanel
│   │   └── lib/                 # API istemcisi + localStorage
│   └── Dockerfile
├── final_models/
│   ├── bert_general/            # Aşama 1 BERT (LFS)
│   ├── bert_age_group/          # Aşama 2 BERT (LFS)
│   ├── xgboost_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── age_group_columns.pkl    # XGBoost feature sırası
│   ├── age_label_names.pkl
│   └── bert_general_config.pkl  # Optimal threshold
├── docker-compose.yml
└── childguardhybrid_colab_v4.py # Eğitim scripti (W&B entegrasyonlu)
```

---

## API Endpoint'leri

| Method | Path | Açıklama |
|---|---|---|
| GET | `/api/health` | Model yükleme durumu |
| POST | `/api/analyze` | İki aşamalı tahmin (risk + yaş grubu) |
| POST | `/api/explain/shap` | XGBoost SHAP token ağırlıkları |
| POST | `/api/explain/lime` | BERT LIME token ağırlıkları |

### Örnek istek

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"You are so stupid, nobody likes you","age_group":"Teen"}'
```

```json
{
  "label": "🚨 RİSKLİ / SİBER ZORBALIK",
  "final_score": 0.999,
  "bert_score": 0.999,
  "xgb_score": 0.0,
  "age_group": "Teen",
  "age_prediction": {
    "predicted": "Teen",
    "scores": {"Younger": 0.0009, "Pre-Teen": 0.0005, "Teen": 0.9986}
  }
}
```

---

## Kurulum

### Klonlama (LFS dahil)

```bash
git lfs install
git clone https://github.com/omerozcn/ChildGuardHybrid.git
cd ChildGuardHybrid
```

> BERT modelleri Git LFS ile saklanır. `git lfs install` yapmadan klonlarsan model dosyaları pointer olarak iner.

### Lokal çalıştırma

**Backend:**

```bash
cd backend
pip install -r requirements.txt
MODEL_BASE_PATH=../final_models uvicorn api.main:app --reload --port 8000
```

**Frontend:**

```bash
cd frontend
npm install
npm run dev
```

- Backend: http://localhost:8000 (Swagger: `/docs`)
- Frontend: http://localhost:3000

### Docker

```bash
docker-compose up --build
```

---

## Model Performansı

| Model | Accuracy | F1 (Zararlı) |
|---|---|---|
| BERT — Genel | 0.92 | 0.75 |
| XGBoost | 0.86 | 0.65 |
| BERT — Yaş Grubu | 0.92 | 0.91 (macro) |

---

## Bilinen Sınırlamalar

- LIME yaklaşık 2–5 saniye sürer (frontend'de skeleton loader mevcut).
- `scikit-learn` pickle sürüm uyumsuzluğu uyarısı (1.6.1 → 1.8.0) — runtime'ı etkilemiyor.
- CUDA NVML init uyarısı (CPU üzerinde çalışır, GPU varsa kullanır).

---

## Lisans

Akademik kullanım için açıktır. Detaylı bilgi için iletişime geçin.
