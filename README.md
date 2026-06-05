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

**Hibrit skor (`backend/api/services/hybrid.py`):**

XGBoost'un düşük precision sorununa karşı koruma için **gate** mantığı vardır — XGB skoru 0.65 eşiğinin altındaysa hibrit birleşime katılmaz:

```
XGB skoru ≥ 0.65 → final = 0.6 × bert + 0.4 × xgb
XGB skoru < 0.65 → final = bert
```

Karar eşiği `hybrid.py` modülünde `THRESHOLD = 0.45` olarak sabittir. `final > 0.45` olduğunda içerik **RİSKLİ** olarak işaretlenir ve yaş grubu modeli çalıştırılır.

> Not: Eğitim betiği, BERT bileşeni için F1-maksimum optimal eşiği (0.30) ayrıca öğrenip `bert_general_config.pkl` dosyasına yazar. Bu değer eğitimden gelen bilgilendirici bir çıktıdır; çalışma zamanı kararı yukarıdaki 0.45 sabiti üzerinden alınır.

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

Aşağıdaki değerler bu repoda yer alan eğitilmiş modellerin (`final_models/`) bağımsız test kümesi üzerindeki sonuçlarıdır; Weights & Biases `childguard-v3` projesinden çekilmiştir.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| BERT — Genel (Aşama 1) | 0.9123 | 0.7460 | 0.7556 | 0.7508 |
| XGBoost (Aşama 1) | 0.6034 | 0.2511 | 0.6399 | 0.3607 |
| BERT — Yaş Grubu (Aşama 2, macro) | 0.9502 | 0.9470 | 0.9488 | 0.9478 |

XGBoost'un düşük precision değeri nedeniyle, runtime'da yalnızca skor 0.65 eşiğini aştığında karara dahil edilir (yukarıda anlatılan gate mantığı).

---

## Veri Kümesi

Eğitim, **ChildGuard** veri kümesi üzerinde gerçekleştirilmiştir:

> Kashyap, G. S., Azeez, M. A., Ali, R., Siddiqui, Z. H., Gao, J., & Naseem, U. (2025). *ChildGuard: A Specialized Dataset for Combatting Child-Targeted Hate Speech.* arXiv:2506.21613v2.

351.877 örnek; üç platformdan (Reddit, X, YouTube) toplanmış; üç yaş grubuna (Younger Children, Pre-Teens, Teens) göre etiketlenmiştir. Veri kümesi erişimi için ilgili makalenin GitHub sayfasına bakınız.

---

## Eğitim

`childguardhybrid_colab_v4.py` Google Colab üzerinde W&B entegrasyonlu olarak çalışacak şekilde yazılmıştır. Kendi eğitimini yapmak için:

1. ChildGuard CSV'lerini Google Drive'a koy.
2. Scripti Colab'da aç, ilk hücredeki yolları kendi Drive yapına göre düzenle.
3. Hücreleri sırayla çalıştır — eğitim sonunda `final_models/` klasörü ve `ChildGuard_v3_Bundle.zip` üretilir.

GPU gereklidir (Colab T4/A100 yeterli). Tahmini süre: BERT genel ~45-60 dk, XGBoost + Optuna ~20-30 dk, BERT yaş grubu ~20-30 dk.

---

## Bilinen Sınırlamalar

- LIME yaklaşık 2–5 saniye sürer (frontend'de skeleton loader mevcut).
- `scikit-learn` pickle sürüm uyumsuzluğu uyarısı (1.6.1 → 1.8.0) — runtime'ı etkilemiyor.
- CUDA NVML init uyarısı (CPU üzerinde çalışır, GPU varsa kullanır).

---

## Atıf

Bu repoyu çalışmanızda kullanırsanız hem bu projeye hem de temel aldığımız ChildGuard veri kümesine atıfta bulunmanız beklenir:

```bibtex
@misc{kashyap2025childguard,
  title  = {ChildGuard: A Specialized Dataset for Combatting Child-Targeted Hate Speech},
  author = {Kashyap, Gautam Siddharth and Azeez, Mohammad Anas and Ali, Rafiq and
            Siddiqui, Zohaib Hasan and Gao, Jiechao and Naseem, Usman},
  year   = {2025},
  eprint = {2506.21613},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

---

## Lisans

Akademik kullanım için açıktır. Ticari kullanım veya yeniden dağıtım öncesinde lütfen depo sahibiyle iletişime geçin.
