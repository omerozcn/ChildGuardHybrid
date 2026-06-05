# ChildGuard AI v3.0

Çevrim içi metinlerde çocuklara yönelik zararlı içerik tespiti.
İki aşamalı hibrit sistem: BERT + XGBoost.
Afyon Kocatepe Üniversitesi, Veri Madenciliği dersi final projesi.

## Mimari

**Aşama 1 — Zararlı mı?**
- `bert_general` (PyTorch) → zararlı olasılığı
- `xgboost_model` → TF-IDF (5000 özellik) + metin uzunluğu + kelime sayısı + yaş grubu one-hot
- Hibrit skor: `final = bert * 0.6 + xgb * 0.4`, eşik > 0.45 → RİSKLİ (`bert_general_config.pkl`'den okunur, saklanan değer: 0.32)

**Aşama 2 — Kime yönelik?** (sadece zararlıysa)
- `bert_age_group` → Younger / Pre-Teen / Teen

## Klasör Yapısı

```
childguardupdated/
├── backend/
│   ├── api/
│   │   ├── main.py              # FastAPI, CORS, lifespan
│   │   ├── schemas.py           # Pydantic modeller
│   │   ├── routers/
│   │   │   ├── analyze.py       # POST /api/analyze
│   │   │   └── explain.py       # POST /api/explain/shap|lime
│   │   └── services/
│   │       ├── bert_service.py  # BERT yükleme + tahmin (PyTorch)
│   │       ├── xgb_service.py   # XGBoost yükleme + tahmin
│   │       ├── hybrid.py        # Hibrit skor birleştirme
│   │       ├── shap_service.py  # SHAP açıklama (TreeExplainer)
│   │       └── lime_service.py  # LIME açıklama (LimeTextExplainer)
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx         # Ana sayfa
│   │   │   └── layout.tsx
│   │   ├── components/
│   │   │   ├── Navbar.tsx       # Model durum göstergesi
│   │   │   ├── AnalyzeForm.tsx  # Metin input + yaş grubu seçimi
│   │   │   ├── ResultCard.tsx   # Skor gösterimi + yaş tahmini
│   │   │   ├── XAIPanel.tsx     # SHAP/LIME panel + highlight + chart
│   │   │   └── HistoryPanel.tsx # localStorage geçmişi
│   │   └── lib/
│   │       ├── api.ts           # Backend API istemcisi
│   │       └── history.ts       # localStorage yönetimi
│   └── Dockerfile
├── final_models/                # Eğitilmiş modeller (Docker volume)
│   ├── bert_general/
│   ├── bert_age_group/
│   ├── xgboost_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── age_group_columns.pkl   # KRİTİK — XGBoost feature sırası
│   ├── age_label_names.pkl
│   └── bert_general_config.pkl  # optimal threshold
├── docker-compose.yml
└── childguardhybrid_colab_v4.py # Güncel eğitim scripti (W&B entegrasyonlu)
```

## Teknolojiler

- Backend: FastAPI + Uvicorn (port 8000)
- ML: PyTorch + HuggingFace Transformers (`bert-base-uncased`), XGBoost, scikit-learn TF-IDF
- XAI: SHAP (XGBoost TreeExplainer), LIME (BERT LimeTextExplainer)
- Frontend: Next.js 16 + TypeScript + Tailwind CSS + shadcn/ui + Recharts
- Deploy: Docker Compose (backend :8000, frontend :3000)
- Experiment Tracking: Weights & Biases (W&B, `childguard-v3` projesi)

## API Endpoint'leri

| Method | Path | Açıklama |
|---|---|---|
| GET | `/api/health` | Model yükleme durumu |
| POST | `/api/analyze` | Ana analiz — iki aşamalı tahmin |
| POST | `/api/explain/shap` | XGBoost SHAP açıklaması |
| POST | `/api/explain/lime` | BERT LIME açıklaması |

## Faz Durumu

- [x] Faz 1 — Eğitim (XGBoost + BERT) — v2 tamamlandı, v3 devam ediyor
- [x] Faz 2 — FastAPI backend
- [x] Faz 3 — SHAP + LIME servisleri
- [x] Faz 4 — Next.js frontend
- [ ] v3 modelleri eğitilince `final_models/` güncellenmeli

## Çalıştırma

```bash
# Lokal
cd backend
pip install -r requirements.txt
MODEL_BASE_PATH=../final_models uvicorn api.main:app --reload --port 8000

cd frontend
npm install
npm run dev

# Docker
docker-compose up --build
# Backend: http://localhost:8000  |  Frontend: http://localhost:3000
```

## Model Performansı (v2 — mevcut modeller)

| Model | Accuracy | F1 (Zararlı) |
|---|---|---|
| BERT Genel | 0.92 | 0.75 |
| XGBoost | 0.86 | 0.65 |
| BERT Yaş Grubu | 0.92 | 0.91 (macro) |

## Bilinen Sorunlar

- BERT dolaylı zorbalığı ("give up", "disappear") kaçırıyor → v3 eğitimi hedef
- XGBoost SHAP çıktısı alakasız tokenlar döndürebiliyor → `shap_service.py`'de kısmi düzeltme
- LIME ~2-5 saniye sürüyor → frontend'de skeleton loader mevcut
- Docker frontend `next.config.js`'de `output: standalone` gerektirir → henüz eklenmedi

## Claude Code Kuralları
- Kısa ve öz ol, açıklama yapma
- Sadece hata varsa göster
- Özet yazma, direkt yap
