# ChildGuard AI v3.0 — Proje Özeti

## Proje Nedir?

Çevrim içi metinlerde çocuklara yönelik zararlı içerik tespiti.
İki aşamalı hibrit sistem: BERT + XGBoost.
Afyon Kocatepe Üniversitesi, Yapay Zeka dersi final projesi.

---

## Mimari

### Aşama 1 — Zararlı mı?
- `bert_general` (PyTorch) → zararlı olasılığı
- `xgboost_model` → TF-IDF (5000 özellik) + metin uzunluğu + kelime sayısı + yaş grubu one-hot
- Hibrit skor: `final = bert * 0.6 + xgb * 0.4`
- Eşik: `> 0.45` → RİSKLİ

### Aşama 2 — Kime yönelik? (sadece zararlıysa)
- `bert_age_group` → Younger / Pre-Teen / Teen

---

## Teknoloji Stack

- **Backend:** FastAPI + Uvicorn (port 8000)
- **ML:** PyTorch BERT (`bert-base-uncased`), XGBoost, scikit-learn TF-IDF
- **XAI:** SHAP (XGBoost için TreeExplainer), LIME (BERT için LimeTextExplainer)
- **Frontend:** Next.js 16 + TypeScript + Tailwind CSS + shadcn/ui + Recharts
- **Deploy:** Docker Compose (backend :8000, frontend :3000)
- **Experiment Tracking:** Weights & Biases (W&B)

---

## Proje Yapısı

```
ChildGuard/
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
│   │       ├── shap_service.py  # SHAP açıklama
│   │       └── lime_service.py  # LIME açıklama
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
│   ├── age_group_columns.pkl
│   ├── age_label_names.pkl
│   └── bert_general_config.pkl  # optimal threshold
├── docker-compose.yml
├── CLAUDE.md
└── childguardhybrid_colab_v4.py # Güncel eğitim scripti (W&B entegrasyonlu)
```

---

## API Endpoint'leri

| Method | Path | Açıklama |
|---|---|---|
| GET | `/api/health` | Model yükleme durumu |
| POST | `/api/analyze` | Ana analiz — iki aşamalı tahmin |
| POST | `/api/explain/shap` | XGBoost SHAP açıklaması |
| POST | `/api/explain/lime` | BERT LIME açıklaması |

### Örnek /api/analyze isteği
```json
{
  "text": "You are stupid and ugly, everyone hates you.",
  "age_group": "Teen"
}
```

### Örnek /api/analyze yanıtı
```json
{
  "label": "🚨 RİSKLİ / SİBER ZORBALIK",
  "final_score": 0.9954,
  "bert_score": 0.9961,
  "xgb_score": 0.9944,
  "age_group": "Teen",
  "age_prediction": {
    "predicted": "Teen",
    "scores": {"Younger": 0.001, "Pre-Teen": 0.001, "Teen": 0.998}
  }
}
```

---

## Model Performansı (v2 — mevcut modeller)

| Model | Accuracy | F1 (Zararlı) |
|---|---|---|
| BERT Genel | 0.92 | 0.75 |
| XGBoost | 0.86 | 0.65 |
| BERT Yaş Grubu | 0.92 | 0.91 (macro) |

---

## Dataset Geçmişi

### v2 (mevcut)
- Kaynak: `ChildGuard_Cleaned.csv` (önceden temizlenmiş)
- Satır: 262,598
- Sorun: Dolaylı zorbalık örnekleri az → BERT implicit hate'i kaçırıyor

### v3 (yeni eğitim — devam ediyor)
- Kaynak: Orijinal repo'dan 3 dosya birleştirildi
  - `ChildGuard.csv` (351k) + `lexical_chilhate.csv` (194k) + `contextual_childhate.csv` (157k)
- Temizleme: null kaldırma, 10–1000 karakter filtresi, duplicate temizleme
- Hedef: Implicit hate tespitini iyileştirmek
- Experiment tracking: W&B (`childguard-v3` projesi)
- Eğitim scripti: `childguardhybrid_colab_v4.py`

---

## Faz Durumu

- [x] Faz 1 — Eğitim (XGBoost + BERT) — v2 tamamlandı, v3 devam ediyor
- [x] Faz 2 — FastAPI backend
- [x] Faz 3 — SHAP + LIME servisleri
- [x] Faz 4 — Next.js frontend
- [ ] v3 modelleri eğitilince final_models/ güncellenmeli

---

## Çalıştırma

```bash
# Lokal geliştirme
cd backend
pip install -r requirements.txt
MODEL_BASE_PATH=../final_models uvicorn api.main:app --reload --port 8000

cd frontend
npm install
npm run dev

# Docker ile
docker-compose up --build
```

---

## Bilinen Sorunlar / Yapılacaklar

- BERT dolaylı zorbalığı ("give up", "disappear", "just a joke") kaçırıyor
  → v3 eğitimi bu sorunu çözmesi hedefleniyor
- XGBoost SHAP çıktısı giriş metniyle alakasız tokenlar döndürebiliyor
  → shap_service.py'de kısmi düzeltme yapıldı (giriş tokenları öne alınıyor)
- LIME ~2-5 saniye sürüyor → frontend'de skeleton loader mevcut
- Docker frontend Dockerfile'ı `next.config.js`'de `output: standalone` gerektirir
  → henüz eklenmedi

---

## Önemli Notlar

- BERT modelleri PyTorch tabanlı — TensorFlow kullanılmıyor
- `age_group_columns.pkl` kritik — XGBoost feature sırası buradan okunuyor
- `bert_general_config.pkl` optimal threshold'u saklıyor (0.32)
- v3 eğitimi bittikten sonra `final_models/` klasörü ZIP'ten güncellenmeli
- W&B run: `wandb.ai` → `childguard-v3` projesi
