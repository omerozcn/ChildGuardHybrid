# ChildGuard AI v4

İngilizce çevrim içi mesajlarda çocuklar için zararlı içeriği tespit eden, **çok etiketli** ve açıklanabilir bir sınıflandırma sistemi.

> Afyon Kocatepe Üniversitesi — Yapay Zeka dersi final projesi.

Tek bir "zararlı mı?" kararı yerine her mesaj için risk türleri ayrı ayrı puanlanır:

| Etiket | Anlam |
|---|---|
| Zararlı içerik | Genel karar (risk seviyesi: düşük / orta / yüksek) |
| Hakaret / aşağılama | Küçümseme, alay, dışlama dahil |
| Tehdit / kendine zarar | Şiddet tehdidi ve "kys" tipi kendine zarar teşviki |
| Cinsel içerik | |
| Kimlik nefreti | Irk, din, cinsiyet, engellilik vb. |
| Küfür | Tek başına zararlı sayılmaz |
| Çocuğu hedef alıyor | Deneysel yardımcı sinyal |

## v3'ten farkı

| | v3 | v4 |
|---|---|---|
| Veri | ChildGuard CSV (etiket kalitesi sorunlu, bkz. [docs/DATA_AUDIT.md](docs/DATA_AUDIT.md)) | Civil Comments + Measuring Hate Speech + ToxiGen + elle yazılmış çocuk güvenliği seti |
| Çıktı | Zararlı / değil + (güvenilmez) yaş grubu | 7 risk türü + risk seviyesi + kural eşleşmeleri + dil uyarısı |
| Model | BERT + XGBoost, elle seçilmiş 0.6/0.4 ağırlık | RoBERTa (çok etiketli) + karakter n-gram LR, doğrulamada öğrenilen yığınlama |
| Eşik seçimi | Test setinde (iyimser skorlar) | Doğrulama setinde, güvenlik etiketlerinde recall ≥ 0.90 |
| Gizlenmiş yazım ("st00pid", "k y s") | Yakalanmıyor | Normalizasyon + karakter n-gramları |
| Kendine zarar teşviki | Modele bağlı | Ek olarak deterministik kural katmanı |
| Açıklama | SHAP (alakasız sözcükler) + LIME (yavaş) | Kelime çıkarma (tüm model) + kesin doğrusal katkılar |
| Değerlendirme | Tek test seti | Test + 150 cümlelik altın set (10 kategori) + v3 ile karşılaştırma |
| Testler | Yok | 51 pytest testi (çekirdek, API, eğitim hattının tamamı) |

Tasarım kararları ve kaynaklar: [docs/RESEARCH.md](docs/RESEARCH.md)

## Mimari

```
mesaj → normalizasyon ─┬─ RoBERTa (çok etiketli) ──────────┐
                       └─ TF-IDF kelime+karakter → LR ─────┤
                                                           ▼
                                   yığınlama + kalibrasyon (etiket başına)
                                                           ▼
                     kural katmanı (kendine zarar / tehdit) → eşikler → karar
```

## Klasör yapısı

```
childguard/        Çekirdek paket (eğitim ve backend ortak kullanır)
  labels.py        Etiket şeması
  text.py          Normalizasyon (emoji, gizleme çözme, dil kontrolü)
  rules.py         Yüksek kesinlikli güvenlik kuralları
  pipeline.py      Uçtan uca tahmin + açıklama
training/          Veri hazırlama, eğitim, kalibrasyon, değerlendirme
  run_all.py       Tek komutla bütün hat (--smoke ile duman testi)
  colab_train.ipynb
backend/           FastAPI servisi
frontend/          Next.js arayüzü
data/
  gold/            Altın test seti (eğitimde kullanılmaz)
  supplement/      Çocuk güvenliği ek seti
  raw/             Eski ChildGuard verisi (yalnızca bilgi amaçlı değerlendirme)
tests/             pytest
docs/              Araştırma notları, veri denetimi, Colab rehberi
final_models/      v3 modelleri (karşılaştırma için) ve v4/ (eğitimden sonra)
legacy/            v3 eğitim betiği
```

## Eğitim

**Colab (önerilen, A100):** adım adım rehber → [docs/COLAB.md](docs/COLAB.md)

**Yerel:**

```bash
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/pip install -r training/requirements.txt
.venv/Scripts/python -m pytest -q tests
.venv/Scripts/python -m training.run_all --smoke      # birkaç dakikalık uçtan uca deneme
.venv/Scripts/python -m training.run_all --profile small
```

Model, batch boyutu ve hassasiyet GPU'ya göre otomatik seçilir (A100: `roberta-large` + bf16, ≤16 GB: `roberta-base`). Eğitim kesilirse aynı komut kaldığı yerden devam eder.

## Çalıştırma

Eğitimden sonra `final_models/v4/` içinde `config.json`, `lexical.joblib` ve `transformer/` bulunur.

```bash
# Backend (depo kökünden)
.venv/Scripts/pip install -r backend/requirements.txt
cd backend && ../.venv/Scripts/uvicorn api.main:app --reload --port 8000

# Frontend
cd frontend && npm install && npm run dev
```

Docker: `docker-compose up --build` (backend :8000, frontend :3000).

### API

| Method | Path | Açıklama |
|---|---|---|
| GET | `/api/health` | Model durumu |
| POST | `/api/analyze` | `{"text": "..."}` → risk seviyesi, 6 kategori, hedef çocuk sinyali, kural eşleşmeleri |
| POST | `/api/analyze/batch` | `{"texts": [...]}` (en fazla 64) |
| POST | `/api/explain` | `{"text": "...", "label": "toxicity"}` → kelime etkileri + n-gram katkıları |

## Bilinen sınırlamalar

- Yalnızca İngilizce; diğer dillerde arayüz uyarı gösterir.
- Tek mesaja bakar, konuşma bağlamını kullanmaz.
- Grooming ayrı bir problemdir; `sexual` etiketi açık cinsel içeriği hedefler.
- "Çocuğu hedef alıyor" sinyali az sayıda örnekle öğrenilir (deneysel).
- Altın set tek kişi tarafından etiketlendi; ikinci bir gözden geçirme önerilir.
- ToxiGen yalnızca araştırma amaçlı lisanslıdır; ticari kullanımda `--no-toxigen` ile yeniden eğitin.

## Atıf

Kullanılan veri setleri: Civil Comments (Jigsaw, CC0), Measuring Hate Speech (UC Berkeley D-Lab, CC-BY-4.0), ToxiGen (Hartvigsen et al., 2022). Eski veri: Kashyap et al. (2025), *ChildGuard*, arXiv:2506.21613.

## Lisans

Akademik kullanım için açıktır. Ticari kullanım öncesinde depo sahibiyle ve kullanılan veri setlerinin lisanslarıyla kontrol edin.
