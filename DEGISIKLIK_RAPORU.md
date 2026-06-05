# ChildGuard AI — Proje Değişiklik Raporu

Kaynak: dosya zaman damgaları, CLAUDE.md, CLAUDE_v2.md, ChildGuardLastHybrid.ipynb, TEST_REPORT.md, PROJECT_SUMMARY.md

---

## Zaman Çizelgesi

### 11 Nisan 2026 — İlk Sürüm (v1 Taslak)

| Saat  | Dosya/Klasör                  | Değişiklik |
|-------|-------------------------------|------------|
| 20:31 | `childguardhybrid.py`         | İlk lokal eğitim scripti oluşturuldu |
| 20:34 | `childguardhybrid_colab.py`   | Google Colab uyumlu eğitim scripti eklendi |
| 22:27 | `backend/`                    | FastAPI backend (main, schemas, routers, services) kuruldu |
| 22:27 | `docker-compose.yml`          | Docker Compose yapılandırması eklendi |
| 22:27 | `setup.sh`                    | Kurulum scripti oluşturuldu |

**Durum:** Sadece backend var. XAI yok. Frontend yok. Model eğitimi lokal/Colab'da.

---

### 14 Nisan 2026 — v2.0 Kararlı Sürüm

| Saat  | Dosya/Klasör                        | Değişiklik |
|-------|-------------------------------------|------------|
| 10:30 | `CLAUDE_v2.md`                      | v2.0 proje belgesi oluşturuldu |
| 10:33 | `final_models/*`                    | v2 modelleri Drive'dan indirildi (tüm pkl ve BERT klasörleri) |
| 10:40 | `files (1)/`, `files (2)/`          | Zip paketleri çıkarıldı |
| 10:48 | —                                   | files (2) paketi açıldı |
| 10:51 | `frontend/`                         | Next.js 16 frontend eklendi |
| 11:32 | `childguardhybrid_colab_v4.py`      | W&B entegrasyonlu yeni eğitim scripti |
| 11:39 | `BM_-2025-2026_Bahar-Ders-Programi_v5.xlsx` | Ders programı dosyası eklendi |
| 14:29 | `PROJECT_SUMMARY.md`                | Detaylı proje özeti belgesi oluşturuldu |
| 14:30 | `CLAUDE.md`                         | CLAUDE_v2.md → v3.0 formatına yükseltildi |

**Durum:** Frontend + XAI + W&B entegrasyonu tamamlandı. Tüm fazlar tamamlandı.

---

### 28 Nisan 2026 — v3 Eğitim Denemesi ve Kritik Hata Çözümleri

| Saat  | Dosya/Klasör          | Değişiklik |
|-------|-----------------------|------------|
| 13:44 | `untitled7.py`        | Ara script güncellendi |
| 13:45 | `Untitled7.ipynb`     | Deneme notebook'taki mantıksal hatalar düzeltildi. |

**Untitled7.ipynb İçinde Yapılan Kritik Düzeltmeler:**
1. **Veri Sızıntısı (Data Leakage) Çözümü:** XGBoost modelinin eğitim verisinden `Age_Group` (Yaş Grubu) özellikleri tamamen çıkarıldı. Canlı ortamda bilinmeyen bu verinin modelin kararını etkilemesi engellendi.
2. **Aşırı Öğrenme (Overfitting) Engeli:** BERT modellerindeki azınlık sınıfları kopyalama (upsampling) yöntemi silindi, yerine PyTorch `WeightedRandomSampler` entegre edildi.
3. **Emoji ve Bağlam Desteği:** `MAX_LEN` 128'den 256'ya çıkarıldı ve metin temizliğinde emojileri metne döken `emoji` kütüphanesi eklendi.

*(Not: Bu değişiklikler `CLAUDE_v2.md` belgesinde yer almayan, v3 eğitim aşamasına ait yeni düzeltmelerdir.)*

---

### 29 Nisan 2026 — v3 Eğitim Notebook (ChildGuardLastHybrid)

| Saat  | Dosya/Klasör                    | Değişiklik |
|-------|---------------------------------|------------|
| 05:40 | `ChildGuardLastHybrid.ipynb`    | v3 eğitim notebook tamamlandı ve kaydedildi |

**Durum:** v3 eğitimi Google Colab'da tamamlandı. Sonuçlar W&B'e loglandı.

---

## CLAUDE.md Versiyonlar Arası Farklar

### v2.0 → v3.0 (14 Nisan 2026 10:30 → 14:30)

| Alan | CLAUDE_v2.md (v2.0) | CLAUDE.md (v3.0) |
|------|---------------------|------------------|
| Sürüm | v2.0 | v3.0 |
| XGBoost özellikler | "TF-IDF + meta özellikler" (belirsiz) | TF-IDF 5000 özellik + metin uzunluğu + kelime sayısı + yaş grubu one-hot (açık) |
| Threshold | Belirtilmemiş | > 0.45 → RİSKLİ; bert_general_config.pkl'de saklanan: 0.32 |
| XAI (SHAP/LIME) | Yok — "Faz 3'te gelecek" | Tamamlandı: shap_service.py + lime_service.py |
| Frontend | Yok — "Faz 4'te gelecek" | Tamamlandı: Next.js 16 + Tailwind + shadcn/ui + Recharts |
| Frontend bileşenleri | Yok | Navbar, AnalyzeForm, ResultCard, XAIPanel, HistoryPanel |
| W&B Experiment Tracking | Yok | Eklendi (`childguard-v3` projesi) |
| Eğitim scripti | `childguardhybrid_colab_final.py` | `childguardhybrid_colab_v4.py` |
| Docker | Sadece backend | Backend :8000 + Frontend :3000 + volume açıklaması |
| `age_group_columns.pkl` | Belirtildi | KRİTİK olarak işaretlendi |
| Bilinen sorunlar bölümü | Yok | Eklendi (4 madde) |
| Claude Code kuralları | Yok | Eklendi |

---

## Model Performansı Karşılaştırması

### v2 Modelleri (14 Nisan 2026 — mevcut final_models/)

| Model | Accuracy | F1 |
|-------|----------|----|
| BERT Genel | 0.92 | 0.75 |
| XGBoost | 0.86 | 0.65 |
| BERT Yaş Grubu | 0.92 | 0.91 (macro) |

### v3 Modelleri (29 Nisan 2026 — ChildGuardLastHybrid.ipynb çıktısı)

| Model | Accuracy | F1 | Threshold | Değişim |
|-------|----------|----|-----------|---------|
| BERT Genel | 0.9123 | 0.7508 | 0.30 | ≈ sabit |
| XGBoost | 0.6034 | 0.3607 | — | ↓ Ciddi düşüş |
| BERT Yaş Grubu | 0.9502 | 0.9478 (macro) | — | ↑ İyileşme |

> **Not:** XGBoost'un v3'te düşmesinin nedeni dataset değişikliği:
> v2'de önceden temizlenmiş tek CSV (262,598 satır) kullanılırken,
> v3'te 3 kaynak birleştirildi (ChildGuard.csv + lexical + contextual → 276,641 satır)
> ve "Unknown" yaş grubu içeren çok sayıda satır eklendi (toplam 236,818).
> XGBoost bu gürültüye daha duyarlı.

---

## Dataset Evrimi

| Versiyon | Kaynak | Satır | Özellik |
|----------|--------|-------|---------|
| v2 | ChildGuard_Cleaned.csv (tek dosya) | 262,598 | Önceden temizlenmiş |
| v3 | ChildGuard.csv + lexical_chilhate.csv + contextual_childhate.csv | 276,641 | 3 kaynak birleştirildi, null/dup/karakter filtresi uygulandı |

---

## Backend Servisleri Ekleme Tarihi

Tüm backend servisleri 11 Nisan 2026'da oluşturuldu:

| Servis | Dosya | Açıklama |
|--------|-------|---------|
| BERT tahmini | `services/bert_service.py` | PyTorch yükleme + tahmin |
| XGBoost tahmini | `services/xgb_service.py` | XGBoost yükleme + tahmin |
| Hibrit skor | `services/hybrid.py` | bert×0.6 + xgb×0.4 birleştirme |
| SHAP açıklama | `services/shap_service.py` | TreeExplainer |
| LIME açıklama | `services/lime_service.py` | LimeTextExplainer |

> **Not:** SHAP/LIME servisleri backend'de 11 Nisan'da oluşturuldu ancak CLAUDE.md'ye
> "tamamlandı" olarak ancak 14 Nisan'da (v3 güncellemesinde) işaretlendi.

---

## Test Raporu (14 Nisan 2026)

`TEST_REPORT.md` — Tüm endpoint'ler test edildi, 6/6 PASS:

| Endpoint | Süre | Sonuç |
|----------|------|-------|
| GET /api/health | 0.001s | PASS |
| POST /api/analyze (zararlı) | 0.116s | PASS — %99.5 güven |
| POST /api/analyze (güvenli) | 0.059s | PASS — %13.6 risk |
| POST /api/explain/shap | 0.021s | PASS (global vocab tokenleri görünüyor) |
| POST /api/explain/lime | 2.741s | PASS |
| Frontend :3000 | 0.028s | PASS |

---

## Bilinen Açık Sorunlar (CLAUDE.md'den)

1. **Dolaylı zorbalık** — "give up", "disappear" gibi ifadeler BERT'ten kaçıyor → v3 eğitimi hedef
2. **SHAP token alakasızlığı** — Global TF-IDF vocab'a dayandığından giriş metniyle ilgisiz tokenlar çıkabiliyor → shap_service.py'de kısmi düzeltme var
3. **LIME gecikme** — ~2-5 saniye → frontend'de skeleton loader mevcut
4. **Docker frontend** — `next.config.js`'de `output: standalone` eksik → production hazır değil

---

*Rapor tarihi: 29 Nisan 2026*
