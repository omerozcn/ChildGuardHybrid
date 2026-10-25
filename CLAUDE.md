# ChildGuard AI v4

İngilizce mesajlarda çok etiketli zararlı içerik tespiti (çocuk güvenliği odaklı).
Afyon Kocatepe Üniversitesi, Yapay Zeka dersi final projesi.

## Mimari

- `childguard/` çekirdek paket: etiketler, normalizasyon, kural katmanı, tahmin hattı. Eğitim ve backend **aynı** kodu kullanır; normalizasyonu tek yerde değiştir.
- Model: RoBERTa (çok etiketli) + TF-IDF kelime/karakter → LR, doğrulamada öğrenilen yığınlama, ardından kural tabanları ve etiket başına eşikler.
- Etiketler: toxicity, insult, threat, sexual, identity_hate, profanity, targets_minor. Bilgi taşımayan hücreler `-1` (kayıpta maskelenir).
- Model klasörü: `final_models/v4/{config.json, lexical.joblib, transformer/}`. `final_models/bert_*` v3 modelleridir; yalnızca karşılaştırma için.

## Kurallar

- Eşik veya hiperparametre **asla** test setinde seçilmez; `calibrate.py` doğrulamada çalışır, `evaluate.py` testi bir kez kullanır.
- `data/gold/gold_test.csv` eğitime girmez; `data/supplement/` ile örtüşmediği testle doğrulanır.
- Eğitim yolları `training/paths.py` üzerinden; Colab'da `CG_WORK` ortam değişkeni Drive klasörünü gösterir.
- Kod değiştikten sonra: `.venv/Scripts/python -m pytest -q tests` (eğitim hattının uçtan uca testi dahil).
- Uzun eğitimden önce: `python -m training.run_all --smoke`.

## Çalıştırma

```bash
.venv/Scripts/python -m pytest -q tests
.venv/Scripts/python -m training.run_all --smoke
.venv/Scripts/python -m training.run_all --profile small     # yerel
# Colab: docs/COLAB.md
cd backend && ../.venv/Scripts/uvicorn api.main:app --reload --port 8000
cd frontend && npm run dev
```

## Claude Code Kuralları
- Kısa ve öz ol, açıklama yapma
- Sadece hata varsa göster
- Özet yazma, direkt yap
