# Colab'da Eğitim — Adım Adım

Hedef: **A100 (40 GB)**. Her adımı ayrı bir hücrede çalıştırın ve bir sonrakine geçmeden çıktıyı kontrol edin.
Sıra: ortam → kod → testler → duman testi → tam eğitim. Duman testi geçmeden tam eğitime başlamayın.

## Adım 0 — Çalışma zamanı
**Çalışma zamanı → Çalışma zamanı türünü değiştir → A100 GPU** (varsa *Yüksek RAM* açık).

## Adım 1 — Donanım kontrolü
```python
import torch, sys, shutil, psutil
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
print("Python :", sys.version.split()[0])
print("PyTorch:", torch.__version__, "| CUDA:", torch.version.cuda)
assert torch.cuda.is_available(), "GPU yok! Çalışma zamanı türünü A100 yap."
props = torch.cuda.get_device_properties(0)
print("GPU    :", props.name, f"{props.total_memory/2**30:.0f} GB", "| bf16:", torch.cuda.is_bf16_supported())
print("RAM    :", round(psutil.virtual_memory().total/2**30), "GB")
print("Disk   :", round(shutil.disk_usage("/content").free/2**30), "GB boş")
```

## Adım 2 — Drive
```python
from google.colab import drive
import os, shutil
drive.mount("/content/drive")
WORK = "/content/drive/MyDrive/ChildGuard_v4"
for sub in ["models", "reports"]:
    os.makedirs(f"{WORK}/{sub}", exist_ok=True)
# Hugging Face önbelleği YEREL diskte olmalı: Drive'dan geri okunan büyük ağırlık
# dosyaları bozuk gelebiliyor ("header too large"). İndirme zaten çok hızlı.
os.environ["HF_HOME"] = "/content/hf_cache"
os.environ["CG_WORK"] = WORK                      # veri, model ve rapor Drive'a
os.environ["CG_CKPT"] = "/content/cg_checkpoints" # checkpoint'ler hızlı yerel diske (her biri ~4 GB)
print(WORK, os.listdir(WORK))
du = shutil.disk_usage("/content/drive/MyDrive")
print(f"Drive: {du.free/2**30:.0f} GB boş / {du.total/2**30:.0f} GB")
```

## Adım 3 — Kütüphaneler
```python
!pip install -q -U "transformers>=4.44" "datasets>=2.20" accelerate emoji pyarrow scikit-learn matplotlib pytest
import transformers, datasets, sklearn
print("transformers", transformers.__version__, "| datasets", datasets.__version__, "| sklearn", sklearn.__version__)
```

## Adım 4 — Kodu çek
Büyük LFS dosyalarından yalnızca karşılaştırma için gerekenler indirilir (~0,5 GB).
```python
!which git-lfs > /dev/null || (apt-get -qq update && apt-get -qq install -y git-lfs > /dev/null)
%cd /content
!rm -rf ChildGuardHybrid
!GIT_LFS_SKIP_SMUDGE=1 git clone -q https://github.com/omerozcn/ChildGuardHybrid.git
%cd /content/ChildGuardHybrid
!git lfs install --local
!git lfs pull --include="final_models/bert_general/*,data/raw/*"
!git log --oneline -1 && ls
```

## Adım 5 — Birim testleri (~2 dk)
Kodun bu ortamda (Python/kütüphane sürümleri) doğru çalıştığını, eğitime hiç başlamadan doğrular.
```python
!python -m pytest -q tests -p no:cacheprovider
```
Beklenen: `51 passed`. Bir test kırılırsa çıktıyı paylaşın; eğitime geçmeyin.

## Adım 6 — Duman testi (~5–10 dk)
Bütün hattı gerçek veriyle ve gerçek modelle (A100'de `roberta-large`), çok küçük ölçekte çalıştırır:
bellek ön testi, bf16, veri indirme, sözcüksel model, transformer, kalibrasyon, değerlendirme, v3 karşılaştırması, model yükleme.
Ayrı bir klasöre (`$WORK/smoke`) yazar, asıl eğitime karışmaz.
```python
!python -u -m training.run_all --smoke
```
Son satırda **`✅ SMOKE OK`** görmelisiniz. Görmüyorsanız tam eğitime geçmeyin.

## Adım 7 — Tam eğitim (~1–2 saat)
Duman testinde ölçülen hıza göre tahmini süreyi başta yazar. Oturum düşerse **aynı hücreyi tekrar çalıştırın**:
tamamlanan aşamalar (veri, sözcüksel model — Drive'da) atlanır. Checkpoint'ler yerel diskte olduğu için
oturum *tamamen* sıfırlanırsa transformer baştan eğitilir; yalnızca hücre/çekirdek yeniden başlarsa kaldığı yerden devam eder.
```python
!python -u -m training.run_all --profile full
```

## Adım 8 — Sonuçlar
```python
from IPython.display import Markdown, Image, display
display(Markdown(open(f"{WORK}/reports/eval_report.md", encoding="utf-8").read()))
for f in ["pr_curves.png", "gold_recall_by_category.png"]:
    p = f"{WORK}/reports/{f}"
    if os.path.exists(p): display(Image(p))
```

## Adım 9 — Modeli bilgisayara al
Model Drive'da `ChildGuard_v4/models/v4` klasöründe. İsterseniz zip olarak da indirin:
```python
!cd "$WORK" && zip -qr /content/childguard_v4.zip models/v4 reports
from google.colab import files; files.download("/content/childguard_v4.zip")
```
Bilgisayarınızda zip'teki `models/v4` klasörünü depodaki `final_models/v4` olarak koyun; backend onu yükler.

## Sorun giderme
| Belirti | Çözüm |
|---|---|
| `❌ Bellek yetmedi` | Önce `!python -u -m training.train_transformer --bs 16 --grad-accum 2 --fresh` çalıştırın; bitince Adım 7'yi tekrarlayın (transformer aşaması atlanır) |
| `❌ Checkpoint klasöründe farklı bir eğitimin kayıtları var` | Ayarları değiştirdiyseniz `!python -m training.run_all --profile full --redo` |
| Oturum düştü | Adım 2, 3, 4'ü tekrar çalıştırıp Adım 7'yi aynen çalıştırın |
| HF hız sınırı uyarısı | Zararsız; isterseniz `os.environ["HF_TOKEN"]="..."` |
| `SafetensorError: header too large` | Önbellek Drive'da kalmış. `HF_HOME` yerel olmalı (Adım 2); eski önbelleği `!rm -rf "$WORK/hf_cache"` ile silin. Kod bu durumda modeli bir kez kendiliğinden yeniden indirir. |
