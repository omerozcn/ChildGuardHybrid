# v4 Tasarım Kararları ve Kaynaklar

## 1. Görev tanımı

**İngilizce mesajlarda çok etiketli risk tespiti.** Tek bir "zararlı mı?" kararı yerine:

| Etiket | Anlam | Ana kaynak |
|---|---|---|
| `toxicity` | Genel zararlılık (ana karar) | Civil Comments, MHS, ToxiGen, ek set |
| `insult` | Hakaret, aşağılama, küçümseme | Civil Comments, MHS (insult/humiliate) |
| `threat` | Şiddet tehdidi ve kendine zarar teşviki | Civil Comments, MHS (violence), ek set, kurallar |
| `sexual` | Cinsel içerik | Civil Comments (sexual_explicit), ToxiGen (lewd) |
| `identity_hate` | Kimliğe yönelik nefret | Civil Comments (identity_attack), MHS, ToxiGen |
| `profanity` | Küfür | Civil Comments (obscene) |
| `targets_minor` | Hedef bir çocuk/genç (deneysel) | MHS (target_age_children / teenagers), ek set |

Bir kaynağın bilgi taşımadığı etiketler **maskelenir** (kayba girmez); böylece farklı şemalara sahip veri setleri tek modelde birleşebilir.

## 2. Veri kaynakları

| Veri seti | Boyut | Lisans | Neden |
|---|---|---|---|
| [google/civil_comments](https://huggingface.co/datasets/google/civil_comments) | ~2M yorum | CC0-1.0 | 7 alt tür için annotatör oranları (yumuşak etiket); kendi train/val/test bölmeleri |
| [ucberkeley-dlab/measuring-hate-speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) | 39.565 yorum / 135.556 annotasyon | CC-BY-4.0 | Hedef yaş grubu sütunları (çocuk/genç), hakaret/aşağılama/şiddet ölçekleri |
| [toxigen/toxigen-data](https://huggingface.co/datasets/toxigen/toxigen-data) (annotated) | ~9,9K | Yalnızca araştırma | Örtük (implicit) nefret — en zayıf noktamız |
| `data/supplement/child_safety_supplement.csv` | ~115 | Proje | Kendine zarar teşviki, çocukları küçümseme, gizlenmiş yazım ve zor negatifler |

Değerlendirilip **alınmayanlar:**

* **HateXplain** — Hugging Face'te yükleme betiğine bağlı; güncel `datasets` sürümleri betik çalıştırmıyor. İleride parquet sürümü bulunursa eklenebilir.
* **Implicit Hate Corpus (SALT-NLP)** — erişim form gerektiriyor.
* **Cyberbullying Classification (Wang et al. 2020)** — lisansı belirsiz.
* **Kendine zarar teşviki** için uygun, açık lisanslı bir veri seti bulunamadı. Bu boşluk ek set + yüksek kesinlikli kural katmanı ile kapatıldı.

## 3. Model mimarisi

```
metin ─┬─ normalizasyon (childguard/text.py) ─┬─ Transformer (roberta-base, çok etiketli) ─┐
       │                                     └─ TF-IDF kelime+karakter → LR (etiket başına) ┤
       │                                                                                    ▼
       │                                            Yığınlama + kalibrasyon (etiket başına LR)
       │                                                                                    ▼
       └──────────────── kural katmanı (kendine zarar / tehdit tabanları) ──► eşikler ──► karar
```

* **Transformer:** varsayılan `roberta-base`. Civil Comments kıyaslamasında BERT ailesi modellerin benzer sonuç verdiği, RoBERTa'nın en iyiler arasında olduğu raporlanmıştır (Duchêne et al., 2023, arXiv:2301.11125). 4 GB VRAM'de fp16 ile eğitilebiliyor (ölçüm: tepe 2,4 GB, ~96 örnek/s @128 belirteç).
* **Sözcüksel model:** Karakter n-gramları (2–5) "st00pid", "l o s e r" gibi gizlemeleri yakalar; doğrusal olduğu için açıklaması kesindir.
* **Yığınlama:** v3'teki elle seçilmiş 0.6/0.4 ağırlıklar ve 0.65 "kapı" yerine, ağırlıklar doğrulama setinde lojistik regresyonla öğrenilir. Bu aynı zamanda olasılıkları kalibre eder.
* **Kural katmanı:** "kys", "kill yourself", "nobody would miss you" gibi kalıplar için deterministik taban. Olumsuzluk ("don't hurt yourself") ve aktarılan söz ("if someone tells you to…") istisnaları vardır.

## 4. Değerlendirme hijyeni

v3'teki iki hata düzeltildi:

1. BERT eşiği **test setinde** aranıyordu → artık doğrulama setinde.
2. XGBoost Optuna araması **test F1'ini** maksimize ediyordu → artık hiperparametreler doğrulama setinde.

Eşik politikası (çocuk güvenliğinde kaçırmak daha pahalıdır):

* `toxicity`, `threat`, `sexual`: doğrulamada **recall ≥ 0.90** sağlayan en yüksek eşik.
* Diğerleri: F1'i en yüksek yapan eşik.
* "Yüksek risk": precision ≥ 0.90 sağlayan en düşük eşik.

**Altın set** (`data/gold/gold_test.csv`): eğitimde hiç kullanılmayan, elle yazılmış 150 gerçekçi mesaj; 10 kategori. Ek setle örtüşmediği testle doğrulanır. Etiketler tek kişi tarafından yazıldı; projeyi geliştirirken ikinci bir kişinin gözden geçirmesi önerilir.

## 5. Sürüm geçmişi

**v4.0** — ilk tam eğitim (roberta-large, A100, 54 dk). Altın sette yakalama oranı v3'ün 0,30'undan 0,77'ye çıktı; ancak zor negatiflerde 14/20 yanlış alarm vardı. Sebep: tehdit eşiği "recall ≥ 0,90" kuralıyla 0,026'ya inmişti ve her türün düşük eşiği genel kararı yükseltiyordu.

**v4.1** — yeniden eğitim yok, yalnızca kalibrasyon: güvenlik etiketlerinde hedef recall noktasında precision < 0,40 ise F2-optimum eşik; genel kararı yalnızca türlerin yüksek risk eşiği yükseltir. Test F1 0,75 → 0,77, zor negatif yanlış alarmları 14 → 6. Yan etki: cinsel içerikte genel karar 3/6 → 0/6.

**v4.2** — veri ve kural:
* Tek başına küfür zararlı sayılmaz (Civil Comments'ta küfür yüksek, diğer türler düşükse toxicity düşürülür); küfür kategorisi korunur.
* `training/augment.py`: örtük küçümseme, dışlama, grooming/cinsel davet, örtük kendine zarar ve zor negatifler (kendini kötüleme, mağdur anlatımı, deyimler, küfürlü ünlemler, destek mesajları) için ~4.200 şablon cümle; eğitimde 3 kez tekrarlanır. Altın setle kelime örtüşmesi (Jaccard) en fazla 0,5.
* Hangi türün genel kararı hangi eşikte yükselteceği doğrulama setinde seçilir (`raise_overall`).
* Uyarı: şablonlar ve altın set aynı kişi tarafından yazıldı; altın set sonuçları iyimser olabilir. Bağımsız biri tarafından yazılmış yeni bir test seti önerilir.

## 6. Bilinen sınırlamalar

* Yalnızca İngilizce. Türkçe ve diğer diller için arayüzde uyarı gösterilir.
* Tek mesaja bakar; konuşma bağlamı (önceki mesajlar) kullanılmaz.
* Grooming (çocuk istismarı amaçlı yakınlaşma) ayrı bir problemdir; `sexual` etiketi yalnızca açık cinsel içeriği hedefler.
* `targets_minor` sinyali az sayıda pozitif örnekle öğrenilir; deneysel olarak işaretlenmiştir.
* ToxiGen'in lisansı yalnızca araştırma kullanımına izin verir. Ticari kullanımda `--no-toxigen` ile yeniden eğitilmelidir.
