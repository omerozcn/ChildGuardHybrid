# Eski ChildGuard Verisinin Denetimi

`data/raw/ChildGuard_Cleaned.csv` (262.598 satır) üzerinde yapılan inceleme.
v4'te bu verinin neden varsayılan olarak eğitime **katılmadığını** açıklar.

## Bulgular

| Ölçüm | Değer | Neden önemli |
|---|---|---|
| Zararlı (1) satırlarda çocuk kelimesi geçme oranı (kid, baby, child, teen, brat, toddler, school) | **%4,8** | Veri "çocuklara yönelik nefret" olarak tanıtılıyor; makaledeki etiketleme kuralına göre çocuk referansı olmayan içerik 0 olmalı. Etiketler pratikte genel toksikliği temsil ediyor. |
| Yaş grubu "Unknown" olan satırlar | **%83,8** (220.043) | Makale yaşı belirlenemeyen satırların atıldığını söylüyor. |
| "Younger Children" satırlarında kendi yaş ipucunun (toddler, baby, preschool…) geçme oranı | **%4,0** | Yaş etiketleri metin içeriğiyle neredeyse ilişkisiz. |
| "Pre-Teens" satırlarında yaş ipucu | **%0,9** | 〃 |
| "Teens" satırlarında yaş ipucu | **%1,8** | 〃 |
| Ortanca kelime sayısı: Younger / diğer gruplar | **52 / 26** | Yaş modeli yaşı değil metin uzunluğu gibi biçim izlerini öğrenebilir; v3'teki 0,95 F1 bu yüzden yanıltıcı olabilir. |
| Wikipedia tartışma sayfası izi taşıyan satırlar (utc, wikipedia, talk page…) | %27 (Unknown) – %52 (Younger) | Veri büyük ölçüde başka bir toksiklik derleminden geliyor. |
| İngilizce dışı (Hırvatça/Sırpça) görünen satırlar | ~1.100 | Makale yalnızca İngilizce içerik tutulduğunu söylüyor. |
| Biçim | Karışık | Bir kısım kök haline getirilmiş ve stop-word'leri atılmış ("personally agree sailor cuteness favour"), bir kısım ham. Kullanıcı girdisi her zaman ham olduğundan eğitim/kullanım uyumsuzluğu doğar. |
| `predicted_class` sütunu | `actual_class` ile %86 aynı | Bir modelin tahmini; eğitimde kullanılmamalı. |

## Makaledeki tutarsızlıklar (arXiv:2506.21613v2)

* Lexical/contextual alt küme boyutları (157K / 194K) bölümler arasında yer değiştiriyor.
* Yaş grubu sayıları metin ile Tablo 5 arasında tutmuyor (ör. Teens 19.874 ↔ 11.479).
* Kaynakçada "NOTE: This appears to be a placeholder citation" notları duruyor.
* Raporlanan BERT F1 (~%54) ile bu projedeki v3 BERT'in F1'i (0,75) farklı metrik ve veriyle ölçüldüğü için karşılaştırılamıyor.

## v4'teki karar

* Eski veri **eğitime girmez** (istenirse `prepare_data --with-legacy`).
* `evaluate.py` bu veriyi temizlenmiş haliyle (wiki izleri, ASCII dışı ve aşırı kısa/uzun satırlar çıkarılarak) yalnızca **bilgi amaçlı** bir ek değerlendirme olarak raporlar.
* Yaş grubu aşaması kaldırıldı; yerine insan etiketli `target_age_children / target_age_teenagers` sütunlarından (Measuring Hate Speech) öğrenilen deneysel **"çocuğu hedef alıyor"** sinyali geldi.

Denetimi tekrar üretmek için kullanılan ölçümler `docs/RESEARCH.md` içinde açıklanmıştır.
