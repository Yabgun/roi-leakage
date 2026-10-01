# Modeller: hangisi ne yapar, nerede tanımlı, ağırlığı nerede

Çalışmada tek bir model yoktur. Makalenin iki sorusu vardır. Her soru farklı modellerle cevaplanır:

1. **Sızıntı (saldırı tarafı):** Π_ROI'nin sunucuya açtığı bilgi teşhisi ele veriyor mu? Bunu ölçen modeller
   **saldırganlardır**. Sunucunun gördüğünden teşhisi tahmin ederler. Başarılı olmaları sızıntı demektir.
2. **Çözüm (FoveaHE):** Sızıntı olmadan, hızlı ve doğru şifreli teşhis mümkün mü? Bunu gösteren modeller
   hastanenin istediği **şifreli teşhis modelleridir**. Başarılı olmaları istenir.

**Makalenin iki ana modeli:**
- Sızıntıyı gösteren: **bağlam saldırganı (ResNet-18), Π_ROI görüşü** (Tablo III, "Bağlam" satırı).
- Önerilen yöntemin modeli: **FoveaHE F32_G16 + Model D** (beyin MR) ve **+ Model D2** (göğüs röntgeni) (Tablo V, VIII).

Etiketler ve her modelin girdisi/çıktısı: `docs/ETIKETLER.md`.

## Model haritası

| Model | Makalede | Rolü | Mimari | Girdi | Tanım (kod) | Eğitim / çalıştırma |
|---|---|---|---|---|---|---|
| Meta veri saldırganı | Tablo II | Saldırgan | HistGradientBoosting (300 yineleme, öğrenme oranı 0.05, 15 yaprak) | ROI konum/büyüklük/şekil + girdi boyutu (23 öznitelik) | `experiments/attack_metadata.py:29-31`, `attacks/metadata.py` | `python -m experiments.attack_metadata` (tablo), `python -m analysis.saldiri_a_tahmin` (tahmin kaydı) |
| Bağlam saldırganı | Tablo III, VIII | Saldırgan | ImageNet ön eğitimli ResNet-18, son katman 3 sınıf | Sunucunun gördüğü 224×224 görüntü (ROI sıfır) + gizli bölge göstergesi | `attacks/context_cnn.py:150-153` | `python -m experiments.attack_context` (AdamW, OneCycle en çok 3e-4, yığın 64; beyin 12, COVID-QU-Ex 5 epoch) |
| ResNet-18 bilgi üst sınırı | Metin (FoveaHE bilgi düzeyi) | Ölçüm aracı | Bağlam saldırganıyla aynı ağ, gizli bölge yok | FoveaHE temsilinden geri çizilen 224×224 görüntü | `attacks/context_cnn.py:150-196` | `python -m experiments.fovea_info` |
| Π_ROI hesaplaması (kurban) | Bulgular, maliyet | Yeniden üretim | İki lineer tur: M1 (20×n), M2 (10×20), rastgele ağırlık | Seçici şifreli görüntü | `he/piroi.py:45-112` | `python -m experiments.piroi_benchmark` (eğitim yok) |
| Model D | Tablo V, VIII | Şifreli teşhis | İki tam bağlantılı katman, 20 gizli birim, aktivasyon yok (Π_ROI'nin iki turunun eğitilmiş karşılığı) | Tam görüntü (= Π_ROI'nin teşhis modeli) ya da FoveaHE temsili | `foveahe/he_models.py:37-45` | `python -m experiments.fovea_models` |
| Model D2 | Tablo V, VIII | Şifreli teşhis | Model D + yığın normalizasyonu + kare aktivasyon | Aynı | `foveahe/he_models.py:37-45` | Aynı |
| Model C | Tablo V | Şifreli teşhis | Katman başına tek evrişim (adım = çekirdek), yığın normalizasyonu, kare, tam bağlantılı | FoveaHE katmanları | `foveahe/he_cnn.py:38-60` | Aynı |
| Şifreli özet + D/D2 | Tablo VIII | Rakip yaklaşım | İstemcide dondurulmuş ResNet-18 (512 değer), sunucuda şifreli D/D2 | 224×224 görüntünün özeti | `experiments/fovea_baselines.py:50-72` | `python -m experiments.fovea_baselines` |
| U-Net | Metin (gerçekçilik testi) | Yardımcı | Küçük U-Net | Göğüs röntgeni → akciğer maskesi | `common/segmentation.py:17-41` | `python -m experiments.lung_segmenter` |

Şifreli çıkarım (CKKS, TenSEAL): Model D ve D2 için `foveahe/he_infer.py`, Model C için `foveahe/he_cnn.py`
(`CNNInference`). Şifreli ve şifresiz sonuç aynıdır: en büyük logit farkı 1.6e-4, tahmin uyumu %100 (makale, Bulgular).

## Ağırlıkların durumu (1 Ekim 2026)

| Model | Kayıtlı ağırlık | Yer | Not |
|---|---|---|---|
| Model D, D2, C | Var: 807 dosya, standardizasyon ve BN katlanmış, CKKS'e hazır `.npz` | `results/checkpoints/fovea_<veri>_<model>_<temsil>_s<tohum>_<kat>.npz` | Makaledeki Tablo V ve VIII modellerinin 5 tohumluk ağırlıklarının hepsi kayıtlı (şifreli özet dahil 480 dosya; 1 Eki'de denetlendi). İlk teslim zip'inde yoktu; Hugging Face'e yüklenecek. |
| Şifreli özet D/D2 | Var: 62 `.npz` | `results/checkpoints/ozet_<veri>_<model>_s<tohum>_<kat>.npz` | Aynı |
| U-Net | Var: 7.8 MB | `results/checkpoints/lung_unet.pt` | Aynı |
| Bağlam saldırganı (ResNet-18) | **Var (1 Eki 2026'dan beri)**: tam ve bağlam görüşü; beyin MR'da 5 kat modeli, COVID-QU-Ex'te 1 model; tohum 0 | `modeller/saldirgan_<veri>_<görüş>_tez_s0/<kat>.pt` | Makaledeki özgün koşular tahminleri kaydetti, ağırlıkları kaydetmedi. Aynı ayarla yeniden eğitildi (`boruhatti.egit`). Makaledeki sayılarla fark `results/boruhatti/degerlendirme_saldirgan.json` dosyasında. Ekran kartı hesapları bit bit tekrarlanmadığı için aynı tohumla bile küçük fark olur. |
| Meta veri saldırganı | Yok (gerek yok) | – | Saniyeler içinde yeniden eğitilir; tahminleri `results/preds/saldiri_A_*.npz` |
| Π_ROI | Eğitim yok | – | Rastgele ağırlık (tohum 0), yalnız maliyet ölçümü |

`results/` altındaki `checkpoints/` ve `preds/` klasörleri ile `modeller/` klasörü boyutları nedeniyle git deposunda
değildir; ağırlıklar ayrıca yayımlanır (README).

## Kayıtlı ağırlıklarla sınama (1 Eki 2026)

`python -m boruhatti.degerlendir --makale-modelleri --sifreli 9` sonuçları:
- **Tablo V ve VIII:** Makaledeki şifreli modellerin kayıtlı ağırlıkları tabloları yeniden üretiyor; 32 değerin 32'si
  makaledeki 5 tohum ortalamasıyla aynı. Tohum başına en büyük fark 3.4e-7.
- **Gerçek CKKS şifreli çıkarım:** 9'ar görüntüde en büyük logit farkı 7.7e-5, tahmin uyumu %100. Görüntü başına
  süre Model D 1.20 s, Model C 0.47 s, Model D2 1.27 s.

## Yeniden eğitim (1 Eki 2026)

Makalenin ana modelleri bu bilgisayarda (RTX 2070, 8 GB) `boruhatti.egit` ile yeniden eğitildi; 12 eğitim toplam yaklaşık
27 dk sürdü. Aynısı `python calistir.py --kademe 1` ile yapılır. Karşılaştırmalar `boruhatti.degerlendir`, kabul
kararları `boruhatti.kontrol` ile verilir (`results/boruhatti/kontrol.csv`: 15 denetimin 15'i TUTTU).

**Bağlam saldırganı, makale protokolü (tohum 0).** Ağırlıklar `modeller/saldirgan_<veri>_<görüş>_tez_s0/`.

| Veri | Görüş | AUC | Makaledeki aynı tohum | Fark | Makaledeki 5 tohum | Doğruluk | Makro F1 |
|---|---|---|---|---|---|---|---|
| Beyin MR | Π_ROI (bağlam) | 0.9743 | 0.9759 | −0.0016 | 0.9759 ± 0.0007 | 0.9122 | 0.9003 |
| Beyin MR | Tam görüntü | 0.9842 | 0.9843 | −0.0001 | 0.9843 ± 0.0012 | 0.9347 | 0.9241 |
| COVID-QU-Ex | Π_ROI (bağlam) | 0.9934 | 0.9933 | +0.0001 | 0.9932 ± 0.0001 | 0.9536 | 0.9528 |
| COVID-QU-Ex | Tam görüntü | 0.9967 | 0.9967 | −0.0001 | 0.9966 ± 0.0002 | 0.9698 | 0.9693 |

Saldırgan ekran kartında 16 bit karışık hassasiyetle eğitilir. cuDNN algoritma seçimi de değişebildiği için aynı
tohumla bile sonuç bit bit tekrarlanmaz. Kabul sınırı makaledeki 5 tohum ortalamasından ±0.005 AUC'dir
(`results/beklenen/toleranslar.json`).

**Şifreli modeller (tohum 0).** Ağırlıklar `modeller/foveahe_<model>_<veri>_<temsil>_<protokol>_s0/`. Yeniden
eğitilen ağırlıklar makaledeki kayıtlı ağırlıklarla **bit bit aynıdır** (6 koşunun 6'sı). Aynı test görüntülerinde AUC
farkı 0, tahmin uyumu %100.

| Veri | Model | Temsil | Bölme | Test AUC | Makaledeki aynı model | Makaledeki 5 tohum, aynı bölme |
|---|---|---|---|---|---|---|
| Beyin MR | D | F32_G16 | 5 katın hepsi | 0.9472 | 0.9472 | 0.9469–0.9491 |
| Beyin MR | D | F32_G16 | kat 1 | 0.9300 | 0.9300 | 0.9247–0.9300 |
| Beyin MR | C | F32_G16 | kat 1 | 0.9408 | 0.9408 | 0.9263–0.9408 |
| Beyin MR | D | U512 (tam görüntü) | kat 1 | 0.8914 | 0.8914 | 0.8660–0.8978 |
| COVID-QU-Ex | D2 | F32_G16 | resmi Test | 0.9517 | 0.9517 | 0.9517–0.9563 |
| COVID-QU-Ex | D2 | U256 (tam görüntü) | resmi Test | 0.9465 | 0.9465 | 0.9442–0.9470 |

Makaledeki Tablo V değeri beyin MR'da 5 katın birleşik sonucudur. İlk satırdaki 5 tohumun ortalaması (0.9479),
Tablo V'deki 0.948'dir. "Kat 1" satırları ise yalnız ilk katın 542 test görüntüsünü kapsar. Başka bir ekran kartında
kayan nokta farkı erken durdurmayı kaydırabilir. Kabul sınırı bu yüzden makaledeki 5 tohumun aynı bölmedeki AUC
aralığı ±0.005'tir.

## Eğitim oldu mu, ezberleme var mı (izleme koşuları)

İzleme koşularında (`--protokol izleme`) model her epoch'ta eğitimden ayrı bir doğrulama kümesinde ölçülür. Test
kümesi eğitimde ve model seçiminde kullanılmaz. Kayıtlar yerelde ve wandb'dedir: her adımda öğrenme oranı ve kayıp;
her epoch'ta eğitim/doğrulama kaybı, doğruluk, makro F1 ve AUC. Tablo: `results/tables/izleme_ozet.md`; ham eğriler:
`results/tables/izleme/`; şekiller: `results/figures/izleme_saldirgan.png`, `izleme_foveahe.png`.

| Veri | Model | Seçilen ayar | Epoch (durduğu) | Kullanılan ağırlık | Eğitim doğruluğu | Doğrulama doğruluğu | Fark | Doğrulama AUC |
|---|---|---|---|---|---|---|---|---|
| Beyin MR | Bağlam saldırganı | sabit (makale) | 12 | son epoch | 1.000 | 0.906 | 0.094 | 0.981 |
| COVID-QU-Ex | Bağlam saldırganı | sabit (makale) | 5 | son epoch | 0.948 | 0.929 | 0.019 | 0.987 |
| Beyin MR | Model D, F32_G16 | wd 10, lr 1e-3 | 24 | epoch 14 | 0.944 | 0.832 | 0.112 | 0.947 |
| Beyin MR | Model C, F32_G16 | wd 1, lr 1e-3 | 74 | epoch 64 | 0.994 | 0.844 | 0.150 | 0.946 |
| Beyin MR | Model D, U512 | wd 100, lr 1e-4 | 66 | epoch 56 | 0.949 | 0.730 | 0.219 | 0.887 |
| COVID-QU-Ex | Model D2, F32_G16 | wd 0.01, lr 1e-3 | 25 | epoch 15 | 0.867 | 0.822 | 0.045 | 0.942 |
| COVID-QU-Ex | Model D2, U256 | wd 1e-4, lr 1e-3 | 24 | epoch 14 | 0.873 | 0.825 | 0.048 | 0.942 |

Yorum:
- **Eğitim oldu.** Bütün modellerde eğitim kaybı düşüyor ve doğrulama başarımı yükseliyor. Örnekler: beyin MR
  saldırganında doğrulama AUC 0.961'den 0.981'e, COVID-QU-Ex saldırganında 0.980'den 0.987'ye çıkıyor. Model C'de
  doğrulama doğruluğu 0.67'den 0.84'e çıkıyor.
- **Beyin MR saldırganı eğitim kümesini ezberliyor**, ama bu doğrulama başarımını düşürmüyor. Eğitim doğruluğu
  10. epoch'tan itibaren %100'dür. Doğrulama kaybı ise en düşük değerinde kalır: 4. epoch'ta 0.2975, son epoch'ta 0.2996.
  Doğrulama doğruluğu son 5 epoch'ta 0.906–0.909 arasında sabittir. Makaledeki sabit 12 epoch ayarı bu yüzden
  doğrulama başarımını bozmuyor.
- **COVID-QU-Ex saldırganında** eğitim ve doğrulama arasında 2 puanlık fark var. Doğrulama kaybı en düşük değerine
  4. epoch'ta iner (0.173); 5. epoch'ta hafifçe 0.183'e çıkar.
- **Şifreli modeller** erken durdurmayla eğitilir. Doğrulama kaybı 10 epoch iyileşmeyince eğitim durur ve doğrulama
  kaybının en iyi olduğu epoch'un ağırlıkları kullanılır. Sonraki epoch'lardaki ezberleme kullanılan modele yansımaz.
- **Beyin MR'da eğitim–doğrulama farkı büyüktür.** FoveaHE modellerinde fark 0.11–0.15, COVID-QU-Ex'te 0.05'tir.
  Beyin MR'da 1 843 eğitim görüntüsü vardır, COVID-QU-Ex'te 21 715. En büyük fark tam görüntü modelindedir (U512):
  262 144 girdiye karşılık 1 843 eğitim görüntüsü. FoveaHE temsilinde (1 283 değer) fark daha küçüktür ve doğrulama
  AUC'si daha yüksektir: 0.947'ye karşı 0.887. Bu bulgu makaledeki Tablo V ile tutarlıdır (0.948'e karşı 0.881).

## Başarı ölçüleri

- AUC, doğruluk, dengeli doğruluk, kesinlik, duyarlılık, F1 ve karışıklık matrisleri:
  `results/tables/metrikler_ozet.md`, `metrikler_sinif.md`, `results/figures/karisiklik_*.png`.
- Makaledeki sayılarla birebir karşılaştırma: `results/tables/makale_eslesme.md`.
- Görüntü başına gerçek etiket ve tahminler: `results/tahminler/`.
