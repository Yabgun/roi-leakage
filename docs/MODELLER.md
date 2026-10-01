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
| Bağlam saldırganı (ResNet-18) | **Yok** | – | Özgün koşular tahminleri kaydetti, ağırlıkları kaydetmedi. Aynı ayarla tohum 0 için yeniden eğitilip kaydedilecek; makaledeki sayılarla farkı ölçülüp yazılacak. |
| Meta veri saldırganı | Yok (gerek yok) | – | Saniyeler içinde yeniden eğitilir; tahminleri `results/preds/saldiri_A_*.npz` |
| Π_ROI | Eğitim yok | – | Rastgele ağırlık (tohum 0), yalnız maliyet ölçümü |

`results/` altındaki `checkpoints/` ve `preds/` klasörleri boyutları nedeniyle git deposunda değildir.

## Başarı ölçüleri

- AUC, doğruluk, dengeli doğruluk, kesinlik, duyarlılık, F1 ve karışıklık matrisleri:
  `results/tables/metrikler_ozet.md`, `metrikler_sinif.md`, `results/figures/karisiklik_*.png`.
- Makaledeki sayılarla birebir karşılaştırma: `results/tables/makale_eslesme.md`.
- Görüntü başına gerçek etiket ve tahminler: `results/tahminler/`.
