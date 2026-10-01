# Görüntü başına tahminler (tohum 0)

Her dosyada bir satır bir test görüntüsüdür. Tahmin edilen etiket, olasılığı en yüksek sınıftır. Dosyalar `python -m analysis.tahminler` ile kayıtlı tahminlerden üretilir; yeniden eğitim yapılmaz.

| Dosya | Makalede | Model | Girdi | Görüntü | Doğruluk |
|---|---|---|---|---|---|
| `beyin_meta_veri_saldirgani_s0.csv` | Tablo II | Meta veri saldırganı (HistGB) | ROI konum/büyüklük/şekil + girdi boyutu | 3064 | 0.690 |
| `beyin_baglam_saldirgani_tam_s0.csv` | Tablo III | Bağlam saldırganı (ResNet-18) | Tam görüntü (şifreleme yok) | 3064 | 0.936 |
| `beyin_baglam_saldirgani_piroi_s0.csv` | Tablo III | Bağlam saldırganı (ResNet-18) | Π_ROI görüşü: tümör gizli | 3064 | 0.914 |
| `beyin_model_D_tam_goruntu_s0.csv` | Tablo V | Model D (= Π_ROI'nin modeli) | Tam görüntü 512×512 (şifreli) | 3064 | 0.755 |
| `beyin_model_D_foveahe_F32_G16_s0.csv` | Tablo V, VIII | Model D | FoveaHE F32_G16 (şifreli) | 3064 | 0.840 |
| `beyin_model_C_foveahe_F32_G16_s0.csv` | Tablo V | Model C | FoveaHE F32_G16 (şifreli) | 3064 | 0.850 |
| `beyin_sifreli_ozet_model_D_s0.csv` | Tablo VIII | Şifreli özet + Model D | ResNet-18 özeti, 512 değer (şifreli) | 3064 | 0.864 |
| `covidqu_meta_veri_saldirgani_s0.csv` | Tablo II | Meta veri saldırganı (HistGB) | Akciğer maskesinin konum/büyüklük/şekli | 6788 | 0.648 |
| `covidqu_baglam_saldirgani_tam_s0.csv` | Tablo III | Bağlam saldırganı (ResNet-18) | Tam görüntü (şifreleme yok) | 6788 | 0.971 |
| `covidqu_baglam_saldirgani_piroi_s0.csv` | Tablo III | Bağlam saldırganı (ResNet-18) | Π_ROI görüşü: akciğerler gizli | 6788 | 0.953 |
| `covidqu_model_D2_tam_goruntu_s0.csv` | Tablo V | Model D2 | Tam görüntü 256×256 (şifreli) | 6788 | 0.837 |
| `covidqu_model_D2_foveahe_F32_G16_s0.csv` | Tablo V, VIII | Model D2 | FoveaHE F32_G16 (şifreli) | 6788 | 0.841 |
| `covidqu_model_C_foveahe_F32_G16_s0.csv` | Tablo V | Model C | FoveaHE F32_G16 (şifreli) | 6788 | 0.788 |
| `covidqu_sifreli_ozet_model_D2_s0.csv` | Tablo VIII | Şifreli özet + Model D2 | ResNet-18 özeti, 512 değer (şifreli) | 6788 | 0.908 |

## Sütunlar

- `satir`: manifestodaki satır numarası (kodun içindeki sıra).
- Beyin MR: `goruntu_no` (figshare dosya numarası), `dosya`, `hasta` (hasta kimliği), `kat` (resmi 5 kattan hangisinde test edildiği).
- COVID-QU-Ex: `dosya` (veri kümesindeki yol), `resmi_bolme` (Test).
- `gercek_etiket_no`, `gercek_etiket`: veri kümesinin etiketi (numaralar `docs/ETIKETLER.md`'de).
- `tahmin_no`, `tahmin`: modelin tahmini; `dogru_mu`: tahmin gerçek etiketle aynı mı.
- `olasilik_<sınıf>`: modelin o sınıfa verdiği olasılık; satır toplamı 1.

Beyin MR'da her kesit, hastası test katındayken bir kez tahmin edilir; 5 katın tahminleri birleşince bütün kesitler tabloda yer alır. COVID-QU-Ex'te yalnızca resmi Test bölmesi tahmin edilir.
