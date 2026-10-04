# Ön işleme: veri kümelerine ve model girdilerine uygulanan işlemler

**Kısa cevap:** Evet, ön işleme uygulanmıştır. Aşağıda her adım parametresi, kodun yeri ve gerekçesiyle verilmiştir.
Makaledeki karşılığı "Veri kümeleri" ve "Önişleme" paragraflarıdır. Kod yerleri `dosya:satır` biçimindedir.

Akış:

```
ham veri (figshare .mat, COVID-QU-Ex PNG, Kaggle PNG/JPG)
  └─ experiments/prepare_data.py   → manifestolar, beyin PNG + maske, resmi katlar, Kaggle–COVID-QU-Ex kopya listesi
  └─ experiments/lung_segmenter.py → Kaggle görüntülerine U-Net akciğer maskesi
  └─ experiments/brain_visible_norm.py → beyin MR ham yoğunluk önbelleği (görünür piksel normalizasyonu için)
  └─ foveahe/data.py (ilk kullanımda kendiliğinden) → FoveaHE temsil katmanları
eğitim sırasında: 224×224'e küçültme, sunucu görüşü, artırma (yalnız eğitimde), normalizasyon
```

## 1. Beyin tümörü MR (figshare 1512427; Cheng vd.)

Kaynak: 4 zip içinde 3064 `.mat` dosyası ve resmi 5 kat dosyası `cvind.mat`. Her dosyada görüntü, tümör maskesi,
etiket (1 meningiom, 2 gliom, 3 hipofiz) ve hasta kimliği bulunur. Kaggle'daki PNG kopyalarında hasta kimliği
olmadığından hasta bazlı bölme için özgün `.mat` dosyaları kullanılmıştır.

| Adım | İşlem | Parametre | Kod | Gerekçe |
|---|---|---|---|---|
| B1 | `.mat` okuma (h5py, gerekirse scipy) | – | `experiments/prepare_data.py:119-133` | Görüntü, maske, etiket, hasta kimliği |
| B2 | Kesit başına yoğunluk ölçekleme | 0.5 ve 99.5 yüzdelikleri arası → [0, 255] | `experiments/prepare_data.py:158-159` | Ham MR yoğunluklarının ölçeği kesitten kesite değişir |
| B3 | Tümör maskesinin ikili PNG'ye yazılması | 0 / 255 | `experiments/prepare_data.py:162` | ROI = tümör maskesi |
| B4 | Resmi 5 katın atanması | `cvind.mat` | `experiments/prepare_data.py:136-145, 167-168` | Hasta bazlı bölme; hiçbir hasta iki katta yok (doğrulandı) |
| B5 | Görünür piksel normalizasyonu (bağlam saldırısı, kök neden, savunma, bozuk bağlam) | Ham yoğunluk 224×224 bilineer önbellek; 0.5–99.5 yüzdelikleri **yalnız sunucunun gördüğü piksellerden** | `experiments/brain_visible_norm.py:39-76`, `attacks/context_cnn.py:69-78, 128-151` | Tüm piksellerle ölçekleme, gizli tümörün parlaklığını açık piksellerin ölçeğine taşıyabilir. Gerçek sunucu bunu göremez. Makalede Tablo III'ün beyin sayıları bu düzenle hesaplanmıştır. |

Görüntü boyutları: 3049 kesit 512×512, 15 kesit 256×256. CNN deneylerinde hepsi 224×224'e küçültülür (madde 4).
FoveaHE temsili ise dağıtıldığı çözünürlükten çıkarılır.

## 2. COVID-QU-Ex (Kaggle `anasmohammedtahir/covidqu`; Tahir vd.)

| Adım | İşlem | Parametre | Kod | Gerekçe |
|---|---|---|---|---|
| C1 | "Lung Segmentation Data" alt kümesinin manifestosu: görüntü, akciğer maskesi, etiket, resmi bölme | 33 920 görüntü; Train / Val / Test = 21 715 / 5417 / 6788 | `experiments/prepare_data.py:87-116` | Makaledeki resmi bölmeler |
| C2 | Görüntüler dağıtıldığı gibi kullanılır | 256×256 PNG (kümeyi hazırlayanların boyutu) | – | Ek yoğunluk işlemi yapılmaz |

ROI = veri kümesiyle gelen akciğer maskesi. Enfeksiyon maskesi ROI olarak kullanılmamıştır: normal görüntülerde boş
olduğu için etiketi sızdırır.

## 3. Kaggle göğüs röntgeni (Kaggle `prashant268/chest-xray-covid19-pneumonia`)

| Adım | İşlem | Parametre | Kod | Gerekçe |
|---|---|---|---|---|
| K1 | Manifesto | `train/` ve `test/` klasörleri, 6432 görüntü | `experiments/prepare_data.py:67-84` | Diskteki `train2/`, `test2/`, `val/` kullanılmaz |
| K2 | Görüntü imzası | Fark hash'i (9×8 gri küçültme, 64 bit) ve 32×32 normalize küçük resim | `experiments/prepare_data.py:39-53` | Kopya taraması için |
| K3 | COVID-QU-Ex ile kopya taraması | Hamming ≤ 10 aday çiftler; küçük resim korelasyonu ≥ 0.97 olanlar kopya | `experiments/prepare_data.py:180-204` | İki küme ortak kaynaklardan derlenmiştir; 6432 görüntünün 4665'inin kopyası çıktı. Dış testlerde (1767 görüntü) bunlar dışarıda bırakılır. |
| K4 | Akciğer maskesi üretimi | U-Net, COVID-QU-Ex Train ile 6 epoch; Val Dice 0.978; son işlemde en büyük iki bağlı bileşen (iki akciğer) tutulur, delikler doldurulur | `experiments/lung_segmenter.py`, `common/segmentation.py:55-63` | Kaggle kümesinde maske yok; bağlam görüşü için gerekir |

## 4. Model girdisine dönüştürme (eğitim ve test sırasında)

| Model | İşlem | Parametre | Kod |
|---|---|---|---|
| ResNet-18 (bağlam saldırganı, bilgi üst sınırı) | Küçültme | Görüntü 224×224 bilineer, maske en yakın komşu | `experiments/attack_context.py:33, 45-51` |
|  | Sunucu görüşü | Gizli bölge pikselleri 0; 3 kanal = [görünür, görünür, gizli bölge göstergesi] | `attacks/context_cnn.py:45-66` |
|  | Normalizasyon | ImageNet ortalama/std (gösterge kanalı 0.5/0.5) | `attacks/context_cnn.py:20-21` |
|  | Artırma (yalnız eğitimde) | Rastgele kaydırma ±%4, ölçek 0.93–1.07; görüntü ve maskeye aynı dönüşüm | `attacks/context_cnn.py:81-95` |
|  | Değerlendirme | 32 bit hassasiyet, karışık sıra (sahte AUC'ye karşı) | `attacks/context_cnn.py:202-213` |
| Meta veri saldırganı (HistGradientBoosting) | Maske 256×256 (en yakın komşu); 4 grupta 23 öznitelik: girdi boyutu, konum, büyüklük, şekil | – | `experiments/attack_metadata.py:26, 44-53`, `attacks/metadata.py:11-53` |
| FoveaHE temsili (Model D, D2, C) | Odak penceresi: ROI kutusunun merkezi; kenar = büyük kenar × 1.25, en az görüntü kenarının %10'u | – | `foveahe/representation.py:24-26, 130-152` |
|  | Alan ortalamalı örnekleme (roi_align) ve 8 bit nicemleme | Odak F×F, genel bakış G×G (ör. F32_G16) | `foveahe/representation.py:155-167`, `foveahe/data.py:89-119` |
|  | Şifrelenen vektör | [odak (F²), genel bakış (G²), geometri (3)]; Model C geometri kullanmaz | `foveahe/representation.py:26`, `experiments/fovea_models.py:92-129` |
|  | Standardizasyon | Eğitim kümesinin ortalama/std'si (std ≥ 0.05); aktarımda ilk katmana katlanır | `foveahe/he_models.py:18-29, 48-62` |
| Şifreli özet + Model D/D2 | 224×224 gri → 3 kanal → ImageNet normalizasyonu → dondurulmuş ResNet-18 → 512 değer | – | `experiments/fovea_baselines.py:40-72` |

## 5. Yapılmayanlar

- **Sınıf dengeleme yok:** yeniden örnekleme ya da sınıf ağırlığı kullanılmadı. Sınıf dağılımı
  `results/tables/veri_dagilimi_*.md` dosyalarında. Dengesizliğin etkisi dengeli doğruluk ve makro F1 ile
  `results/tables/metrikler_ozet.md` dosyasında raporlanır.
- **Ana deneylerde histogram eşitleme yok.** Yalnızca kök neden ablasyonunda sızıntıya etkisini ölçmek için denendi.
- **Elle görüntü ayıklama yok.** Tek ayıklama, dış testlerdeki Kaggle–COVID-QU-Ex kopyalarıdır (madde 3).
- **Test verisiyle ayar yapılmadı.** Şifreli modellerde ayar doğrulama kümesiyle yapıldı. ResNet-18'de epoch sayısı
  sabittir: beyin 12, COVID-QU-Ex 5.

## 6. Çalıştırma

İndirme ve bütün ön işleme üç komuttur (`calistir.py` bunları sırayla çalıştırır):

```
python -m boruhatti.veri_indir        # Kaggle (hesapsız) ve figshare'den indirir, çalışmadaki veriyle aynı olduğunu doğrular
python -m boruhatti.modelleri_indir   # makaledeki ağırlıklar (U-Net dahil) Hugging Face'ten; SHA-256 ile doğrulanır
python -m boruhatti.on_isle           # bu belgedeki adımların hepsi; çıktısı olan adımı atlar
```

`modelleri_indir` ön işlemeden önce çalışırsa Kaggle akciğer maskeleri makaledeki U-Net'le üretilir; yoksa U-Net
yeniden eğitilir.

`boruhatti.on_isle` sırasıyla şu betikleri çağırır: `experiments.prepare_data`, `experiments.lung_segmenter`
(kayıtlı U-Net ağırlığıyla yalnız maske, ağırlık yoksa eğitim), `experiments.brain_visible_norm`, 224 px önbellekler,
FoveaHE katmanları ve şifreli özet için ResNet-18 özetleri.
