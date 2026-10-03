# Sağlık Verilerinde Homomorfik Şifreleme ve Zafiyet Analizi

Bu depo, aynı adlı bildirinin (B. Tutumlu, A. Uğur, V. Tataroğlu; Pamukkale Üniversitesi, 2026) kodunu, yeniden üretim
adımlarını ve model ağırlıklarının bağlantısını içerir. İki soru vardır:

1. **Sızıntı.** Görüntünün yalnız ilgi bölgesini (ROI) şifreleyen yöntem (Π_ROI), sunucuya açık kalan bağlamdan teşhisi
   ele veriyor mu? Bunu ölçen modeller saldırganlardır.
2. **Çözüm.** Önerilen FoveaHE (odaklı tam şifreleme), sızıntı olmadan hızlı ve doğru şifreli teşhis yapabiliyor mu?

Veri kümeleri: beyin tümörü MR (figshare; meningiom, gliom, hipofiz tümörü) ve göğüs röntgeni (COVID-QU-Ex; COVID-19,
COVID dışı pnömoni, normal). Şifreleme: CKKS (TenSEAL).

## Hızlı başlangıç (Windows, NVIDIA ekran kartı)

```bat
git clone https://github.com/Yabgun/roi-leakage C:\rl
cd C:\rl
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-kilit.txt --extra-index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python calistir.py --kademe 0
```

- Projeyi **kısa bir klasöre** kurun (ör. `C:\rl`). COVID-QU-Ex dosya adları uzundur; Windows'un 260 karakterlik yol
  sınırı aşılırsa indirme yarıda kalır. `boruhatti.ortam_kontrol` bunu baştan denetler.
- **Hesap gerekmez.** Veriler Kaggle ve figshare'den, model ağırlıkları Hugging Face'ten hesapsız iner.
- `requirements-kilit.txt`, sonuçların üretildiği ortamın birebir aynısını kurar: Python 3.13.0, torch 2.14.0 (CUDA
  12.6) ve bütün alt bağımlılıklar sabit. Daha esnek kurulum için `requirements.txt` (ana paketler sabit).
- Sonuçlar RTX 2070 (8 GB) ekran kartıyla üretildi.

## Kademeler

| Kademe | Komut | Ne yapar | Süre (RTX 2070) |
|---|---|---|---|
| 0 | `calistir.py --kademe 0` | Veriyi indirip doğrular, makaledeki modelleri indirir, ön işler, modelleri test eder, 27 görüntüde gerçek CKKS şifreli çıkarım yapar, makale tablolarını kayıtlı tahminlerden yeniden hesaplar, sonuçları denetler. Eğitim yoktur. | ~50 dk: veri indirme 20, model indirme 8, ön işleme 13, kopya taraması 4, değerlendirme 3 (indirme süresi bağlantıya bağlıdır) |
| 1 | `calistir.py --kademe 1` | Kademe 0 + makalenin ana modellerini bu bilgisayarda yeniden eğitir (12 eğitim, eğitim eğrileriyle) + ezber denetimi (etiket karıştırma). Yeni modeller makaledeki modellerle karşılaştırılır. | Kademe 0 + ~45 dk: 12 eğitim ~22 dk (wandb açıkken ~27), ezber denetimi ~19 dk |
| 2 | aşağıdaki komut listesi | Makaledeki bütün deneyler | ~1–2 gün |

Disk (temiz kurulumda ölçüldü): ham veri 5.2 GB, ön işleme çıktıları 6.6 GB, model ağırlıkları ve kayıtlı tahminler
3.1 GB, Python ortamı ~4.7 GB; toplam ~20 GB. En az 25 GB boş yer önerilir.

Kademe 1, indirilen saldırgan ağırlıklarını (`modeller/saldirgan_*_tez_s0/`) bu bilgisayarda eğitilenlerle değiştirir.
İndirilen sürüme dönmek için: `python -m boruhatti.modelleri_indir`.

### Sonuç nasıl okunur

Her kademenin sonunda `boruhatti.kontrol` çalışır ve `results/boruhatti/kontrol.csv` dosyasını yazar. Her denetim
için sonuç TUTTU, TUTMADI ya da ATLANDI'dır. ATLANDI, o denetimin çıktısının bu koşuda üretilmediğini gösterir (ör.
Kademe 0'da yeniden eğitim denetimleri); git'ten gelen başvuru sonuçları denetimi geçmiş sayılmaz.

| Denetim | Beklenen | Kademe |
|---|---|---|
| Makaledeki şifreli modeller (Tablo V, VIII) | 32 değerin 32'si makaledekiyle aynı | 0, 1 |
| Gerçek CKKS şifreli çıkarım (3 model × 9 görüntü) | Logit farkı ≤ 1e-3, tahmin uyumu %100 | 0, 1 |
| Saldırgan modelleri (4) | AUC, makaledeki 5 tohum ortalamasından en çok ±0.005 farklı | 0, 1 |
| Kayıtlı tahminlerden makale tabloları | 111 değerin 111'i | 0, 1 |
| FoveaHE yeniden eğitimi (6) | AUC, makaledeki 5 tohumun aynı bölmedeki aralığında (±0.005) | 1 |
| Ezber denetimi (4) | Gerçek etiketli sonuç, etiketleri karıştırılmış bütün koşuların üstünde | 1 |

Kabul sınırlarının gerekçeleri: `results/beklenen/toleranslar.json`. Bu çalışmanın kendi sonuçları depoda
`results/boruhatti/` ve `results/tables/` altındadır; kendi koşunuzdan sonra `git diff --stat results/` neyin
değiştiğini gösterir.

## Modeller

Makalenin iki ana modeli: sızıntıyı gösteren **bağlam saldırganı** (ImageNet ön eğitimli ResNet-18, Π_ROI görüşü;
Tablo III) ve önerilen yöntemin modeli **FoveaHE F32_G16 + Model D** (beyin MR) ve **+ Model D2** (göğüs röntgeni)
(Tablo V, VIII). Bütün modeller, girdileri, etiketleri ve ağırlık dosyaları: `docs/MODELLER.md`, `docs/ETIKETLER.md`.

Ağırlıklar Hugging Face'tedir: https://huggingface.co/Btutumlu/roi-leakage. `python -m boruhatti.modelleri_indir`
bunları doğru klasörlere indirir (Kademe 0 ve 1 bunu kendisi yapar).

## Eğitim kayıtları ve ezber denetimi

- Her eğitimde her adımın öğrenme oranı ve kaybı, her epoch'un eğitim ve doğrulama kaybı, doğruluğu, makro F1'i ve
  AUC'si yerelde `modeller/<ad>/<bölüm>/egitim_egrisi_*.csv|png` dosyalarına yazılır.
- `wandb login` yapılmışsa aynı kayıtlar Weights & Biases'a da gider. İstenmezse `calistir.py --wandb-kapali`.
- Özet ve yorum: `docs/MODELLER.md` ("Eğitim oldu mu, ezberleme var mı" ve "Sonuçlar ezbere mi dayanıyor?"),
  `results/tables/izleme_ozet.md`, `results/tables/ezber_denetimi.md`, `results/figures/izleme_*.png`.

## Yeni görüntülerle tahmin

```bat
.venv\Scripts\python -m boruhatti.tahmin --veri covidqu --goruntu grafi.png --otomatik-maske
.venv\Scripts\python -m boruhatti.tahmin --veri beyin --goruntu kesit.png --maske tumor.png --sifreli
.venv\Scripts\python -m boruhatti.tahmin --veri covidqu --klasor yeni_goruntuler\ --otomatik-maske
```

Her görüntü için üç modelin tahmini ve olasılıkları yazılır: saldırgan (tam görüntü), saldırgan (Π_ROI görüşü) ve
FoveaHE. `--sifreli` FoveaHE tahminini gerçek CKKS şifreli çalıştırır. Girdi beklentileri ve sınırlar:
`docs/GIRDI_SOZLESMESI.md`. Yeni bir veri kümesinin eğitim verileriyle kopyası olup olmadığı:
`python -m boruhatti.kopya_kontrol --klasor yeni_goruntuler\`.

## Belgeler

| Belge | İçerik |
|---|---|
| `docs/ON_ISLEME.md` | Veri indirmeden model girdisine kadar her ön işlem adımı, parametresi ve gerekçesi |
| `docs/MODELLER.md` | Model haritası, ağırlıklar, yeniden eğitim sonuçları, eğitim eğrileri ve ezber denetimi |
| `docs/ETIKETLER.md` | Sınıflar, etiket numaraları ve tahmin dosyalarının sütunları |
| `docs/GIRDI_SOZLESMESI.md` | Yeni görüntülerle test için girdi beklentileri |
| `docs/calisma_plani_2026-09-15.md` | Çözüm deneylerinden önce dondurulan başarı ölçütleri (ön kayıt) |

## Kademe 2: makaledeki bütün deneyler

Kademe 0'dan sonra (veri indirilmiş ve ön işlenmiş olarak) aşağıdaki komutlar sırayla çalıştırılır:
`.venv\Scripts\python -m experiments.<ad>`.

| Deney | Komut | Makalede | Çıktı |
|---|---|---|---|
| Π_ROI'nin yeniden üretimi ve maliyeti | `experiments.piroi_benchmark` | Bulgular, maliyet | `results/tables/piroi_*.csv` |
| Saldırı A: meta veri | `experiments.attack_metadata` | Tablo II | `results/tables/saldiri_A_*.csv` |
| Kesit yönü karıştırıcı mı (Saldırı A) | `experiments.brain_orientation` | Saldırı A metni | `results/tables/saldiri_A_beyin_yon_kontrolu_*.json` |
| Saldırı B: bağlam saldırganı | `experiments.attack_context` | Tablo III | `results/tables/saldiri_B_baglam*.csv` |
| Akciğer U-Net'i | `experiments.lung_segmenter` | Gerçekçilik testi | `results/tables/akciger_segmentasyon.json` |
| Gerçekçilik testi | `experiments.attack_context_transfer` | Gerçekçilik testi | `results/tables/saldiri_B_transfer.csv` |
| Kök neden (Grad-CAM, ön işleme) | `experiments.root_cause` | Kök neden | `results/tables/kok_neden_*.csv` |
| Savunma | `experiments.defense_expansion` | Tablo IV | `results/tables/savunma*.csv` |
| FoveaHE bilgi düzeyi | `experiments.fovea_info` | Çözüm | `results/tables/cozum_bilgi.csv` |
| Şifreli modeller | `experiments.fovea_models` | Tablo V | `results/tables/cozum_modeller.csv` |
| Şifreli çıkarım ve maliyet | `experiments.fovea_cost` | Tablo VI | `results/tables/cozum_maliyet.csv` |
| Sızıntı denetimi | `experiments.fovea_leakage` | Tablo VII | `results/tables/cozum_sizinti.csv` |
| Rakip: şifreli özet | `experiments.fovea_baselines` | Tablo VIII | `results/tables/cozum_rakipler.csv` |
| Rakip: bozuk bağlam | `experiments.attack_noisy_context` | Tablo VIII | `results/tables/saldiri_B_gurultu.csv` |
| Dış veride genelleme | `experiments.fovea_transfer` | Çözüm | `results/tables/cozum_genelleme.csv` |
| Bütçe taraması | `experiments.fovea_budget` | Çözüm | `results/tables/cozum_butce.csv` |
| Birleşik özet | `experiments.fovea_pareto` | Çözüm | `results/tables/cozum_ozet.csv` |

Not: RTX 2070 ve cuDNN 9.10'da `channels_last` bellek düzeni saldırgan eğitimini ~7.5 kat yavaşlattığı için
kullanılmaz (adım başına 529 ms yerine 71 ms).

## Veri kümeleri

Veriler bu depoda ve Hugging Face'te yoktur; `boruhatti.veri_indir` özgün kaynaklarından sabit sürümle indirir ve
parmak iziyle doğrular.

| Veri | Kaynak | Sürüm | Lisans | Atıf |
|---|---|---|---|---|
| Beyin tümörü MR | figshare 1512427 | 8 | CC BY 4.0 | Cheng vd., PLOS ONE 10(10) e0140381 (2015) |
| COVID-QU-Ex | Kaggle `anasmohammedtahir/covidqu` | 7 | CC BY-SA 4.0 | Tahir vd., Comput. Biol. Med. 139 (2021) 105002 |
| Kaggle göğüs röntgeni | Kaggle `prashant268/chest-xray-covid19-pneumonia` | 2 | Belirtilmemiş | Kaggle sayfası |

## Klasörler

| Klasör | İçerik |
|---|---|
| `boruhatti/` | Uçtan uca boru hattı: ortam denetimi, indirme, ön işleme, eğitim, değerlendirme, tahmin, denetim |
| `experiments/` | Makaledeki deneylerin betikleri (Kademe 2) |
| `attacks/`, `defenses/` | Saldırı ve savunma kodu |
| `he/` | Π_ROI'nin TenSEAL ile yeniden üretimi |
| `foveahe/` | FoveaHE: odaklı temsil, şifreli çalışabilen modeller, şifreli çıkarım |
| `analysis/` | Ölçüler, tablolar ve şekiller |
| `common/` | Ortak yardımcılar |
| `results/` | Tablolar, şekiller, görüntü başına tahminler ve beklenen değerler |
| `docs/` | Belgeler |
| `pilot/` | Eylül 2026 pilot betikleri |

## Sorun giderme

| Belirti | Çözüm |
|---|---|
| İndirmede `WinError 206` ya da "yol çok uzun" | Projeyi kısa bir klasöre taşıyın (ör. `C:\rl`). |
| `torch.cuda.is_available()` False | torch'u yukarıdaki CUDA 12.6 komutuyla yeniden kurun; NVIDIA sürücüsü güncel olmalı. |
| wandb giriş soruyor ya da istenmiyor | `calistir.py --wandb-kapali` ya da `set WANDB_MODE=disabled` |
| İndirme yarıda kaldı | Aynı komutu yeniden çalıştırın; tamamlanmış dosyalar yeniden inmez. |
