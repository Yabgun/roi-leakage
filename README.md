# roi-leakage

ROI-seçici homomorfik şifrelemeye karşı sızıntı istismarı saldırıları ve sızıntısız çözüm (FoveaHE, odaklı tam şifreleme). Lisans tezi deney kodu; plan için `../TEZ_PLANI_v3.md` (saldırı) ve `../COZUM_PLANI.md` (çözüm).

## Kurulum (Windows, Python 3.13)

```bash
python -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install -r requirements.txt
```

Tüm deneyler proje kökünden çalıştırılır: `.venv\Scripts\python -m experiments.<ad>`.

## Klasörler

| Klasör | İçerik |
|---|---|
| `data/raw/` | İndirilen ham veriler (git dışı): `kaggle_cxr`, `covid_qu_ex`, `brain_tumor_figshare`, `stroke` |
| `data/processed/` | Hazırlanmış manifestolar, bölmeler, önbellek (FoveaHE katmanları: `fovea/`) |
| `he/` | Π_ROI'nin TenSEAL ile yeniden üretimi |
| `attacks/` | Meta veri, bağlam ve inpainting saldırıları |
| `defenses/` | Sabit ROI, sızıntı-güdümlü genişletme |
| `foveahe/` | Çözüm: odaklı temsil (`representation.py`), katman önbelleği (`data.py`), şifreli çalışabilen modeller (`he_models.py`), şifreli çıkarım (`he_infer.py`) |
| `analysis/` | Isı haritaları, önişleme ablasyonu, şekiller |
| `experiments/` | Tek komutla çalışan deney betikleri |
| `results/` | Tablolar (`tables/`), şekiller (`figures/`), günlükler (`logs/`) |
| `pilot/` | Eylül 2026 pilot betikleri |

## Adımlar

| # | Adım | Komut | Çıktı |
|---|---|---|---|
| 1 | Klasör düzeni ve proje iskeleti | (tek seferlik) | `../README.md`, bu depo |
| 2 | Ortam | yukarıdaki kurulum | `requirements.txt` |
| 3a | Veri hazırlığı (Kaggle CXR, COVID-QU-Ex, beyin MR) | `python -m experiments.prepare_data` | `data/processed/*_manifest.csv`, `data/processed/brain/meta.csv`, `data/processed/veri_ozeti.json` |
| 3b | Veri setleri arası tekrar eden görüntüler | `python -m experiments.prepare_data --only dups` | `data/processed/cxr_capraz_tekrarlar.csv` |
| 4 | Π_ROI yeniden üretimi (doğruluk + maliyet) | `python -m experiments.piroi_benchmark` | `results/tables/piroi_*.csv`, `results/figures/piroi_hiz_kazanci.png` |
| 5 | Saldırı A: meta veri | `python -m experiments.attack_metadata` | `results/tables/saldiri_A_*.csv` |
| 6 | Saldırı B: bağlam (CNN) | `python -m experiments.attack_context` | `results/tables/saldiri_B_baglam.csv` |
| 6b | Akciğer segmentasyonu (Kaggle maskeleri) | `python -m experiments.lung_segmenter` | `results/tables/akciger_segmentasyon.json` |
| 6c | Gerçekçilik testi (başka kaynaktan saldırgan) | `python -m experiments.attack_context_transfer` | `results/tables/saldiri_B_transfer.csv` |
| 7 | Kök neden (Grad-CAM, önişleme) | `python -m experiments.root_cause` | `results/tables/kok_neden_*.csv`, `results/figures/kok_neden_gradcam_*.png` |
| 8 | Savunma ve gerçek bedel | `python -m experiments.defense_expansion` | `results/tables/savunma*.csv`, `results/figures/savunma_*.png` |
| Ç0 | FoveaHE temsil ve Model C öz sınamaları | `python -m foveahe.representation`, `python -m foveahe.he_cnn` | (konsol) |
| Ç1 | Çözüm Adım 1: odaklı temsil teşhis bilgisini koruyor mu (ResNet-18 üst sınır; tam, eş örnekli ve ROI penceresi referansları) | `python -m experiments.fovea_info` (kuyruk: `experiments\run_fovea_step1.ps1`) | `results/tables/cozum_bilgi.csv|md`, `results/figures/cozum_bilgi_egrisi.png`, `results/figures/cozum_temsil_ornek_*.png` |
| Ç2 | Çözüm Adım 2: şifreli çalışabilen modeller (D: Π_ROI sınıfı, D2: kare aktivasyon), tam görüntüyle adil karşılaştırma | `python -m experiments.fovea_models` (kuyruk: `experiments\run_fovea_step2_cpu.ps1`) | `results/tables/cozum_modeller.csv|md` |
| Ç3 | Çözüm Adım 3: gerçekten şifreli çıkarım, doğruluk eşleşmesi, maliyet dökümü (makine boşken) | `python -m experiments.fovea_cost` | `results/tables/cozum_dogruluk_eslesme.csv|md`, `results/tables/cozum_maliyet.csv|md` |
| Ç4 | Çözüm Adım 4: sızıntı denetimi (paket meta verisi, yan kanal; Π_ROI pozitif kontrol) | `python -m experiments.fovea_leakage` | `results/tables/cozum_sizinti.csv|md` |
| Ç5a | Çözüm Adım 5, rakip: şifreli özet (HETAL tarzı; ImageNet ResNet-18 özeti + şifreli D/D2, istemci CPU süresi) | `python -m experiments.fovea_baselines` | `results/tables/cozum_rakipler.csv|md` |
| Ç5b | Çözüm Adım 5, rakip: gürültülü/bulanık bağlam saldırısı (Bi-CryptoNets tarzı) ve aynı düzeyde fayda | `python -m experiments.attack_noisy_context` | `results/tables/saldiri_B_gurultu.csv|md` |
| Ç6 | Çözüm Adım 6: birleşik özet — hız × doğruluk × sızıntı şekli ve gizlilik şartlı hız tablosu (tüm adımların tablolarını birleştirir) | `python -m experiments.fovea_pareto` | `results/tables/cozum_ozet.csv|md`, `results/figures/cozum_pareto_{brain,covidqu}.png` |
| ÇG | Özgünlük bonusu: harici derlemede genelleme (COVID-QU-Ex'te eğitilmiş şifreli modeller yeniden eğitilmeden Kaggle CXR'de, kopyalar çıkarılmış 1.767 görüntü; tam görüntü, eş örnekli, FoveaHE, şifreli özet) | `python -m experiments.fovea_transfer` | `results/tables/cozum_genelleme.csv|md`, `results/figures/cozum_genelleme.png` |
| K9 | Kalan tüm çözüm deneyleri tek kuyrukta; pencere başlığı ve çubukta yüzde, kalan süre, tahmini bitiş; kapanırsa aynı komutla kaldığı yerden (`-Deneme`: sahte işlerle sınama) | `powershell -NoProfile -ExecutionPolicy Bypass -File experiments\run_fovea_final.ps1` | `results/logs/kuyruk.log`, `results/logs/kuyruk9_*.out.log|err.log` |
| ÇB | Özgünlük bonusu: bütçe duyarlı odak (sabit şifreli değer bütçesinde odak ↔ genel bakış payı; Model C ve D) | `python -m experiments.fovea_budget` | `results/tables/cozum_butce.csv|md`, `results/tables/cozum_butce_en_iyi.md`, `results/figures/cozum_butce.png` |

Not: RTX 2070 + cuDNN 9.10'da `channels_last` bellek düzeni eğitimi ~7.5 kat yavaşlattığı için kullanılmıyor (ölçüm: 529 ms/adım yerine 71 ms/adım).
