# roi-leakage

ROI-seçici homomorfik şifrelemeye karşı sızıntı istismarı saldırıları. Lisans tezi deney kodu; plan için `../TEZ_PLANI_v3.md`.

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
| `data/processed/` | Hazırlanmış manifestolar, bölmeler, önbellek |
| `he/` | Π_ROI'nin TenSEAL ile yeniden üretimi |
| `attacks/` | Meta veri, bağlam ve inpainting saldırıları |
| `defenses/` | Sabit ROI, sızıntı-güdümlü genişletme |
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

Not: RTX 2070 + cuDNN 9.10'da `channels_last` bellek düzeni eğitimi ~7.5 kat yavaşlattığı için kullanılmıyor (ölçüm: 529 ms/adım yerine 71 ms/adım).
