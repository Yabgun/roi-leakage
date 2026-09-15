# Deney Özeti (otomatik)
## Veri
- Beyin MR: 3064 kesit, 233 hasta, sınıflar {'glioma': 1426, 'pituitary': 930, 'meningioma': 708}, birden fazla katta görünen hasta: 0
- COVID-QU-Ex: {'Test/COVID-19': 2395, 'Test/Non-COVID': 2253, 'Test/Normal': 2140, 'Train/COVID-19': 7658, 'Train/Non-COVID': 7208, 'Train/Normal': 6849, 'Val/COVID-19': 1903, 'Val/Non-COVID': 1802, 'Val/Normal': 1712}
- Kaggle CXR: {'test/COVID19': 116, 'test/NORMAL': 317, 'test/PNEUMONIA': 855, 'train/COVID19': 460, 'train/NORMAL': 1266, 'train/PNEUMONIA': 3418}
- COVID-QU-Ex'te kopyası olan Kaggle görüntüsü: 4665/6432

## Π_ROI yeniden üretimi
| size | ornek | rho | maks_mutlak_hata | toplam_s |
|---|---|---|---|---|
| 64 | beyin MR, tümör maskesi | 0.0176 | 1.43e-06 | 1.29 |
| 64 | akciğer grafisi, akciğer maskesi | 0.2 | 3.02e-06 | 1.69 |
| 128 | beyin MR, tümör maskesi | 0.0175 | 1.27e-06 | 1.68 |
| 128 | akciğer grafisi, akciğer maskesi | 0.198 | 4.12e-06 | 1.95 |

| size | rho | ciphertext_sayisi | sifreli_slot | sifreleme_s | sunucu_s | toplam_s | hiz_kazanci_toplam | yukleme_MB |
|---|---|---|---|---|---|---|---|---|
| 64 | 0.009 | 1 | 64 | 0.012 | 1.205 | 1.237 | 1.599 | 0.855 |
| 64 | 0.048 | 1 | 256 | 0.011 | 1.483 | 1.512 | 1.308 | 0.855 |
| 64 | 0.098 | 1 | 512 | 0.011 | 1.556 | 1.587 | 1.246 | 0.856 |
| 64 | 0.250 | 1 | 1024 | 0.011 | 1.670 | 1.701 | 1.163 | 0.856 |
| 64 | 0.494 | 1 | 2048 | 0.011 | 1.648 | 1.676 | 1.180 | 0.856 |
| 64 | 0.739 | 1 | 4096 | 0.012 | 1.762 | 1.793 | 1.103 | 0.856 |
| 64 | 0.908 | 1 | 4096 | 0.012 | 1.944 | 1.977 | 1.001 | 0.856 |
| 64 | 1.000 | 1 | 4096 | 0.012 | 1.947 | 1.978 | 1.000 | 0.856 |
| 128 | 0.010 | 1 | 256 | 0.012 | 1.457 | 1.490 | 2.476 | 0.855 |
| 128 | 0.051 | 1 | 1024 | 0.013 | 1.749 | 1.784 | 2.067 | 0.855 |
| 128 | 0.098 | 1 | 2048 | 0.012 | 1.818 | 1.852 | 1.992 | 0.856 |
| 128 | 0.250 | 1 | 4096 | 0.014 | 1.960 | 1.997 | 1.847 | 0.855 |
| 128 | 0.505 | 2 | 8320 | 0.022 | 2.666 | 2.706 | 1.363 | 1.711 |
| 128 | 0.752 | 2 | 16384 | 0.026 | 3.766 | 3.813 | 0.967 | 1.712 |
| 128 | 0.894 | 2 | 16384 | 0.029 | 3.738 | 3.785 | 0.974 | 1.711 |
| 128 | 1.000 | 2 | 16384 | 0.025 | 3.644 | 3.689 | 1.000 | 1.712 |
| 256 | 0.010 | 1 | 1024 | 0.011 | 1.602 | 1.631 | 8.942 | 0.856 |
| 256 | 0.050 | 1 | 4096 | 0.012 | 2.008 | 2.040 | 7.151 | 0.855 |
| 256 | 0.100 | 1 | 8192 | 0.014 | 2.106 | 2.137 | 6.824 | 0.856 |
| 256 | 0.250 | 2 | 16384 | 0.027 | 3.737 | 3.786 | 3.852 | 1.711 |
| 256 | 0.500 | 4 | 32768 | 0.056 | 7.234 | 7.311 | 1.995 | 3.424 |
| 256 | 0.752 | 7 | 49408 | 0.094 | 11.925 | 12.042 | 1.211 | 5.990 |
| 256 | 0.901 | 8 | 59392 | 0.104 | 14.052 | 14.176 | 1.029 | 6.845 |
| 256 | 1.000 | 8 | 65536 | 0.117 | 14.446 | 14.584 | 1.000 | 6.844 |
| 512 | 0.010 | 1 | 4096 | 0.015 | 2.038 | 2.072 | 27.336 | 0.856 |
| 512 | 0.050 | 2 | 16384 | 0.024 | 3.847 | 3.892 | 14.552 | 1.711 |
| 512 | 0.100 | 4 | 26624 | 0.049 | 7.006 | 7.074 | 8.007 | 3.422 |
| 512 | 0.250 | 8 | 65536 | 0.105 | 14.211 | 14.340 | 3.950 | 6.844 |
| 512 | 0.500 | 16 | 131072 | 0.238 | 28.919 | 29.181 | 1.941 | 13.694 |
| 512 | 0.749 | 24 | 196608 | 0.327 | 41.676 | 42.022 | 1.348 | 20.537 |
| 512 | 0.901 | 29 | 237568 | 0.379 | 51.511 | 51.909 | 1.091 | 24.813 |
| 512 | 1.000 | 32 | 262144 | 0.411 | 56.202 | 56.637 | 1.000 | 27.381 |

![](figures/piroi_hiz_kazanci.png)

## Saldırı A: yalnızca ROI meta verisi
| veri | hedef | roi | oznitelik_grubu | auc_ort | ci95_alt | ci95_ust | n |
|---|---|---|---|---|---|---|---|
| akciğer grafisi (COVID-QU-Ex) | teşhis (3 sınıf) | akciğer maskesi | konum | 0.729 | 0.720 | 0.739 | 6788 |
| akciğer grafisi (COVID-QU-Ex) | teşhis (3 sınıf) | akciğer maskesi | buyukluk | 0.765 | 0.757 | 0.775 | 6788 |
| akciğer grafisi (COVID-QU-Ex) | teşhis (3 sınıf) | akciğer maskesi | sekil | 0.780 | 0.771 | 0.788 | 6788 |
| akciğer grafisi (COVID-QU-Ex) | teşhis (3 sınıf) | akciğer maskesi | tumu | 0.834 | 0.827 | 0.841 | 6788 |
| akciğer grafisi (COVID-QU-Ex) | Normal / Non-COVID | akciğer maskesi | konum | 0.788 | 0.772 | 0.798 | 4393 |
| akciğer grafisi (COVID-QU-Ex) | Normal / Non-COVID | akciğer maskesi | buyukluk | 0.824 | 0.812 | 0.837 | 4393 |
| akciğer grafisi (COVID-QU-Ex) | Normal / Non-COVID | akciğer maskesi | sekil | 0.873 | 0.862 | 0.883 | 4393 |
| akciğer grafisi (COVID-QU-Ex) | Normal / Non-COVID | akciğer maskesi | tumu | 0.904 | 0.895 | 0.913 | 4393 |
| akciğer grafisi (COVID-QU-Ex) | Normal / COVID-19 | akciğer maskesi | konum | 0.779 | 0.765 | 0.791 | 4535 |
| akciğer grafisi (COVID-QU-Ex) | Normal / COVID-19 | akciğer maskesi | buyukluk | 0.777 | 0.767 | 0.793 | 4535 |
| akciğer grafisi (COVID-QU-Ex) | Normal / COVID-19 | akciğer maskesi | sekil | 0.790 | 0.780 | 0.803 | 4535 |
| akciğer grafisi (COVID-QU-Ex) | Normal / COVID-19 | akciğer maskesi | tumu | 0.853 | 0.842 | 0.863 | 4535 |
| akciğer grafisi (COVID-QU-Ex) | Non-COVID / COVID-19 | akciğer maskesi | konum | 0.694 | 0.680 | 0.708 | 4648 |
| akciğer grafisi (COVID-QU-Ex) | Non-COVID / COVID-19 | akciğer maskesi | buyukluk | 0.781 | 0.766 | 0.792 | 4648 |
| akciğer grafisi (COVID-QU-Ex) | Non-COVID / COVID-19 | akciğer maskesi | sekil | 0.783 | 0.770 | 0.795 | 4648 |
| akciğer grafisi (COVID-QU-Ex) | Non-COVID / COVID-19 | akciğer maskesi | tumu | 0.832 | 0.820 | 0.842 | 4648 |
| akciğer grafisi (COVID-QU-Ex, enfeksiyon alt kümesi) | COVID-19 / diğer | enfeksiyon (lezyon) maskesi | yalnızca ROI alanı | 1.000 |  |  | 5826 |
| akciğer grafisi (Kaggle, orijinal çözünürlük) | NORMAL / PNEUMONIA | (ROI'den bağımsız) | girdi_boyutu | 0.934 |  |  | 1172 |
| akciğer grafisi (Kaggle, orijinal çözünürlük) | NORMAL / COVID19 | (ROI'den bağımsız) | girdi_boyutu | 0.920 |  |  | 433 |
| beyin MR (Cheng) | tümör tipi (3 sınıf) | tümör maskesi | girdi_boyutu | 0.457 | 0.401 | 0.519 | 3064 |
| beyin MR (Cheng) | tümör tipi (3 sınıf) | tümör maskesi | konum | 0.775 | 0.743 | 0.804 | 3064 |
| beyin MR (Cheng) | tümör tipi (3 sınıf) | tümör maskesi | buyukluk | 0.708 | 0.674 | 0.736 | 3064 |
| beyin MR (Cheng) | tümör tipi (3 sınıf) | tümör maskesi | sekil | 0.722 | 0.691 | 0.747 | 3064 |
| beyin MR (Cheng) | tümör tipi (3 sınıf) | tümör maskesi | tumu | 0.830 | 0.801 | 0.855 | 3064 |

Beyin, sınıf bazında (bire karşı hepsi) AUC: {'meningioma': 0.7138445656240109, 'glioma': 0.826673910474752, 'pituitary': 0.9490617851276316}

Kesit yönü kontrolü: yalnızca yön AUC=0.540, yalnızca meta veri AUC=0.830, ikisi birlikte AUC=0.858; yön içinde: aksiyel 0.841 (n=1126), koronal 0.847 (n=969), sagital 0.891 (n=969). Görsel doğrulama (12'şer rastgele örnek): sagital 12/12, aksiyel ~9-10/12, koronal ~8/12.

## Saldırı B: bağlam (CNN)
| veri | gorus | tohum | gizli_alan_ort | auc | ci95_alt | ci95_ust | auc_Normal_vs_Non-COVID | auc_Normal_vs_COVID-19 | auc_Non-COVID_vs_COVID-19 |
|---|---|---|---|---|---|---|---|---|---|
| beyin MR (Cheng) | tam | 0 | 0.000 | 0.984 | 0.964 | 0.996 |  |  |  |
| beyin MR (Cheng) | tam | 1 | 0.000 | 0.985 | 0.967 | 0.996 |  |  |  |
| beyin MR (Cheng) | tam | 2 | 0.000 | 0.986 | 0.969 | 0.996 |  |  |  |
| beyin MR (Cheng) | tam | 3 | 0.000 | 0.985 | 0.967 | 0.995 |  |  |  |
| beyin MR (Cheng) | tam | 4 | 0.000 | 0.986 | 0.969 | 0.996 |  |  |  |
| beyin MR (Cheng) | baglam | 0 | 0.017 | 0.977 | 0.956 | 0.991 |  |  |  |
| beyin MR (Cheng) | baglam | 1 | 0.017 | 0.976 | 0.956 | 0.990 |  |  |  |
| beyin MR (Cheng) | baglam | 2 | 0.017 | 0.976 | 0.956 | 0.990 |  |  |  |
| beyin MR (Cheng) | baglam | 3 | 0.017 | 0.978 | 0.959 | 0.991 |  |  |  |
| beyin MR (Cheng) | baglam | 4 | 0.017 | 0.977 | 0.956 | 0.991 |  |  |  |
| beyin MR (Cheng) | baglam_genis40 | 0 | 0.248 | 0.966 | 0.944 | 0.983 |  |  |  |
| beyin MR (Cheng) | baglam_genis40 | 1 | 0.248 | 0.964 | 0.943 | 0.982 |  |  |  |
| beyin MR (Cheng) | baglam_genis40 | 2 | 0.248 | 0.968 | 0.950 | 0.983 |  |  |  |
| beyin MR (Cheng) | baglam_genis40 | 3 | 0.248 | 0.968 | 0.948 | 0.984 |  |  |  |
| beyin MR (Cheng) | baglam_genis40 | 4 | 0.248 | 0.970 | 0.950 | 0.985 |  |  |  |
| akciğer grafisi (COVID-QU-Ex) | tam | 0 | 0.000 | 0.997 | 0.996 | 0.998 | 0.993 | 1.000 | 1.000 |
| akciğer grafisi (COVID-QU-Ex) | tam | 1 | 0.000 | 0.997 | 0.996 | 0.997 | 0.993 | 1.000 | 1.000 |
| akciğer grafisi (COVID-QU-Ex) | tam | 2 | 0.000 | 0.996 | 0.995 | 0.997 | 0.992 | 1.000 | 1.000 |
| akciğer grafisi (COVID-QU-Ex) | tam | 3 | 0.000 | 0.997 | 0.996 | 0.997 | 0.992 | 1.000 | 1.000 |
| akciğer grafisi (COVID-QU-Ex) | tam | 4 | 0.000 | 0.996 | 0.995 | 0.997 | 0.992 | 1.000 | 1.000 |
| akciğer grafisi (COVID-QU-Ex) | baglam | 0 | 0.235 | 0.993 | 0.992 | 0.995 | 0.984 | 1.000 | 1.000 |
| akciğer grafisi (COVID-QU-Ex) | baglam | 1 | 0.235 | 0.993 | 0.992 | 0.994 | 0.983 | 1.000 | 1.000 |
| akciğer grafisi (COVID-QU-Ex) | baglam | 2 | 0.235 | 0.993 | 0.992 | 0.994 | 0.983 | 0.999 | 1.000 |
| akciğer grafisi (COVID-QU-Ex) | baglam | 3 | 0.235 | 0.993 | 0.992 | 0.994 | 0.983 | 1.000 | 1.000 |
| akciğer grafisi (COVID-QU-Ex) | baglam | 4 | 0.235 | 0.993 | 0.992 | 0.994 | 0.983 | 1.000 | 1.000 |
| akciğer grafisi (COVID-QU-Ex) | baglam_genis40 | 0 | 0.874 | 0.957 | 0.954 | 0.961 | 0.951 | 0.977 | 0.976 |
| akciğer grafisi (COVID-QU-Ex) | baglam_genis40 | 1 | 0.874 | 0.956 | 0.952 | 0.960 | 0.949 | 0.976 | 0.975 |
| akciğer grafisi (COVID-QU-Ex) | baglam_genis40 | 2 | 0.874 | 0.955 | 0.952 | 0.959 | 0.950 | 0.974 | 0.974 |
| akciğer grafisi (COVID-QU-Ex) | baglam_genis40 | 3 | 0.874 | 0.954 | 0.950 | 0.958 | 0.949 | 0.974 | 0.974 |
| akciğer grafisi (COVID-QU-Ex) | baglam_genis40 | 4 | 0.874 | 0.955 | 0.951 | 0.959 | 0.951 | 0.975 | 0.973 |
| beyin MR (Cheng) | yalniz_roi | 0 | 0.983 | 0.978 | 0.971 | 0.985 |  |  |  |
| beyin MR (Cheng) | baglam_genis10 | 0 | 0.051 | 0.972 | 0.951 | 0.987 |  |  |  |
| beyin MR (Cheng) | baglam_genis20 | 0 | 0.101 | 0.967 | 0.945 | 0.984 |  |  |  |
| beyin MR (Cheng) | kutu | 0 | 0.025 | 0.973 | 0.950 | 0.989 |  |  |  |
| akciğer grafisi (COVID-QU-Ex) | yalniz_roi | 0 | 0.765 | 0.990 | 0.989 | 0.992 | 0.991 | 0.995 | 0.995 |
| akciğer grafisi (COVID-QU-Ex) | baglam_genis10 | 0 | 0.425 | 0.991 | 0.990 | 0.993 | 0.979 | 0.999 | 0.999 |
| akciğer grafisi (COVID-QU-Ex) | baglam_genis20 | 0 | 0.609 | 0.989 | 0.987 | 0.991 | 0.975 | 0.999 | 0.998 |
| akciğer grafisi (COVID-QU-Ex) | kutu | 0 | 0.513 | 0.984 | 0.982 | 0.986 | 0.960 | 0.999 | 0.999 |

## Gerçekçilik testi
- Akciğer segmentasyonu: {'covidqu_val_dice': 0.9775937180690576, 'kaggle_maske_sayisi': 6432, 'kaggle_akciger_orani_medyan': 0.22841597576530612, 'kaggle_akciger_orani_cok_kucuk(<%5)': 0}
| yon | gorus | tohum | auc | ci95_alt | ci95_ust | n_egitim | n_test |
|---|---|---|---|---|---|---|---|
| COVID-QU-Ex -> Kaggle | tam | 0 | 0.999 | 0.998 | 1.000 | 17571 | 1497 |
| Kaggle -> COVID-QU-Ex | tam | 0 | 0.928 | 0.921 | 0.936 | 1215 | 4393 |
| COVID-QU-Ex -> COVID-QU-Ex (referans) | tam | 0 | 0.992 | 0.990 | 0.994 | 17571 | 4393 |
| COVID-QU-Ex -> Kaggle | baglam | 0 | 0.998 | 0.996 | 0.999 | 17571 | 1497 |
| Kaggle -> COVID-QU-Ex | baglam | 0 | 0.934 | 0.928 | 0.941 | 1215 | 4393 |
| COVID-QU-Ex -> COVID-QU-Ex (referans) | baglam | 0 | 0.984 | 0.980 | 0.987 | 17571 | 4393 |

## Kök neden
Grad-CAM dikkat kütlesinin bölgelere dağılımı:

| veri | sinif | akciger (gizli) | kenar cerceve | akciger ustu | akciger alti | akcigerler arasi | yanlar | tumor (gizli) | tumor cevresi | beynin geri kalani | arka plan |
|---|---|---|---|---|---|---|---|---|---|---|---|
| covidqu | COVID-19 | 0.287 | 0.263 | 0.022 | 0.075 | 0.176 | 0.177 |  |  |  |  |
| covidqu | Non-COVID | 0.259 | 0.257 | 0.030 | 0.060 | 0.191 | 0.201 |  |  |  |  |
| covidqu | Normal | 0.283 | 0.269 | 0.040 | 0.048 | 0.213 | 0.147 |  |  |  |  |
| covidqu | hepsi | 0.276 | 0.263 | 0.030 | 0.061 | 0.193 | 0.175 |  |  |  |  |
| brain | glioma |  |  |  |  |  |  | 0.033 | 0.104 | 0.440 | 0.423 |
| brain | meningioma |  |  |  |  |  |  | 0.032 | 0.099 | 0.485 | 0.300 |
| brain | pituitary |  |  |  |  |  |  | 0.024 | 0.124 | 0.720 | 0.132 |
| brain | hepsi |  |  |  |  |  |  | 0.030 | 0.109 | 0.534 | 0.305 |

Önişleme ablasyonu:

| veri | onisleme | auc_3sinif |
|---|---|---|
| covidqu | yok | 0.993 |
| covidqu | histogram_esitleme | 0.993 |
| covidqu | kenar_gizle_5 | 0.993 |
| covidqu | kenar_gizle_10 | 0.990 |
| covidqu | esitleme+kenar_10 | 0.988 |

![](figures/kok_neden_gradcam_covidqu.png)

![](figures/kok_neden_gradcam_brain.png)

## Savunma ve gerçek bedel
| veri | politika | adim | gizli_alan | auc | hiz_kazanci_256 | hiz_kazanci_512 |
|---|---|---|---|---|---|---|
| covidqu | tamamen_gizli | 0 | 1.000 | 0.500 | 1.000 | 1.000 |
| covidqu | goruntu_roi | 0 | 0.234 | 0.990 | 4.038 | 4.174 |
| covidqu | kanonik | 0 | 0.633 | 0.974 | 1.486 | 1.570 |
| covidqu | kanonik_genis8 | 0 | 0.795 | 0.957 | 1.152 | 1.257 |
| covidqu | kanonik_genis16 | 0 | 0.911 | 0.910 | 1.026 | 1.081 |
| covidqu | kanonik_genis32 | 0 | 0.996 | 0.681 | 1.001 | 1.003 |
| covidqu | kanonik_genis64 | 0 | 1.000 | 0.500 | 1.000 | 1.000 |
| covidqu | sizinti_gudumlu | 1 | 0.692 | 0.965 | 1.336 | 1.448 |
| covidqu | sizinti_gudumlu | 2 | 0.744 | 0.946 | 1.227 | 1.356 |
| covidqu | sizinti_gudumlu | 3 | 0.782 | 0.940 | 1.169 | 1.281 |
| covidqu | sizinti_gudumlu | 4 | 0.827 | 0.919 | 1.112 | 1.203 |
| covidqu | sizinti_gudumlu | 5 | 0.862 | 0.899 | 1.071 | 1.148 |
| covidqu | sizinti_gudumlu | 6 | 0.900 | 0.883 | 1.030 | 1.093 |
| covidqu | sizinti_gudumlu | 7 | 0.931 | 0.864 | 1.020 | 1.062 |
| covidqu | sizinti_gudumlu | 8 | 0.955 | 0.830 | 1.013 | 1.039 |
| covidqu | sizinti_gudumlu | 9 | 0.981 | 0.776 | 1.005 | 1.016 |
| covidqu | sizinti_gudumlu | 10 | 1.000 | 0.500 | 1.000 | 1.000 |
| covidqu | rastgele_0 | 2 | 0.721 | 0.966 | 1.274 | 1.396 |
| covidqu | rastgele_0 | 4 | 0.789 | 0.959 | 1.160 | 1.269 |
| covidqu | rastgele_0 | 6 | 0.866 | 0.948 | 1.067 | 1.141 |
| covidqu | rastgele_0 | 8 | 0.933 | 0.920 | 1.019 | 1.060 |
| covidqu | rastgele_0 | 10 | 1.000 | 0.500 | 1.000 | 1.000 |
| covidqu | rastgele_1 | 2 | 0.729 | 0.958 | 1.256 | 1.381 |
| covidqu | rastgele_1 | 4 | 0.809 | 0.950 | 1.134 | 1.233 |
| covidqu | rastgele_1 | 6 | 0.866 | 0.936 | 1.067 | 1.141 |
| covidqu | rastgele_1 | 8 | 0.937 | 0.895 | 1.018 | 1.056 |
| covidqu | rastgele_1 | 10 | 1.000 | 0.500 | 1.000 | 1.000 |
| brain | tamamen_gizli | 0 | 1.000 | 0.500 | 1.000 | 1.000 |
| brain | goruntu_roi | 0 | 0.016 | 0.941 | 8.645 | 24.240 |
| brain | kanonik | 0 | 0.334 | 0.910 | 2.936 | 2.933 |
| brain | kanonik_genis8 | 0 | 0.439 | 0.920 | 2.262 | 2.217 |
| brain | kanonik_genis16 | 0 | 0.550 | 0.918 | 1.767 | 1.783 |
| brain | kanonik_genis32 | 0 | 0.791 | 0.892 | 1.157 | 1.265 |
| brain | kanonik_genis64 | 0 | 1.000 | 0.619 | 1.000 | 1.000 |
| brain | sizinti_gudumlu | 1 | 0.396 | 0.910 | 2.494 | 2.460 |
| brain | sizinti_gudumlu | 2 | 0.457 | 0.896 | 2.174 | 2.126 |
| brain | sizinti_gudumlu | 3 | 0.516 | 0.901 | 1.915 | 1.886 |
| brain | sizinti_gudumlu | 4 | 0.568 | 0.900 | 1.697 | 1.732 |
| brain | sizinti_gudumlu | 5 | 0.623 | 0.893 | 1.515 | 1.593 |
| brain | sizinti_gudumlu | 6 | 0.672 | 0.889 | 1.383 | 1.487 |
| brain | sizinti_gudumlu | 7 | 0.720 | 0.892 | 1.275 | 1.397 |
| brain | sizinti_gudumlu | 8 | 0.778 | 0.872 | 1.175 | 1.289 |
| brain | sizinti_gudumlu | 9 | 0.841 | 0.859 | 1.096 | 1.180 |
| brain | sizinti_gudumlu | 10 | 0.884 | 0.842 | 1.047 | 1.115 |
| brain | sizinti_gudumlu | 11 | 0.946 | 0.859 | 1.015 | 1.047 |
| brain | sizinti_gudumlu | 12 | 0.996 | 0.811 | 1.001 | 1.004 |
| brain | sizinti_gudumlu | 13 | 1.000 | 0.500 | 1.000 | 1.000 |
| brain | rastgele_0 | 2 | 0.434 | 0.913 | 2.283 | 2.239 |
| brain | rastgele_0 | 4 | 0.529 | 0.916 | 1.854 | 1.844 |
| brain | rastgele_0 | 6 | 0.632 | 0.913 | 1.490 | 1.573 |
| brain | rastgele_0 | 8 | 0.736 | 0.894 | 1.242 | 1.369 |
| brain | rastgele_0 | 10 | 0.819 | 0.894 | 1.122 | 1.216 |
| brain | rastgele_0 | 12 | 0.938 | 0.862 | 1.018 | 1.056 |
| brain | rastgele_0 | 13 | 1.000 | 0.500 | 1.000 | 1.000 |
| brain | rastgele_1 | 2 | 0.424 | 0.909 | 2.335 | 2.294 |
| brain | rastgele_1 | 4 | 0.517 | 0.897 | 1.910 | 1.883 |
| brain | rastgele_1 | 6 | 0.622 | 0.886 | 1.519 | 1.596 |
| brain | rastgele_1 | 8 | 0.734 | 0.881 | 1.246 | 1.372 |
| brain | rastgele_1 | 10 | 0.850 | 0.878 | 1.085 | 1.165 |
| brain | rastgele_1 | 12 | 0.938 | 0.858 | 1.018 | 1.056 |
| brain | rastgele_1 | 13 | 1.000 | 0.500 | 1.000 | 1.000 |

```json
{
  "brain": {
    "AUC<=0.8 icin min gizli alan (sizinti_gudumlu)": 1.0,
    "AUC<=0.7 icin min gizli alan (sizinti_gudumlu)": 1.0,
    "AUC<=0.6 icin min gizli alan (sizinti_gudumlu)": 1.0,
    "goruntu_roi AUC": 0.9409431483307976,
    "kanonik AUC": 0.9099113222382939
  },
  "covidqu": {
    "AUC<=0.8 icin min gizli alan (sizinti_gudumlu)": 0.9809036980635467,
    "AUC<=0.7 icin min gizli alan (sizinti_gudumlu)": 1.0,
    "AUC<=0.6 icin min gizli alan (sizinti_gudumlu)": 1.0,
    "goruntu_roi AUC": 0.9895731284478012,
    "kanonik AUC": 0.9737116284276577
  }
}
```

![](figures/savunma_covidqu.png)

![](figures/savunma_brain.png)
