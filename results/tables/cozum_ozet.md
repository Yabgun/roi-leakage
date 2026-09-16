| veri | yontem | model | sifreli_deger | teshis_auc | teshis_auc_std | tohum_sayisi | saldirgan_auc | sure_s | hiz_kazanci | sizinti_turu |
|---|---|---|---|---|---|---|---|---|---|---|
| beyin MR (Cheng) | tam şifreleme | D | 262144 | 0.8811 | 0.0044 | 5 | 0.5258 | 44.4573 | 1.0000 | yok |
| beyin MR (Cheng) | Π_ROI (varsayılan ROI) | D | 262144 | 0.8811 | 0.0044 | 5 | 0.9759 | 1.8366 | 24.2063 | açık pikseller + ROI meta verisi |
| beyin MR (Cheng) | Π_ROI (gizlilik şartlı, saldırgan ≤ 0.8) | D | 262144 | 0.8811 | 0.0044 | 5 | 0.8000 | 44.4506 | 1.0002 | sınırlı açık piksel |
| beyin MR (Cheng) | FoveaHE F32_G16 | D | 1283 | 0.9479 | 0.0010 | 5 | 0.5258 | 1.2135 | 36.6351 | yok |
| beyin MR (Cheng) | FoveaHE F64_G32 | D | 5123 | 0.9454 | 0.0017 | 5 | 0.5258 | 1.3974 | 31.8150 | yok |
| beyin MR (Cheng) | şifreli özet (ImageNet ResNet-18) | D | 512 | 0.9551 | 0.0010 | 5 | 0.5258 | 1.0101 | 44.0111 | yok |
| beyin MR (Cheng) | bozuk bağlam (gauss 0.4) | D | 262144 | 0.8811 | 0.0044 | 5 | 0.9670 | 1.8366 | 24.2063 | gürültülü/bulanık açık bağlam |
| akciğer grafisi (COVID-QU-Ex) | tam şifreleme | D2 | 65536 | 0.9460 | 0.0012 | 5 | 0.5027 | 10.3579 | 1.0000 | yok |
| akciğer grafisi (COVID-QU-Ex) | Π_ROI (varsayılan ROI) | D2 | 65536 | 0.9460 | 0.0012 | 5 | 0.9932 | 2.6885 | 3.8527 | açık pikseller + ROI meta verisi |
| akciğer grafisi (COVID-QU-Ex) | Π_ROI (gizlilik şartlı, saldırgan ≤ 0.8) | D2 | 65536 | 0.9460 | 0.0012 | 5 | 0.8000 | 10.3039 | 1.0052 | sınırlı açık piksel |
| akciğer grafisi (COVID-QU-Ex) | FoveaHE F32_G16 | D2 | 1283 | 0.9543 | 0.0017 | 5 | 0.5027 | 1.2990 | 7.9739 | yok |
| akciğer grafisi (COVID-QU-Ex) | FoveaHE F64_G32 | D2 | 5123 | 0.9519 | 0.0013 | 5 | 0.5027 | 1.4882 | 6.9599 | yok |
| akciğer grafisi (COVID-QU-Ex) | şifreli özet (ImageNet ResNet-18) | D2 | 512 | 0.9815 | 0.0009 | 5 | 0.5027 | 1.0601 | 9.7703 | yok |
| akciğer grafisi (COVID-QU-Ex) | bozuk bağlam (gauss 0.4) | D2 | 65536 | 0.9460 | 0.0012 | 5 | 0.9834 | 2.6885 | 3.8527 | gürültülü/bulanık açık bağlam |
