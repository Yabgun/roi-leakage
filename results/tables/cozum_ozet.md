| veri | yontem | model | sifreli_deger | teshis_auc | teshis_auc_std | tohum_sayisi | saldirgan_auc | sure_s | hiz_kazanci | sizinti_turu |
|---|---|---|---|---|---|---|---|---|---|---|
| beyin MR (Cheng) | tam şifreleme | D2 | 262144 | 0.8819 |  | 1 | 0.5258 | 44.4573 | 1.0000 | yok |
| beyin MR (Cheng) | Π_ROI (varsayılan ROI) | D2 | 262144 | 0.8819 |  | 1 | 0.9765 | 1.8341 | 24.2397 | açık pikseller + ROI meta verisi |
| beyin MR (Cheng) | Π_ROI (gizlilik şartlı, saldırgan ≤ 0.8) | D2 | 262144 | 0.8819 |  | 1 | 0.8000 | 44.4514 | 1.0001 | sınırlı açık piksel |
| beyin MR (Cheng) | FoveaHE F32_G16 | C | 1280 | 0.9505 |  | 1 | 0.5258 | 0.4766 | 21.3168 | yok |
| beyin MR (Cheng) | FoveaHE F64_G32 | D | 5123 | 0.9468 |  | 1 | 0.5258 | 1.3974 | 31.8150 | yok |
| beyin MR (Cheng) | şifreli özet (ImageNet ResNet-18) | D | 512 | 0.9558 |  | 1 | 0.5258 | 1.0101 | 44.0111 | yok |
| beyin MR (Cheng) | bozuk bağlam (gauss 0.4) | D2 | 262144 | 0.8819 |  | 1 | 0.9685 | 1.8341 | 24.2397 | gürültülü/bulanık açık bağlam |
| akciğer grafisi (COVID-QU-Ex) | tam şifreleme | D2 | 65536 | 0.9465 |  | 1 | 0.5027 | 10.3579 | 1.0000 | yok |
| akciğer grafisi (COVID-QU-Ex) | Π_ROI (varsayılan ROI) | D2 | 65536 | 0.9465 |  | 1 | 0.9932 | 2.5648 | 4.0385 | açık pikseller + ROI meta verisi |
| akciğer grafisi (COVID-QU-Ex) | Π_ROI (gizlilik şartlı, saldırgan ≤ 0.8) | D2 | 65536 | 0.9465 |  | 1 | 0.8000 | 10.3039 | 1.0052 | sınırlı açık piksel |
| akciğer grafisi (COVID-QU-Ex) | FoveaHE F32_G16 | D2 | 1283 | 0.9517 |  | 1 | 0.5027 | 1.2990 | 7.9739 | yok |
| akciğer grafisi (COVID-QU-Ex) | FoveaHE F64_G32 | D2 | 5123 | 0.9504 |  | 1 | 0.5027 | 1.4882 | 6.9599 | yok |
| akciğer grafisi (COVID-QU-Ex) | şifreli özet (ImageNet ResNet-18) | D2 | 512 | 0.9812 |  | 1 | 0.5027 | 1.0601 | 9.7703 | yok |
| akciğer grafisi (COVID-QU-Ex) | bozuk bağlam (gauss 0.4) | D2 | 65536 | 0.9465 |  | 1 | 0.9834 | 2.5648 | 4.0385 | gürültülü/bulanık açık bağlam |
