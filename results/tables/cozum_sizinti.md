# Çözüm Adım 4: sızıntı denetimi

Ana ölçü kat ortalaması saldırgan AUC'si (5 tohum; sınıf dengeli katlar, beyinde hasta bazlı). `bos_p95`: etiket permütasyonuyla boş dağılımın %95'lik değeri; `sans_duzeyinde = evet` gözlenen AUC bu değeri aşmıyor demektir. Π_ROI satırları aynı örneklem ve protokolle pozitif kontroldür.

| veri | yontem | saldiri | oznitelikler | n | auc_kat_ort | auc_std | bos_p95 | p_degeri | sans_duzeyinde | auc_havuz |
|---|---|---|---|---|---|---|---|---|---|---|
| beyin MR (Cheng) | Π_ROI | A: paket meta verisi | ciphertext sayısı, yükleme boyutu | 3064 | 0.5614 | 0.0000 | 0.5093 | 0.0099 | hayır | 0.5508 |
| beyin MR (Cheng) | FoveaHE | A: paket meta verisi | ciphertext sayısı, slot sayısı, yükleme baytı | 3064 | 0.4905 | 0.0000 | 0.5202 | 0.8515 | evet | 0.4894 |
| beyin MR (Cheng) | Π_ROI | yan kanal (alt küme) | ciphertext sayısı, yükleme boyutu (süre bunlarla doğrusal) | 450 | 0.5567 | 0.0000 | 0.5203 | 0.0099 | hayır | 0.5471 |
| beyin MR (Cheng) | FoveaHE | yan kanal (alt küme) | sunucu süresi, indirme baytı | 450 | 0.5258 | 0.0000 | 0.5462 | 0.1980 | evet | 0.5241 |
| beyin MR (Cheng) | FoveaHE | yan kanal (alt küme) | yalnızca sunucu süresi | 450 | 0.4844 | 0.0000 | 0.5398 | 0.7327 | evet | 0.4851 |
| akciğer grafisi (COVID-QU-Ex) | Π_ROI | A: paket meta verisi | ciphertext sayısı, yükleme boyutu | 3000 | 0.5732 | 0.0000 | 0.5143 | 0.0099 | hayır | 0.5648 |
| akciğer grafisi (COVID-QU-Ex) | FoveaHE | A: paket meta verisi | ciphertext sayısı, slot sayısı, yükleme baytı | 3000 | 0.5027 | 0.0000 | 0.5178 | 0.4257 | evet | 0.5027 |
| akciğer grafisi (COVID-QU-Ex) | Π_ROI | yan kanal (alt küme) | ciphertext sayısı, yükleme boyutu (süre bunlarla doğrusal) | 450 | 0.5889 | 0.0000 | 0.5327 | 0.0099 | hayır | 0.5605 |
| akciğer grafisi (COVID-QU-Ex) | FoveaHE | yan kanal (alt küme) | sunucu süresi, indirme baytı | 450 | 0.4682 | 0.0000 | 0.5350 | 0.8416 | evet | 0.4715 |
| akciğer grafisi (COVID-QU-Ex) | FoveaHE | yan kanal (alt küme) | yalnızca sunucu süresi | 450 | 0.4480 | 0.0000 | 0.5325 | 0.9802 | evet | 0.4487 |

## Referanslar (kesin tablolardan)

| veri | yontem | saldiri | oznitelikler | n | auc | kaynak |
|---|---|---|---|---|---|---|
| beyin MR (Cheng) | Π_ROI | A: ROI meta verisi | konum, boyut, şekil | 3064.0000 | 0.8299 | saldiri_A_meta_veri.csv (5 tohum, havuzlanmış kat dışı) |
| beyin MR (Cheng) | Π_ROI | B: açık bağlam (ROI gizli) | açık pikseller | 3064.0000 | 0.9765 | saldiri_B_baglam.csv (5 tohum ortalaması) |
| beyin MR (Cheng) | FoveaHE | B: açık bağlam | açık piksel yok |  |  | tanımsal olarak uygulanamaz: sunucu yalnızca ciphertext görür |
| beyin MR (Cheng) | akıl sağlığı | sunucu görüşü sabit görüntü | bilgisiz girdi | 3064.0000 | 0.5000 | cozum_bilgi.csv, Adım 1 sabit (beyinde kat ortalaması) |
| akciğer grafisi (COVID-QU-Ex) | Π_ROI | A: ROI meta verisi | konum, boyut, şekil | 6788.0000 | 0.8337 | saldiri_A_meta_veri.csv (5 tohum, havuzlanmış kat dışı) |
| akciğer grafisi (COVID-QU-Ex) | Π_ROI | B: açık bağlam (ROI gizli) | açık pikseller | 6788.0000 | 0.9932 | saldiri_B_baglam.csv (5 tohum ortalaması) |
| akciğer grafisi (COVID-QU-Ex) | FoveaHE | B: açık bağlam | açık piksel yok |  |  | tanımsal olarak uygulanamaz: sunucu yalnızca ciphertext görür |
| akciğer grafisi (COVID-QU-Ex) | akıl sağlığı | sunucu görüşü sabit görüntü | bilgisiz girdi | 6788.0000 | 0.5000 | cozum_bilgi.csv, Adım 1 sabit (beyinde kat ortalaması) |

## Sızıntı fonksiyonları: sunucunun öğrendiği

| Yöntem | Sunucunun öğrendiği |
|---|---|
| Π_ROI (ePrint 2026/103) | model ağırlıkları; ROI dışındaki tüm açık pikseller; ROI konumu, boyutu ve şekli; girdi boyutu; ROI ciphertext sayısı ve yükleme boyutu (ROI alanıyla değişir) |
| Encrypt What Matters (arXiv 2609.09357) | ROI dışı açık bölge ve ROI yerleşimi (ROI dışı açık kabul edilir) |
| Bi-CryptoNets (arXiv 2402.01296) | hassas olmayan kısım (gürültü eklenmiş ama açık) |
| FoveaHE (odaklı tam şifreleme) | yalnızca herkese açık sabitler: CKKS parametreleri (N = 16384), slot sayısı (8.192), ciphertext sayısı (1), model mimarisi; her hasta için aynı |

## Protokol kuralı (IND-CPA-D)

- Çözülen sonuç (logit, olasılık ya da karar) sunucuya geri gönderilmez; sunucuya çözme kâhini verilmez.
- Gerekçe: CKKS'de çözülmüş sonuçlara erişen sunucu gizli anahtarı kurtarabilir (IND-CPA-D; bu projedeki PoC `pilot/indcpad_poc_tenseal.py`, TenSEAL ile 0.15 s).
- Model ağırlıkları sunucuda açık metindir (Π_ROI ile aynı tehdit modeli); model gizliliği kapsam dışıdır.
