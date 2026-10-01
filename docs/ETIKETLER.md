# Etiketler ve modellerin girdi/çıktısı

Bu dosya `python -m analysis.tahminler` ile koddaki etiket listelerinden üretilir.

## Etiket numaraları

| Veri kümesi | Numara | Sınıf (koddaki ad) | Türkçe ad | Kaynak etiketi |
|---|---|---|---|---|
| Beyin MR (figshare 1512427) | 0 | meningioma | Meningiom | `cjdata.label` = 1 |
| Beyin MR (figshare 1512427) | 1 | glioma | Gliom | `cjdata.label` = 2 |
| Beyin MR (figshare 1512427) | 2 | pituitary | Hipofiz tümörü | `cjdata.label` = 3 |
| COVID-QU-Ex | 0 | COVID-19 | COVID-19 | klasör adı `COVID-19` |
| COVID-QU-Ex | 1 | Non-COVID | COVID dışı pnömoni | klasör adı `Non-COVID` |
| COVID-QU-Ex | 2 | Normal | Normal | klasör adı `Normal` |
| Kaggle CXR (prashant268) | – | COVID19 / NORMAL / PNEUMONIA | COVID-19 / Normal / Pnömoni | klasör adı |

Kaggle kümesi meta veri (girdi boyutu) deneyinde, gerçekçilik testinde (bir yönde saldırganın eğitim kümesi, diğer yönde test kümesi) ve genelleme testinde (yalnız test) kullanılır.

İkili görevler: Tablo II'deki "normal/pnömoni" satırları ve gerçekçilik testi normal ile pnömoniyi ayırır; normal = 0, pnömoni = 1 (COVID-QU-Ex'te `Non-COVID`, Kaggle'da `PNEUMONIA`).

## Her model neyi görür, neyi tahmin eder

Makalenin ana tablolarında hedef (etiket) bütün modellerde aynıdır: beyin MR'da tümör tipi, göğüs röntgeninde 3 sınıflı teşhis. Modeller **girdileri** bakımından ayrılır.

| Model | Makalede | Girdi | Çıktı |
|---|---|---|---|
| Meta veri saldırganı (HistGradientBoosting) | Tablo II | Π_ROI'nin sunucuya açtığı ROI konumu, büyüklüğü, şekli ve girdi boyutu (sayısal öznitelikler) | 3 sınıf olasılığı |
| Bağlam saldırganı (ImageNet ön eğitimli ResNet-18) | Tablo III | Sunucunun gördüğü 224×224 görüntü: ROI pikselleri sıfır + gizli bölge göstergesi kanalı | 3 sınıf olasılığı |
| Model D / D2 / C | Tablo V, VIII | Tam görüntü (Π_ROI'nin teşhis modeli) ya da FoveaHE temsili; şifreli çalışır (CKKS) | 3 sınıf skoru (şifreli; istemci çözer) |
| Şifreli özet + Model D / D2 | Tablo VIII | İstemcide ResNet-18'in çıkardığı 512 değerlik özet; şifreli | 3 sınıf skoru (şifreli) |
| U-Net | Metin (gerçekçilik testi) | Göğüs röntgeni | Akciğer maskesi (etiket değil) |

Saldırganlar sunucunun yerine geçer: gizlenmiş bilgiyi (teşhisi) sunucunun görebildiğinden tahmin etmeye çalışır. Başarılı olmaları sızıntı demektir. Model D, D2, C ve şifreli özet ise hastanenin istediği teşhis hizmetidir; başarılı olmaları istenir.
