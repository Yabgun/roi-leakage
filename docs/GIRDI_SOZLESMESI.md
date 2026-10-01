# Girdi sözleşmesi: modeller ne bekler (yeni ve projede kullanılmamış görüntülerle test için)

Modeller makaledeki iki veri kümesinde eğitilmiştir. Başka bir kaynaktan gelen görüntülerle test yaparken aşağıdakilere
dikkat edilmelidir. Tahmin komutu: `python -m boruhatti.tahmin` (ayrıntı dosyanın başında).

## Ortak

| Konu | Beklenen |
|---|---|
| Sınıflar | Beyin MR: meningiom, gliom, hipofiz tümörü. Göğüs röntgeni: COVID-19, COVID dışı pnömoni (viral ya da bakteriyel), normal. Bu sınıflar dışındaki görüntüler (ör. tümörsüz beyin, tüberküloz) için çıktı anlamsızdır: model yine bu üç sınıftan birini seçer. |
| Görüntü | Tek kanallı (gri ton) 2B görüntü; PNG, JPG, BMP ya da TIFF. Renkli görüntü griye çevrilir. |
| Boyut | Kare olmayan görüntü en-boy oranı korunmadan kareye getirilir. En iyi sonuç kare ya da kareye yakın görüntülerle alınır. |
| ROI maskesi | Görüntüyle aynı boyutta ikili PNG (beyaz = ROI). `saldirgan_baglam` ve `foveahe` için gerekir; `saldirgan_tam` için gerekmez. |
| Modeller | Beyin MR'da 5 katın modellerinin ortalaması; göğüs röntgeninde tek model (tohum 0). |

## Beyin MR

- **Modalite:** T1 ağırlıklı, kontrast madde verilmiş MR kesiti (figshare 1512427 ile aynı tür). Aksiyel, koronal ya da sagital olabilir. T2, FLAIR ya da kontrastsız T1 kesitlerde başarı düşer.
- **Yoğunluk:** Kesit başına 0.5 ve 99.5 yüzdelikleri arasında ölçeklenir (eğitimdekiyle aynı). Ham DICOM yerine PNG/JPG verilecekse pencereleme yapılmadan dışa aktarılmalıdır.
- **ROI maskesi:** Tümör maskesi kullanıcıdan gelmelidir; boru hattında tümör bölütleme modeli yoktur. Maske yoksa yalnız `saldirgan_tam` çalışır.
- **Dikkat:** Kaggle'daki bazı beyin tümörü kümeleri bu çalışmanın figshare görüntülerini içerir. Örneğin yaygın kullanılan "Brain Tumor MRI Dataset" (`masoudnickparvar/brain-tumor-mri-dataset`), açıklamasına göre figshare, SARTAJ ve Br35H kümelerinin birleşimidir. Dış test yapılmadan önce `python -m boruhatti.kopya_kontrol --klasor <klasör> --veri beyin` ile kopyalar ayıklanmalıdır.

## Göğüs röntgeni

- **Modalite:** Arka-ön (PA) ya da ön-arka (AP) göğüs grafisi. Yan grafiler eğitim verisinde azdır.
- **ROI maskesi:** `--otomatik-maske` verilirse akciğer maskesini çalışmadaki U-Net çıkarır (COVID-QU-Ex doğrulama kümesinde Dice 0.978). Kendi maskeniz varsa `--maske` ile verin.
- **Dikkat:** Kaggle'daki göğüs röntgeni derlemeleri sıklıkla aynı kaynakları paylaşır. Makalede Kaggle CXR'nin 6432 görüntüsünün 4665'inin COVID-QU-Ex içinde karşılığı bulundu; iki küme aynı çocuk hastalar kaynağını paylaşır (makale, Sınırlılıklar). Önce `python -m boruhatti.kopya_kontrol --klasor <klasör> --veri covidqu` çalıştırılmalıdır.

## Bilinen sınırlar (makale, Sınırlılıklar)

- Mutlak AUC değerleri veri kaynağına özgü izlerden (çekim cihazı, hastane, hasta yaşı) etkilenmiş olabilir. Farklı hastanenin verisinde başarı düşebilir. Makalenin asıl iddiası sınıflandırma başarısı değil; açık bırakılan bağlamın teşhisi ele verdiği ve FoveaHE'nin bunu sızdırmadan yaptığıdır.
- FoveaHE ve bağlam saldırganı ROI'nin istemcide bilindiğini varsayar (maske gerekir).
- Şifreli modeller sığdır (Model D, D2, C); tek bir CKKS parametre kümesi denenmiştir.

## Çalışmanın kendi bölmeleri içinde kopya denetimi

`python -m boruhatti.kopya_kontrol --ic` (sonuç: `results/tables/kopya_ic_ozet.md`, `results/tables/kopya_etki.md`):

- **COVID-QU-Ex:** Test görüntülerinin %1.2'sinin (79) Train + Val'de çok benzer bir eşi var.
- **Beyin MR:** Kesitlerin %1'inin (31) başka katta çok benzer bir eşi var. Hiçbiri aynı hasta kimliğinden değil; bir kısmı yalnız son harfiyle ayrılan kimliklerden (ör. MR024780B / MR024780E). Bunlar aynı kişinin farklı çekimleri olabilir.
- **Etki:** Bu görüntüler testten çıkarılınca makaledeki ana AUC'ler en fazla 0.0007 değişir. Kök kimliği başka katta olan bütün kesitler (734) de çıkarılırsa değişim en fazla 0.008'dir ve hiçbir bulgu değişmez.
