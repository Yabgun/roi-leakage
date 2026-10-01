"""Uçtan uca boru hattı: veriyi özgün kaynağından indir → ön işle → eğit → model dosyası → tahmin → değerlendir.

Danışman geri bildirimi (1 Ekim 2026) için eklenmiştir; makaledeki deney kodunu (`experiments/`) değiştirmeden
üzerine kurulur. Adımlar ve kademeler için depo kökündeki README ve `calistir.py`.

Modüller:
- ortam_kontrol: Python, paketler, ekran kartı, sürücü, disk denetimi
- veri_indir: Kaggle (kagglehub, sürüm sabit) ve figshare'den indirme, parmak iziyle doğrulama
- on_isle: manifestolar, beyin MR PNG/maskeler, Kaggle akciğer maskeleri, görünür normalizasyon önbelleği, FoveaHE katmanları
- egit: saldırgan (ResNet-18) ve şifreli modeller (D, D2, C); model dosyası, eğitim eğrileri (yerel + wandb)
- degerlendir: kayıtlı modellerle test; ölçüler, karışıklık matrisi, görüntü başına tahminler, makaleyle karşılaştırma
- tahmin: tek görüntüye tahmin (yeni ve projede hiç kullanılmamış görüntüler için)
- kontrol: sonuçların beklenen değerlerle (makale) toleranslı karşılaştırması
- kopya_kontrol: veri kümeleri arasında ve yeni bir klasörle kopya görüntü taraması
"""
