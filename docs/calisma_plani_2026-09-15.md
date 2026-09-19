# Çözüm çalışma planı — dondurulan başarı ölçütleri (15 Eylül 2026)

Bu dosya, önerilen yöntemin (FoveaHE) başarı ölçütlerinin çözüm deneyleri
başlamadan **önce** belirlendiğini belgelemek için, çalışma planından (özel not
`Tez/COZUM_PLANI.md` §6) çıkarılmış tarihli bir kopyadır. Ölçütler 15 Eylül 2026'da
donduruldu ve sonradan değiştirilmedi.

## Başarı ölçütleri (15 Eylül 2026'da donduruldu)

Değiştirilirse gerekçesiyle not düşülür.

| Boyut | Ölçüt | Hedef |
|---|---|---|
| Gizlilik (sızıntı) | Sunucunun gördüğü veriye Saldırı A ve B, yan kanal kontrolü | AUC ≤ 0.55 |
| Temsilin bilgi kaybı (Adım 1) | ResNet-18: odaklı temsil ile tam görüntü farkı | ≤ 0.02 AUC |
| Şifreli model doğruluğu | Aynı model sınıfında odaklı temsil ile tam görüntü farkı | ≤ 0.03 AUC |
| Şifreleme doğruluğu | Şifreli ve şifresiz sonuç farkı | ≤ 0.005 AUC |
| Hız | Tam şifrelemeye göre toplam süre oranı | ≥ 5× |

Hedef tutmazsa dürüstçe raporlanır; tez saldırı katkısıyla yine ayakta kalır.

## Sürüm izi (git)

- Bilgi kaybı, model kaybı ve şifreli doğruluk eşikleri, ilk sonuçlardan önce deney
  betikleriyle birlikte kod deposuna işlendi: commit `1be0d66` (`fovea_info`,
  ölçüt ≤ 0.02), `edf0933` (`fovea_models`, ölçüt ≤ 0.03), `868002d`
  (`fovea_cost`, ölçüt ≤ 0.005). İlk sonuç commit'i `803a90c` (15 Eylül 2026 21:46).
- Sızıntı (≤ 0.55) ve hız (≥ 5×) eşikleri kod deposuna ilk kez sonuçlarla birlikte
  kaydedildi; bu iki eşiğin önceden belirlendiği bu çalışma planı belgesiyle
  belgelenmektedir.
