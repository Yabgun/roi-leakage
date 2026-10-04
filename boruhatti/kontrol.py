"""Yeniden üretim denetimi: sonuçları beklenen değerlerle karşılaştırır; TUTTU / TUTMADI / ATLANDI yazar (madde 6).

Denetlenenler (hangisinin çıktısı varsa):
1. Makaledeki şifreli modeller (`degerlendir --makale-modelleri`): 5 tohum AUC ortalaması makaledeki Tablo V ve VIII
   değerine 3 basamakta eşit olmalı. Ağırlıklar aynı olduğundan fark yalnızca kayan nokta düzeyindedir.
2. Gerçek CKKS şifreli çıkarım (`degerlendir --sifreli`): şifreli ve şifresiz logit farkı ≤ 1e-3, tahmin uyumu %100.
3. Saldırgan modelleri (`degerlendir --saldirgan`): yeniden eğitilmiş ya da indirilmiş modelin AUC'si makaledeki 5 tohum
   ortalamasından en fazla `saldirgan_auc` toleransı kadar farklı olabilir. Ekran kartı hesapları bit bit tekrarlanmadığı
   için aynı tohumla bile küçük fark olur. Tolerans bu çalışmanın kendi yeniden koşularından belirlenmiştir
   (`results/beklenen/toleranslar.json`).
4. Yeniden eğitilen şifreli modeller (`degerlendir --foveahe`): AUC, makaledeki 5 tohumun aynı test bölmesindeki
   AUC aralığında (± `foveahe_auc_pay`) olmalı. Bu bilgisayarda ağırlıklar makaledekiyle bit bit aynıdır; başka bir
   ekran kartında kayan nokta farkı erken durdurmayı kaydırabilir, o zaman fark en çok bir tohum değişikliği kadardır.
5. Ezber denetimi (`boruhatti.ezber_denetimi`): gerçek etiketlerle eğitilen modelin test AUC'si, etiketleri
   karıştırılarak eğitilen bütün koşuların test AUC'sinden yüksek olmalı (sonuç ezberle açıklanamaz).
6. Kayıtlı tahminlerden makale tabloları (`python -m analysis.metrikler`): Tablo II, III, V ve IX'daki sınıflandırma
   ölçüleri, Tablo VIII'in teşhis AUC değerleri ve metindeki sınıflandırma değerleri (141 değer).

Bir sonuç dosyası bu koşuda yazılmadıysa (ör. git'ten gelen başvuru sonucu) o denetim ATLANDI sayılır, geçmiş kabul
edilmez. `calistir.py` bunun için koşunun başlangıç anını `--baslangic` ile verir.

Çalıştırma: python -m boruhatti.kontrol
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

import pandas as pd

import config
from boruhatti.ortak import CIKTI, json_oku

TOLERANS_DOSYASI = config.RESULTS / "beklenen" / "toleranslar.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--baslangic", type=float, default=None,
                    help="bu andan (Unix saniyesi) önce yazılmış sonuç dosyaları sayılmaz; calistir.py verir")
    args = ap.parse_args()
    tol = json_oku(TOLERANS_DOSYASI)
    satirlar = []

    def ekle(ad, sonuc, ayrinti):
        satirlar.append({"denetim": ad, "sonuc": "TUTTU" if sonuc else "TUTMADI", "ayrinti": ayrinti})

    def taze(f, ad) -> bool:
        """Sonuç dosyası bu koşuda mı yazıldı? Git'ten gelen başvuru sonuçları denetimi geçmiş sayılmaz."""
        if not f.exists():
            return False
        if args.baslangic is not None and f.stat().st_mtime < args.baslangic:
            tarih = dt.datetime.fromtimestamp(f.stat().st_mtime).strftime("%d.%m.%Y %H:%M")
            satirlar.append({"denetim": ad, "sonuc": "ATLANDI", "ayrinti": f"bu koşuda üretilmedi ({f.name}, {tarih})"})
            return False
        return True

    f = CIKTI / "degerlendirme_makale_modelleri.csv"
    if taze(f, "Makaledeki şifreli modeller (Tablo V, VIII)"):
        df = pd.read_csv(f)
        ekle("Makaledeki şifreli modeller (Tablo V, VIII)", bool(df.tuttu.all()),
             f"{int(df.tuttu.sum())}/{len(df)} değer makaledeki 5 tohum ortalamasıyla aynı")
    f = CIKTI / "degerlendirme_sifreli.json"
    if taze(f, "CKKS şifreli çıkarım"):
        for r in json_oku(f):
            ok = r["en_buyuk_logit_farki"] <= tol["sifreli_logit_farki"] and r["tahmin_uyumu"] == 1.0
            ekle(f"CKKS şifreli çıkarım: {r['veri']}, {r['model']} {r['temsil']}", ok,
                 f"{r['goruntu']} görüntü, en büyük logit farkı {r['en_buyuk_logit_farki']:.1e}, tahmin uyumu "
                 f"%{100 * r['tahmin_uyumu']:.0f}, görüntü başına {r['goruntu_basina_s']:.2f} s")
    f = CIKTI / "degerlendirme_saldirgan.json"
    if taze(f, "Saldırgan modelleri"):
        for r in json_oku(f):
            if "makaledeki_5_tohum_auc" not in r:
                continue
            fark = r["auc"] - r["makaledeki_5_tohum_auc"]
            ekle(f"Saldırgan: {r['model']}", abs(fark) <= tol["saldirgan_auc"],
                 f"AUC {r['auc']:.4f}; makalede 5 tohum {r['makaledeki_5_tohum_auc']:.4f} ± "
                 f"{r['makaledeki_5_tohum_std']:.4f}; fark {fark:+.4f} (tolerans ±{tol['saldirgan_auc']})")
    f = CIKTI / "degerlendirme_foveahe.json"
    if taze(f, "FoveaHE yeniden eğitim"):
        pay = tol["foveahe_auc_pay"]
        for r in json_oku(f):
            alt, ust = r["makaledeki_5_tohum_auc_aralik"]
            agirlik = ("ağırlıklar bit bit aynı" if r["en_buyuk_agirlik_farki"] == 0
                       else f"en büyük ağırlık farkı {r['en_buyuk_agirlik_farki']:.1e}")
            ekle(f"FoveaHE yeniden eğitim: {r['model']}", alt - pay <= r["auc"] <= ust + pay,
                 f"AUC {r['auc']:.4f}; makaledeki aynı tohum ve bölmenin modeli {r['makaledeki_model_auc']:.4f} "
                 f"(fark {r['fark']:+.4f}, tahmin uyumu %{100 * r['tahmin_uyumu']:.1f}, {agirlik}); makaledeki 5 tohum "
                 f"aynı bölmede {alt:.4f}–{ust:.4f} (pay ±{pay})")
    f = config.TABLES / "ezber_denetimi.csv"
    if taze(f, "Ezber denetimi (etiket karıştırma)"):
        for r in pd.read_csv(f).itertuples(index=False):
            ekle(f"Ezber denetimi (etiket karıştırma): {r.veri}, {r.model}",
                 r.gercek_etiketle_test_auc > r.karisik_test_auc_en_yuksek,
                 f"gerçek etiketle test AUC {r.gercek_etiketle_test_auc:.4f}; karıştırılmış etiketle {r.karistirma_sayisi} "
                 f"koşu: ort. {r.karisik_test_auc_ort:.4f}, en yüksek {r.karisik_test_auc_en_yuksek:.4f}")
    f = config.TABLES / "makale_eslesme.csv"
    if taze(f, "Kayıtlı tahminlerden makale tabloları (Tablo II, III, V, VIII, IX ve metin)"):
        df = pd.read_csv(f)
        ekle("Kayıtlı tahminlerden makale tabloları (Tablo II, III, V, VIII, IX ve metin)", bool(df.tuttu.all()),
             f"{int(df.tuttu.sum())}/{len(df)} değer")
    df = pd.DataFrame(satirlar, columns=["denetim", "sonuc", "ayrinti"])
    with pd.option_context("display.width", 250, "display.max_colwidth", 140):
        print(df.to_string(index=False))
    if not (df.sonuc != "ATLANDI").any():
        sys.exit("Denetlenecek çıktı yok: önce `python -m boruhatti.degerlendir ...` çalıştırın.")
    tutmayan, atlanan = int((df.sonuc == "TUTMADI").sum()), int((df.sonuc == "ATLANDI").sum())
    ek = f" ({atlanan} denetim bu koşuda üretilmediği için atlandı)" if atlanan else ""
    print("\nSONUÇ:", f"bütün denetimler TUTTU{ek}." if tutmayan == 0 else f"{tutmayan} denetim TUTMADI{ek}.")
    if args.baslangic is None:
        print("Not: dosya tarihlerine bakılmadı; `calistir.py` yalnız o koşuda üretilen sonuçları sayar.")
    df.to_csv(CIKTI / "kontrol.csv", index=False, encoding="utf-8-sig")
    sys.exit(1 if tutmayan else 0)


if __name__ == "__main__":
    main()
