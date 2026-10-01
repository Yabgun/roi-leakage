"""Veri kümelerini özgün kaynaklarından indirir ve çalışmada kullanılan veriyle aynı olduğunu doğrular (madde 7).

Kaynaklar `boruhatti/veri_kaynaklari.json` dosyasındadır:
- COVID-QU-Ex ve Kaggle CXR: Kaggle, sürüm numarasıyla sabitlenmiş (kagglehub). Herkese açık kümeler olduğundan
  Kaggle hesabı ve anahtarı gerekmez.
- Beyin tümörü MR: figshare 1512427, sürüm 8. Her dosya figshare'in verdiği MD5 ile denetlenir. Kaggle'daki PNG
  kopyalarında hasta kimliği olmadığı için özgün .mat dosyaları kullanılır.

İndirme bitince her kümenin parmak izi (dosya sayısı, toplam bayt, dosya yolu + SHA-256 listesinin özeti) çalışmada
kullanılan verinin parmak iziyle karşılaştırılır; aynıysa veri bayt bayt aynıdır. Veri hiçbir yere yüklenmez ve
depoda dağıtılmaz; lisanslar kaynak sayfalarındadır (COVID-QU-Ex CC BY-SA 4.0, figshare CC BY 4.0).

Windows notu: COVID-QU-Ex'in klasör ve dosya adları uzundur (en uzun göreli yol 130 karakter). Proje kısa bir
klasörde olmalıdır (ör. C:\\roi-leakage); betik yol uzunluğunu indirmeden önce denetler.

Çalıştırma:
  python -m boruhatti.veri_indir                      # üç küme: indir + doğrula (~3.7 GB indirme)
  python -m boruhatti.veri_indir --kume beyin         # yalnız bir küme
  python -m boruhatti.veri_indir --yalniz-dogrula     # indirmeden, diskteki veriyi doğrula
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import config
from boruhatti.ortak import json_oku, json_yaz

KAYNAKLAR = Path(__file__).with_name("veri_kaynaklari.json")
ISARET = ".extracted"  # experiments.prepare_data bu işaret varsa zip açmayı atlar


def hedef(kume: dict) -> Path:
    return config.ROOT / kume["hedef"]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def parmak_izi(kok: Path, altlar: list[str] | None = None) -> dict:
    """Dosya sayısı, toplam bayt ve "göreli_yol<TAB>bayt<TAB>sha256" satırlarının (yola göre sıralı) SHA-256 özeti."""
    dosyalar = [p for alt in (altlar or [""]) for p in (kok / alt).rglob("*") if p.is_file() and p.name != ISARET]
    dosyalar.sort(key=lambda p: p.relative_to(kok).as_posix())
    with ThreadPoolExecutor(4) as ex:
        ozetler = list(ex.map(_sha256, dosyalar))
    satirlar = [f"{p.relative_to(kok).as_posix()}\t{p.stat().st_size}\t{h}" for p, h in zip(dosyalar, ozetler)]
    return {"dosya_sayisi": len(dosyalar), "toplam_bayt": sum(p.stat().st_size for p in dosyalar),
            "sha256_ozet": hashlib.sha256("\n".join(satirlar).encode("utf-8")).hexdigest()}


def yol_denetimi(gec: bool) -> None:
    uzunluk = len(str(config.DATA_RAW.resolve())) + len("/covid_qu_ex/extracted/") + 130
    if uzunluk > 255 and not gec:
        raise SystemExit(f"En uzun veri yolu ~{uzunluk} karakter olacak; Windows sınırı 260. Projeyi kısa bir klasöre "
                         "taşıyın (ör. C:\\roi-leakage) ya da Windows'ta uzun yol desteğini açıp --yol-denetimini-gec "
                         "ile çalıştırın.")


def indir_kaggle(ad: str, k: dict) -> None:
    import kagglehub
    kok = hedef(k)
    tanim = f"{k['kaynak']}/versions/{k['surum']}"
    if ad == "covidqu":
        if (kok / ISARET).exists():
            print(f"[{ad}] zaten indirilmiş: {kok}")
            return
        kok.mkdir(parents=True, exist_ok=True)
        print(f"[{ad}] Kaggle {tanim} indiriliyor (~{k['indirme_mb']} MB) → {kok}")
        kagglehub.dataset_download(tanim, output_dir=str(kok))
        (kok / ISARET).touch()
        return
    # Kaggle CXR: kümede Data/train ve Data/test; çalışma yalnız bu iki klasörü kullanır
    if all((kok / alt).exists() for alt in k["kullanilan"]):
        print(f"[{ad}] zaten indirilmiş: {kok}")
        return
    gecici = kok / "_kaggle_indirilen"
    gecici.mkdir(parents=True, exist_ok=True)
    print(f"[{ad}] Kaggle {tanim} indiriliyor (~{k['indirme_mb']} MB) → {gecici}")
    kagglehub.dataset_download(tanim, output_dir=str(gecici))
    for alt in k["kullanilan"]:
        kaynak = gecici / k["alt_klasor"] / alt
        if not kaynak.exists():
            raise SystemExit(f"[{ad}] beklenen klasör yok: {kaynak}")
        shutil.move(str(kaynak), str(kok / alt))
    print(f"[{ad}] {', '.join(k['kullanilan'])} klasörleri {kok} altına taşındı")


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def indir_figshare(ad: str, k: dict) -> None:
    kok = hedef(k)
    kok.mkdir(parents=True, exist_ok=True)
    for d in k["dosyalar"]:
        yol = kok / d["ad"]
        if yol.exists() and yol.stat().st_size == d["bayt"] and _md5(yol) == d["md5"]:
            print(f"[{ad}] {d['ad']}: zaten var, MD5 aynı")
            continue
        parca = yol.with_name(yol.name + ".part")
        print(f"[{ad}] {d['ad']} indiriliyor ({d['bayt'] / 1e6:.1f} MB)")
        t0 = time.perf_counter()
        istek = urllib.request.Request(d["url"], headers={"User-Agent": "roi-leakage-veri-indir"})
        with urllib.request.urlopen(istek, timeout=120) as r, open(parca, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        if _md5(parca) != d["md5"]:
            raise SystemExit(f"[{ad}] {d['ad']}: MD5 tutmadı; indirmeyi yeniden deneyin")
        parca.replace(yol)
        print(f"[{ad}] {d['ad']}: tamam, MD5 aynı ({time.perf_counter() - t0:.0f} s)")


def dogrula(ad: str, k: dict) -> bool:
    kok = hedef(k)
    if k["tur"] == "figshare":
        tamam = all((kok / d["ad"]).exists() and _md5(kok / d["ad"]) == d["md5"] for d in k["dosyalar"])
        print(f"[{ad}] figshare MD5 denetimi: {'AYNI' if tamam else 'FARKLI ya da eksik'}")
        return tamam
    beklenen = k.get("parmak_izi")
    if not beklenen:
        print(f"[{ad}] beklenen parmak izi tanımlı değil")
        return False
    print(f"[{ad}] parmak izi hesaplanıyor ({beklenen['dosya_sayisi']} dosya)...")
    pi = parmak_izi(kok, k.get("kullanilan"))
    tamam = pi == beklenen
    print(f"[{ad}] {pi['dosya_sayisi']} dosya, {pi['toplam_bayt'] / 1e9:.2f} GB → çalışmadaki veriyle "
          f"{'AYNI' if tamam else 'FARKLI'}")
    if not tamam:
        print(f"    beklenen: {beklenen}\n    bulunan : {pi}")
    return tamam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kume", nargs="*", default=["covidqu", "kaggle_cxr", "beyin"])
    ap.add_argument("--yalniz-dogrula", action="store_true")
    ap.add_argument("--parmak-izi-yaz", action="store_true", help="(bakım) diskteki verinin parmak izini kaynak dosyasına yaz")
    ap.add_argument("--yol-denetimini-gec", action="store_true")
    args = ap.parse_args()
    kaynaklar = json_oku(KAYNAKLAR)
    if args.parmak_izi_yaz:
        for ad in args.kume:
            k = kaynaklar["kumeler"][ad]
            if k["tur"] == "kaggle":
                k["parmak_izi"] = parmak_izi(hedef(k), k.get("kullanilan"))
                print(ad, k["parmak_izi"])
        json_yaz(KAYNAKLAR, kaynaklar)
        return
    if not args.yalniz_dogrula:
        yol_denetimi(args.yol_denetimini_gec)
        bos = shutil.disk_usage(config.ROOT).free / 1024 ** 3
        if bos < 10:
            raise SystemExit(f"Boş disk {bos:.0f} GB; indirme ve ön işleme için en az ~25 GB gerekir.")
        for ad in args.kume:
            k = kaynaklar["kumeler"][ad]
            (indir_figshare if k["tur"] == "figshare" else indir_kaggle)(ad, k)
    sonuc = {ad: dogrula(ad, kaynaklar["kumeler"][ad]) for ad in args.kume}
    print("\nVeri doğrulaması:", "hepsi çalışmadaki veriyle aynı." if all(sonuc.values()) else
          f"farklı ya da eksik küme var: {[a for a, s in sonuc.items() if not s]}")
    if not all(sonuc.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
