"""Tek komutla yeniden üretim (danışman maddeleri 6 ve 7).

Kademeler:
- 0: veriyi indir ve doğrula → ön işle → makaledeki modelleri (indirilen ağırlıklar) test et → birkaç görüntüde gerçek
     CKKS şifreli çıkarım → makale tablolarını kayıtlı tahminlerden yeniden hesapla → denetim. Eğitim yoktur; süresinin
     çoğu indirme ve ön işlemedir.
Denetim (`boruhatti.kontrol`) yalnız bu koşuda üretilen sonuçları sayar; git'ten gelen başvuru sonuçları sayılmaz.
- 1: Kademe 0 + makalenin ana modellerini bu bilgisayarda yeniden eğit (saldırgan: makale protokolü ve izleme
     koşuları; FoveaHE modelleri: makaledeki ilk bölmede, eğitim eğrileriyle; beyin MR Model D ayrıca 5 katın hepsiyle)
     → ezber denetimi (etiket karıştırma) → yeni modelleri test et ve makaledeki modellerle karşılaştır → denetim.
     RTX 2070'te eğitimler ~27 dk, ezber denetimi ~19 dk sürdü.
- 2: makaledeki bütün deneyler (`experiments/`); çok uzun, isteğe bağlı. Komutlar README'de.

Örnek: python calistir.py --kademe 1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

EGITIM = [
    ["--model", "saldirgan", "--veri", "beyin", "--gorus", "baglam", "--protokol", "tez"],
    ["--model", "saldirgan", "--veri", "beyin", "--gorus", "tam", "--protokol", "tez"],
    ["--model", "saldirgan", "--veri", "covidqu", "--gorus", "baglam", "--protokol", "tez"],
    ["--model", "saldirgan", "--veri", "covidqu", "--gorus", "tam", "--protokol", "tez"],
    ["--model", "saldirgan", "--veri", "beyin", "--gorus", "baglam", "--protokol", "izleme"],
    ["--model", "saldirgan", "--veri", "covidqu", "--gorus", "baglam", "--protokol", "izleme"],
    ["--model", "foveahe_D", "--veri", "beyin", "--temsil", "F32_G16", "--protokol", "izleme"],
    ["--model", "foveahe_D", "--veri", "beyin", "--temsil", "F32_G16", "--protokol", "tez"],  # 5 katın hepsi (~30 s)
    ["--model", "foveahe_C", "--veri", "beyin", "--temsil", "F32_G16", "--protokol", "izleme"],
    ["--model", "foveahe_D2", "--veri", "covidqu", "--temsil", "F32_G16", "--protokol", "izleme"],
    ["--model", "foveahe_D", "--veri", "beyin", "--temsil", "U512", "--protokol", "izleme"],
    ["--model", "foveahe_D2", "--veri", "covidqu", "--temsil", "U256", "--protokol", "izleme"],
]
SALDIRGAN_MODELLERI = [f"modeller/saldirgan_{v}_{g}_tez_s0" for v in ("beyin", "covidqu") for g in ("baglam", "tam")]
# Kademe 1'de yeniden eğitilen şifreli modeller (izleme koşusu = makaledeki ilk bölme); makaledeki modelle karşılaştırılır
FOVEAHE_MODELLERI = [f"modeller/foveahe_{a[1].split('_')[1]}_{a[3]}_{a[5]}_{a[7]}_s0" for a in EGITIM if a[1].startswith("foveahe")]


def calistir(modul: str, *arg: str, zorunlu: bool = True) -> int:
    komut = [sys.executable, "-m", modul, *arg]
    print(f"\n=== {' '.join(komut[2:])} ===", flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(komut)
    print(f"--- {modul}: çıkış {r.returncode}, {time.perf_counter() - t0:.0f} s", flush=True)
    if r.returncode and zorunlu:
        sys.exit(f"{modul} başarısız oldu; yukarıdaki iletiye bakın.")
    return r.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kademe", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--wandb-kapali", action="store_true", help="eğitim eğrilerini yalnız yerelde kaydet")
    args = ap.parse_args()
    if args.kademe == 2:
        sys.exit("Kademe 2 (bütün deneyler) README'deki komut listesiyle elle çalıştırılır; ~1–2 gün ekran kartı ister.")
    t0, baslangic = time.perf_counter(), time.time()  # kontrol yalnız bu andan sonra yazılan sonuçları sayar
    calistir("boruhatti.ortam_kontrol", zorunlu=False)
    calistir("boruhatti.veri_indir")
    calistir("boruhatti.on_isle")
    calistir("boruhatti.kopya_kontrol", "--ic", zorunlu=False)
    if args.kademe == 1:
        for arg in EGITIM:
            calistir("boruhatti.egit", *arg, *(["--wandb-kapali"] if args.wandb_kapali else []))
        calistir("boruhatti.ezber_denetimi")  # etiket karıştırma testi (~19 dk)
    calistir("boruhatti.degerlendir", "--makale-modelleri", "--sifreli", "9", "--saldirgan", *SALDIRGAN_MODELLERI,
             *(["--foveahe", *FOVEAHE_MODELLERI] if args.kademe == 1 else []))
    calistir("analysis.metrikler", zorunlu=False)  # makale tablolarını kayıtlı tahminlerden yeniden hesaplar
    kod = calistir("boruhatti.kontrol", "--baslangic", repr(baslangic), zorunlu=False)
    print(f"\nKademe {args.kademe} bitti ({(time.perf_counter() - t0) / 60:.0f} dk).")
    sys.exit(kod)


if __name__ == "__main__":
    main()
