"""Makaledeki model ağırlıklarını ve kayıtlı tahminleri Hugging Face'ten indirir (danışman maddeleri 1 ve 6).

Depo: https://huggingface.co/Btutumlu/roi-leakage (sürüm `REVIZYON` ile sabit). Hesap gerekmez.
İndirilenler projedeki yerlerine yazılır:
- `modeller/makale/`: Tablo V ve VIII'deki şifreli modellerin CKKS'e hazır ağırlıkları (5 tohum) ve akciğer U-Net'i
- `modeller/saldirgan_<veri>_<görüş>_tez_s0/`: bağlam saldırganı (ResNet-18) ağırlıkları, ayar ve sonuç dosyaları
- `results/preds/`: makaledeki koşuların kayıtlı tahminleri (makale tabloları bunlardan yeniden hesaplanır)
Her dosya `modeller/hf_manifest.json` içindeki boyut ve SHA-256 özetiyle doğrulanır. Hepsi zaten yerindeyse ve
doğrulanırsa indirme yapılmaz. Toplam 1 060 dosya, ~3.1 GB.

Çalıştırma: python -m boruhatti.modelleri_indir [--yalniz-dogrula]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time

import config
from boruhatti.ortak import MODELLER

HF_DEPO = "Btutumlu/roi-leakage"
REVIZYON = "d3c0f6d649a30cfe4d57b130e2941e2bffeb96c6"  # 1 Ekim 2026 yüklemesi; depo değişse de aynı dosyalar iner
DESENLER = ["modeller/*", "results/preds/*"]  # fnmatch: * alt klasörleri de kapsar
MANIFEST = MODELLER / "hf_manifest.json"


def sha256(yol) -> str:
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for parca in iter(lambda: f.read(1 << 20), b""):
            h.update(parca)
    return h.hexdigest()


def dogrula() -> tuple[int, list[str]]:
    """(manifestodaki dosya sayısı, eksik ya da bozuk dosyalar)."""
    if not MANIFEST.exists():
        return 0, [str(MANIFEST.relative_to(config.ROOT))]
    dosyalar = json.loads(MANIFEST.read_text(encoding="utf-8"))["dosyalar"]
    hatali = []
    for d in dosyalar:
        yol = config.ROOT / d["yol"]
        if not yol.exists() or yol.stat().st_size != d["bayt"] or sha256(yol) != d["sha256"]:
            hatali.append(d["yol"])
    return len(dosyalar), hatali


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--yalniz-dogrula", action="store_true", help="indirmeden yalnız doğrula")
    args = ap.parse_args()
    n, hatali = dogrula()
    if hatali and not args.yalniz_dogrula:
        from huggingface_hub import snapshot_download
        print(f"[modelleri_indir] {HF_DEPO} ({REVIZYON}) → {config.ROOT}", flush=True)
        t0 = time.perf_counter()
        snapshot_download(repo_id=HF_DEPO, revision=REVIZYON, local_dir=config.ROOT, allow_patterns=DESENLER)
        print(f"  indirme bitti ({time.perf_counter() - t0:.0f} s)", flush=True)
        n, hatali = dogrula()
    if hatali:
        print(f"[modelleri_indir] {len(hatali)} dosya eksik ya da bozuk; ilk beşi: {hatali[:5]}")
        sys.exit(1)
    print(f"[modelleri_indir] {n} dosyanın hepsi yerinde ve doğrulandı (boyut ve SHA-256).")


if __name__ == "__main__":
    main()
