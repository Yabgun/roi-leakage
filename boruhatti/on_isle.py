"""Ön işleme: indirilen ham veriden makaledeki deneylerin girdilerini üretir (danışman maddesi 7).

Adımlar (ayrıntı ve gerekçeler: `docs/ON_ISLEME.md`). Çıktısı zaten olan adım atlanır; `--zorla` ile yeniden yapılır.
1. manifestolar: Kaggle CXR, COVID-QU-Ex ve beyin MR (.mat → PNG + tümör maskesi, resmi 5 kat); Kaggle ile COVID-QU-Ex
   arasında kopya taraması (`experiments/prepare_data.py`)
2. kaggle_maskeleri: Kaggle görüntülerine U-Net akciğer maskesi. Hazır U-Net ağırlığı (`modeller/lung_unet.pt` ya da
   `results/checkpoints/lung_unet.pt`) varsa onunla, yoksa U-Net eğitilir (`experiments/lung_segmenter.py`).
3. beyin_ham: beyin MR ham yoğunluk önbelleği (görünür piksel normalizasyonu için; `experiments/brain_visible_norm.py`)
4. onbellek_224: 224×224 görüntü/maske önbellekleri (`experiments/attack_context.py`)
5. fovea: FoveaHE temsil katmanları (makaledeki yapılandırmalar; `foveahe/data.py`)
6. ozet: şifreli özet için ResNet-18 özetleri (`experiments/fovea_baselines.py`)

Çalıştırma: python -m boruhatti.on_isle [--adim ...] [--zorla]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import config
from boruhatti.ortak import MODELLER, aygit

ADIMLAR = ["manifestolar", "kaggle_maskeleri", "beyin_ham", "onbellek_224", "fovea", "ozet"]
FOVEA_ANAHTAR = {"brain": ["f32", "g16", "f64", "g32", "g64", "g512"], "covidqu": ["f32", "g16", "f64", "g32", "g64", "g256"]}


def manifestolar(zorla: bool) -> None:
    from experiments import prepare_data as pd_
    gereken = [config.DATA_PROC / "kaggle_cxr_manifest.csv", config.DATA_PROC / "covidqu_manifest.csv",
               config.DATA_PROC / "brain" / "meta.csv", config.DATA_PROC / "cxr_capraz_tekrarlar.csv"]
    if all(p.exists() for p in gereken) and not zorla:
        print("[manifestolar] zaten var, atlandı")
        return
    ozet_yolu = config.DATA_PROC / "veri_ozeti.json"
    ozet = json.loads(ozet_yolu.read_text(encoding="utf-8")) if ozet_yolu.exists() else {}
    for adim in (pd_.prepare_kaggle, pd_.prepare_covidqu, pd_.prepare_brain, pd_.find_duplicates):
        ozet.update(adim())
        ozet_yolu.write_text(json.dumps(ozet, ensure_ascii=False, indent=2), encoding="utf-8")


def kaggle_maskeleri(zorla: bool) -> None:
    import pandas as pd
    import torch
    from experiments import lung_segmenter as ls
    man = pd.read_csv(config.DATA_PROC / "kaggle_cxr_manifest.csv")
    if "lung_mask_path" in man and man.lung_mask_path.map(lambda p: Path(str(p)).exists()).all() and not zorla:
        print("[kaggle_maskeleri] zaten var, atlandı")
        return
    dev = aygit()
    yol = next((p for p in (MODELLER / "lung_unet.pt", MODELLER / "makale" / "lung_unet.pt", ls.AGIRLIK) if p.exists()), None)
    if yol is None:
        print("[kaggle_maskeleri] U-Net ağırlığı yok; COVID-QU-Ex Train ile eğitiliyor (~10 dk ekran kartında)")
        model, val_dice = ls.unet_egit(dev=dev)
        MODELLER.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODELLER / "lung_unet.pt")
        print(f"[kaggle_maskeleri] U-Net Val Dice = {val_dice:.4f} (makale: 0.978)")
    else:
        print(f"[kaggle_maskeleri] U-Net ağırlığı: {yol}")
        model = ls.unet_yukle(yol, dev)
    print("[kaggle_maskeleri]", ls.kaggle_maskeleri(model, dev))


def beyin_ham(zorla: bool) -> None:
    from experiments.brain_visible_norm import RAW, build
    if RAW.exists() and not zorla:
        print("[beyin_ham] zaten var, atlandı")
        return
    build(force=True)


def onbellek_224(zorla: bool) -> None:
    import pandas as pd
    from experiments.attack_context import build_cache
    meta = pd.read_csv(config.DATA_PROC / "brain" / "meta.csv")
    b = config.DATA_PROC / "brain"
    build_cache("brain", [b / p for p in meta.img_path], [b / p for p in meta.mask_path])
    man = pd.read_csv(config.DATA_PROC / "covidqu_manifest.csv")
    build_cache("covidqu", man.img_path, man.lung_mask_path)
    kg = pd.read_csv(config.DATA_PROC / "kaggle_cxr_manifest.csv")
    if "lung_mask_path" in kg:
        build_cache("kaggle", kg.img_path, kg.lung_mask_path)
    print("[onbellek_224] hazır")


def fovea(zorla: bool) -> None:
    from foveahe.data import load_dataset, load_layers
    for ad, anahtarlar in FOVEA_ANAHTAR.items():
        load_layers(load_dataset(ad), anahtarlar)
    print("[fovea] hazır")


def ozet(zorla: bool) -> None:
    from experiments.fovea_baselines import embeddings
    from foveahe.data import load_dataset
    for ad in ("brain", "covidqu"):
        embeddings(load_dataset(ad), device=aygit())
    print("[ozet] hazır")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adim", nargs="*", default=ADIMLAR, choices=ADIMLAR)
    ap.add_argument("--zorla", action="store_true")
    args = ap.parse_args()
    for ad in args.adim:
        t0 = time.perf_counter()
        globals()[ad](args.zorla)
        print(f"  ({ad}: {time.perf_counter() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
