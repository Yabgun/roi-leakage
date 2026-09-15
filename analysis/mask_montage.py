"""Kaggle CXR için üretilen akciğer maskelerinin görsel kontrolü: rastgele örneklerde maske sınırı çizimi.

Çalıştırma: .venv\\Scripts\\python -m analysis.mask_montage
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage

import config
from common.images import load_gray, load_mask


def main(per_class: int = 8, size: int = 160):
    kag = pd.read_csv(config.DATA_PROC / "kaggle_cxr_manifest.csv")
    rng = np.random.default_rng(3)
    rows = []
    for label in ["NORMAL", "PNEUMONIA", "COVID19"]:
        idx = rng.choice(np.flatnonzero((kag.label == label).to_numpy()), per_class, replace=False)
        tiles = []
        for i in idx:
            g = (load_gray(kag.img_path[i], size) * 255).astype(np.uint8)
            m = load_mask(kag.lung_mask_path[i], size)
            edge = m & ~ndimage.binary_erosion(m, iterations=2)
            rgb = np.stack([g, g, g], axis=-1)
            rgb[edge] = [0, 255, 255]
            tile = Image.fromarray(rgb)
            ImageDraw.Draw(tile).text((3, 2), f"{label[:4]} {kag.lung_frac[i]:.2f}", fill=(255, 255, 0))
            tiles.append(np.asarray(tile))
        rows.append(np.concatenate(tiles, axis=1))
    out = config.FIGURES / "kaggle_akciger_maskeleri_kontrol.png"
    Image.fromarray(np.concatenate(rows, axis=0)).save(out)
    print(out)


if __name__ == "__main__":
    main()
