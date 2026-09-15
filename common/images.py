"""Görüntü ve maske yükleme yardımcıları."""
from __future__ import annotations

import numpy as np
from PIL import Image


def load_gray(path, size: int | None = None) -> np.ndarray:
    """Gri tonlamalı görüntüyü [0,1] float32 olarak döndürür; size verilirse size×size'a indirir."""
    with Image.open(path) as im:
        if size is not None:
            try:
                im.draft("L", (size * 2, size * 2))
            except Exception:
                pass
        g = im.convert("L")
        if size is not None:
            g = g.resize((size, size), Image.BILINEAR)
        return np.asarray(g, dtype=np.float32) / 255.0


def load_mask(path, size: int | None = None) -> np.ndarray:
    """İkili maskeyi bool dizi olarak döndürür (en yakın komşu ile yeniden boyutlandırma)."""
    with Image.open(path) as im:
        g = im.convert("L")
        if size is not None:
            g = g.resize((size, size), Image.NEAREST)
        return np.asarray(g) > 127


def square_box_mask(size: int, rho: float) -> np.ndarray:
    """Görüntünün merkezinde alanı yaklaşık rho olan kare ROI maskesi."""
    side = int(round(np.sqrt(rho) * size))
    side = max(1, min(size, side))
    m = np.zeros((size, size), dtype=bool)
    r0 = (size - side) // 2
    m[r0:r0 + side, r0:r0 + side] = True
    return m
