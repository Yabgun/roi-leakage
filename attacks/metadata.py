"""Saldırı A: yalnızca ROI meta verisinden (konum, büyüklük, şekil) ve girdi boyutundan teşhis çıkarımı.

Π_ROI'nin sızıntı fonksiyonu L = (W, X_non-ROI, f(mtd)) içindeki f(mtd), ROI koordinatları, boyutu ve şeklidir;
sunucu ayrıca girdinin boyutunu öğrenir. Bu modül bu bilgiden sayısal öznitelikler çıkarır.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

FEATURE_GROUPS = {
    "girdi_boyutu": ["img_h", "img_w", "img_aspect"],
    "konum": ["bbox_r0", "bbox_r1", "bbox_c0", "bbox_c1", "cen_r", "cen_c"],
    "buyukluk": ["area_frac", "bbox_h", "bbox_w", "major_axis", "minor_axis", "comp1_frac", "comp2_frac"],
    "sekil": ["bbox_aspect", "extent", "perimeter_norm", "circularity", "eccentricity", "orientation",
              "n_components"],
}
ALL_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]


def mask_features(mask: np.ndarray, img_h: int, img_w: int) -> dict:
    """ROI maskesinden meta veri öznitelikleri. Koordinatlar maske boyutuna göre [0,1] aralığına normalize."""
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    area = int(mask.sum())
    f = {k: np.nan for k in ALL_FEATURES}
    f.update(img_h=float(img_h), img_w=float(img_w), img_aspect=float(img_w) / float(img_h),
             area_frac=area / (H * W), n_components=0.0, comp1_frac=0.0, comp2_frac=0.0)
    if area == 0:
        return f
    rows, cols = np.nonzero(mask)
    r0, r1 = rows.min() / H, (rows.max() + 1) / H
    c0, c1 = cols.min() / W, (cols.max() + 1) / W
    f.update(bbox_r0=r0, bbox_r1=r1, bbox_c0=c0, bbox_c1=c1, bbox_h=r1 - r0, bbox_w=c1 - c0,
             bbox_aspect=(c1 - c0) / max(r1 - r0, 1e-9), extent=area / max((r1 - r0) * H * (c1 - c0) * W, 1e-9),
             cen_r=rows.mean() / H, cen_c=cols.mean() / W)
    perim = int((mask & ~ndimage.binary_erosion(mask)).sum())
    f["perimeter_norm"] = perim / np.sqrt(H * W)
    f["circularity"] = 4 * np.pi * area / max(perim, 1) ** 2
    if area > 2:
        cov = np.cov(np.stack([rows / H, cols / W]))
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
        f["major_axis"] = float(np.sqrt(max(ev[0], 0)))
        f["minor_axis"] = float(np.sqrt(max(ev[1], 0)))
        f["eccentricity"] = float(np.sqrt(max(1 - ev[1] / ev[0], 0))) if ev[0] > 0 else 0.0
        f["orientation"] = float(0.5 * np.arctan2(2 * cov[0, 1], cov[0, 0] - cov[1, 1]))
    lab, n = ndimage.label(mask)
    f["n_components"] = float(n)
    if n:
        sizes = np.sort(np.asarray(ndimage.sum(mask, lab, range(1, n + 1))))[::-1] / (H * W)
        f["comp1_frac"] = float(sizes[0])
        f["comp2_frac"] = float(sizes[1]) if n > 1 else 0.0
    return f
