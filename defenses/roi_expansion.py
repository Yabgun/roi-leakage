"""Savunmalar: sabit (kanonik) ROI ve sızıntı-güdümlü ROI genişletme.

Gizli bölge her zaman görüntünün kendi ROI'sini de içerir (ROI içeriği şifreli kalmalı). Kanonik ROI tüm
görüntülerde aynı olduğu için ROI meta verisini sızdırmaz; genişletme ek yamalarla bağlam sızıntısını azaltır.
"""
from __future__ import annotations

import numpy as np
import torch

from attacks.context_cnn import CpuCache
from common.evaluation import auc_score


def mask_frequency(masks: np.ndarray, chunk: int = 2048) -> np.ndarray:
    counts = np.zeros(masks.shape[1:], dtype=np.int64)
    for i in range(0, len(masks), chunk):
        counts += masks[i:i + chunk].sum(axis=0)
    return counts / len(masks)


def canonical_roi(masks: np.ndarray, coverage: float = 0.99) -> np.ndarray:
    """Tüm maske kütlesinin `coverage` kadarını kapsayan, en sık ROI olan piksellerden oluşan sabit bölge."""
    freq = mask_frequency(masks)
    order = np.argsort(freq.ravel())[::-1]
    mass = np.cumsum(freq.ravel()[order])
    k = int(np.searchsorted(mass, coverage * mass[-1])) + 1
    canon = np.zeros(freq.size, dtype=bool)
    canon[order[:k]] = True
    return canon.reshape(freq.shape)


def dilate_np(region: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return region
    t = torch.from_numpy(region).float()[None, None]
    return (torch.nn.functional.max_pool2d(t, 2 * px + 1, stride=1, padding=px) > 0)[0, 0].numpy()


def patch_grid(size: int, grid: int) -> np.ndarray:
    step = size // grid
    p = np.zeros((grid * grid, size, size), dtype=bool)
    for r in range(grid):
        for c in range(grid):
            r1 = size if r == grid - 1 else (r + 1) * step
            c1 = size if c == grid - 1 else (c + 1) * step
            p[r * grid + c, r * step:r1, c * step:c1] = True
    return p


def const_hidden_fn(hidden: np.ndarray, device: str = "cuda"):
    h = torch.from_numpy(hidden).to(device)[None, None]
    return lambda m: h.expand_as(m) | m


def hidden_fraction(hidden: np.ndarray, masks: np.ndarray, idx: np.ndarray) -> float:
    """Görüntü başına gerçek gizli oran: sabit bölge ∪ görüntünün kendi ROI'si."""
    sub = masks[idx]
    return float(np.mean([(hidden | m).mean() for m in sub]))


@torch.no_grad()
def predict(model, cache: CpuCache, idx: np.ndarray, hidden_fn, batch: int = 256) -> np.ndarray:
    """fp32 ve karışık sırada tahmin: fp16 gürültüsü + sınıfa göre sıralı indeks kaynaklı sahte AUC'yi önler."""
    model.eval()
    idx = np.asarray(idx)
    order = np.random.default_rng(12345).permutation(len(idx))
    out = np.zeros((len(idx), model.fc.out_features), dtype=np.float64)
    for i in range(0, len(idx), batch):
        part = order[i:i + batch]
        x, _ = cache.batch(idx[part], "tam", train=False, extra_hidden_fn=hidden_fn)
        out[part] = torch.softmax(model(x).double(), 1).cpu().numpy()
    return out


def occlusion_drops(model, cache: CpuCache, idx: np.ndarray, y: np.ndarray, hidden: np.ndarray,
                    patches: np.ndarray, candidates) -> tuple[float, dict]:
    """Her aday yama ek olarak gizlendiğinde saldırganın AUC düşüşü (model sabit)."""
    base = auc_score(y[idx], predict(model, cache, idx, const_hidden_fn(hidden)))
    drops = {}
    for p in candidates:
        drops[p] = base - auc_score(y[idx], predict(model, cache, idx, const_hidden_fn(hidden | patches[p])))
    return base, drops
