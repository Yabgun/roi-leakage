"""Akciğer segmentasyonu için küçük U-Net. COVID-QU-Ex akciğer maskeleriyle eğitilir; maskesi olmayan
veri setlerine (Kaggle CXR) akciğer ROI'si üretmek için kullanılır."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, widths=(16, 32, 64, 128, 256)):
        super().__init__()
        self.down = nn.ModuleList()
        cin = 1
        for w in widths:
            self.down.append(_block(cin, w))
            cin = w
        self.up = nn.ModuleList()
        self.fuse = nn.ModuleList()
        for w_skip, w_in in zip(reversed(widths[:-1]), reversed(widths[1:])):
            self.up.append(nn.ConvTranspose2d(w_in, w_skip, 2, stride=2))
            self.fuse.append(_block(2 * w_skip, w_skip))
        self.head = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.down):
            x = blk(x)
            if i < len(self.down) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for up, fuse, s in zip(self.up, self.fuse, reversed(skips)):
            x = fuse(torch.cat([up(x), s], dim=1))
        return self.head(x)


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + eps) / (p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps)).mean()


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    inter = np.logical_and(pred, target).sum()
    return float(2 * inter / (pred.sum() + target.sum() + 1e-9))


def postprocess(mask: np.ndarray) -> np.ndarray:
    """En büyük iki bağlı bileşeni (iki akciğer) tut, delikleri doldur."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    keep = np.argsort(sizes)[::-1][:2] + 1
    out = np.isin(lab, keep)
    return ndimage.binary_fill_holes(out)
