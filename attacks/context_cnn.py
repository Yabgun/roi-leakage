"""Saldırı B: sunucunun gördüğü görüntüden (ROI şifreli, dolayısıyla görünmez) teşhis çıkarımı.

Sunucunun görüşü (Π_ROI sızıntı fonksiyonu): ROI dışındaki açık pikseller + ROI'nin konumu/şekli.
Girdi 3 kanal: [görünür görüntü, görünür görüntü, gizli bölge göstergesi]. ImageNet ön eğitimli ResNet-18.
Veriler CPU'da uint8 önbellekte tutulur, artırma ve görüş oluşturma GPU'da yapılır.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

VIEWS = ("tam", "yalniz_roi", "baglam", "baglam_genis10", "baglam_genis20", "baglam_genis40", "kutu")
_MEAN = torch.tensor([0.485, 0.456, 0.5]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.5]).view(1, 3, 1, 1)


def dilate(mask: torch.Tensor, px: int) -> torch.Tensor:
    return F.max_pool2d(mask.float(), kernel_size=2 * px + 1, stride=1, padding=px) > 0


def bbox_mask(mask: torch.Tensor) -> torch.Tensor:
    """Her görüntü için ROI'nin sınırlayıcı kutusu (boş ROI -> boş kutu)."""
    n, _, h, w = mask.shape
    rows = mask.any(dim=3).squeeze(1)
    cols = mask.any(dim=2).squeeze(1)
    ri = torch.arange(h, device=mask.device).expand(n, h)
    ci = torch.arange(w, device=mask.device).expand(n, w)
    r0 = torch.where(rows, ri, torch.full_like(ri, h)).min(1).values
    r1 = torch.where(rows, ri, torch.full_like(ri, -1)).max(1).values
    c0 = torch.where(cols, ci, torch.full_like(ci, w)).min(1).values
    c1 = torch.where(cols, ci, torch.full_like(ci, -1)).max(1).values
    rr = torch.arange(h, device=mask.device).view(1, h, 1)
    cc = torch.arange(w, device=mask.device).view(1, 1, w)
    box = (rr >= r0.view(n, 1, 1)) & (rr <= r1.view(n, 1, 1)) & (cc >= c0.view(n, 1, 1)) & (cc <= c1.view(n, 1, 1))
    return box.unsqueeze(1)


def hidden_region(mask: torch.Tensor, view: str) -> torch.Tensor:
    if view == "tam":
        return torch.zeros_like(mask)
    if view == "yalniz_roi":
        return ~mask
    if view == "baglam":
        return mask
    if view.startswith("baglam_genis"):
        return dilate(mask, int(view.replace("baglam_genis", "")))
    if view == "kutu":
        return bbox_mask(mask)
    raise ValueError(view)


def server_view(img: torch.Tensor, mask: torch.Tensor, view: str, extra_hidden: torch.Tensor | None = None):
    """img: (N,1,H,W) [0,1]; mask: (N,1,H,W) bool. extra_hidden: savunmalarda ek gizlenen bölge."""
    hidden = hidden_region(mask, view)
    if extra_hidden is not None:
        hidden = hidden | extra_hidden
    vis = img.masked_fill(hidden, 0.0)
    x = torch.cat([vis, vis, hidden.float()], dim=1)
    return (x - _MEAN.to(x.device)) / _STD.to(x.device)


def random_affine(img: torch.Tensor, mask: torch.Tensor, max_shift: float = 0.04, scale=(0.93, 1.07)):
    """Aynı küçük kaydırma/ölçekleme görüntüye (bilineer) ve maskeye (en yakın) uygulanır."""
    n = img.shape[0]
    s = torch.empty(n, device=img.device).uniform_(*scale)
    tx = torch.empty(n, device=img.device).uniform_(-max_shift, max_shift) * 2
    ty = torch.empty(n, device=img.device).uniform_(-max_shift, max_shift) * 2
    theta = torch.zeros(n, 2, 3, device=img.device)
    theta[:, 0, 0] = 1 / s
    theta[:, 1, 1] = 1 / s
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, img.shape, align_corners=False)
    img = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    mask = F.grid_sample(mask.float(), grid, mode="nearest", padding_mode="zeros", align_corners=False) > 0.5
    return img, mask


class CpuCache:
    """uint8 görüntüler ve bool maskeler (N,H,W) CPU'da; yığınlar GPU'ya aktarılır."""

    def __init__(self, images: np.ndarray, masks: np.ndarray, labels: np.ndarray, device: str = "cuda",
                 max_batch: int = 256):
        self.img = torch.from_numpy(images)
        self.mask = torch.from_numpy(masks)
        self.y = torch.from_numpy(np.array(labels, dtype=np.int64, copy=True))
        self.device = device
        # Sabitlenmiş (pinned) ara bellekler: CPU->GPU aktarımı eşzamansız ve hızlı olur
        self._img_buf = torch.empty((max_batch, *images.shape[1:]), dtype=self.img.dtype).pin_memory()
        self._mask_buf = torch.empty((max_batch, *masks.shape[1:]), dtype=self.mask.dtype).pin_memory()

    def batch(self, idx: np.ndarray, view: str, train: bool, extra_hidden_fn=None):
        idx_t = torch.from_numpy(np.asarray(idx, dtype=np.int64))
        n = len(idx_t)
        torch.index_select(self.img, 0, idx_t, out=self._img_buf[:n])
        torch.index_select(self.mask, 0, idx_t, out=self._mask_buf[:n])
        x = self._img_buf[:n].to(self.device, non_blocking=True).float().div_(255).unsqueeze(1)
        m = self._mask_buf[:n].to(self.device, non_blocking=True).unsqueeze(1)
        if train:
            x, m = random_affine(x, m)
        extra = extra_hidden_fn(m) if extra_hidden_fn is not None else None
        return server_view(x, m, view, extra), self.y[idx_t].to(self.device, non_blocking=True)


def make_model(n_classes: int) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def train_and_predict(cache: CpuCache, train_idx: np.ndarray, test_idx: np.ndarray, view: str, n_classes: int,
                      epochs: int = 5, batch_size: int = 64, lr: float = 3e-4, seed: int = 0,
                      extra_hidden_fn=None, log=print):
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    rng = np.random.default_rng(seed)
    device = cache.device
    model = make_model(n_classes).to(device).to(memory_format=torch.contiguous_format)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(train_idx) / batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda")
    t0 = time.perf_counter()
    model.train()
    for ep in range(epochs):
        perm = rng.permutation(train_idx)
        total, count = 0.0, 0
        for i in range(0, len(perm), batch_size):
            x, y = cache.batch(perm[i:i + batch_size], view, train=True, extra_hidden_fn=extra_hidden_fn)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(model(x.contiguous(memory_format=torch.contiguous_format)), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += float(loss) * len(y)
            count += len(y)
        log(f"    epoch {ep + 1}/{epochs} kayıp={total / count:.4f} ({time.perf_counter() - t0:.0f} s)")
    # Değerlendirme fp32 ve karışık sırada yapılır: fp16 sayısal gürültüsü sınıfa göre sıralı test indeksleriyle
    # hizalanıp sahte AUC üretmesin (tanı: tamamen gizli görüşte fp16 + sıralı AUC 0.421, fp32 + karışık 0.500).
    model.eval()
    test_idx = np.asarray(test_idx)
    order = rng.permutation(len(test_idx))
    probs = np.zeros((len(test_idx), n_classes), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(test_idx), 256):
            part = order[i:i + 256]
            x, _ = cache.batch(test_idx[part], view, train=False, extra_hidden_fn=extra_hidden_fn)
            probs[part] = torch.softmax(model(x).double(), 1).cpu().numpy()
    return probs, model
