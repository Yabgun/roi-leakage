"""Şifreli çalışabilen modeller: şifresiz eğitim (PyTorch) ve CKKS için numpy ağırlık aktarımı.

- D (doğrusal): z = W2 (W1 x + b1) + b2, gizli boyut 20. Π_ROI ile aynı model sınıfıdır; seçici şifreleme çıktıyı
  değiştirmediği için bu sınıfın tam görüntüdeki doğruluğu Π_ROI'nin doğruluğudur. Çarpma derinliği 2.
- D2 (kare aktivasyon): z = W2 (BN(W1 x + b1))² + b2. Çıkarımda BN afindir ve ilk katmana katlanır. Derinlik 3.

Eğitimde girdi standardize edilir (eğitim kümesi ortalama/std, std alttan STD_MIN ile sınırlı: katlanan ağırlıklar
CKKS ölçeğinde büyümesin). Aktarımda standardizasyon ve BN ilk katmana katlanır; aktarılan ağırlıklar [0,1] ölçekli
ham temsil vektörü üzerinde çalışır.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

HIDDEN = 20
STD_MIN = 0.05
KINDS = ("D", "D2")


class Standardize(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        return (x - self.mean) / self.std


class Square(nn.Module):
    def forward(self, x):
        return x * x


def make_model(kind: str, n_in: int, n_classes: int, mean: torch.Tensor, std: torch.Tensor,
               hidden: int = HIDDEN) -> nn.Sequential:
    layers = [Standardize(mean, std), nn.Linear(n_in, hidden)]
    if kind == "D2":
        layers += [nn.BatchNorm1d(hidden), Square()]
    elif kind != "D":
        raise ValueError(kind)
    layers.append(nn.Linear(hidden, n_classes))
    return nn.Sequential(*layers)


@torch.no_grad()
def export(model: nn.Sequential) -> dict:
    """Standardizasyon ve BN katlanmış float64 ağırlıklar: kind, W1 (h,n), b1 (h,), W2 (c,h), b2 (c,)."""
    std, lin1 = model[0], model[1]
    w1 = lin1.weight.double() / std.std.double()
    b1 = lin1.bias.double() - (lin1.weight.double() * (std.mean.double() / std.std.double())).sum(1)
    kind = "D2" if isinstance(model[2], nn.BatchNorm1d) else "D"
    if kind == "D2":
        bn = model[2]
        g = bn.weight.double() / torch.sqrt(bn.running_var.double() + bn.eps)
        w1 = w1 * g[:, None]
        b1 = (b1 - bn.running_mean.double()) * g + bn.bias.double()
    lin2 = model[-1]
    return {"kind": kind, "W1": w1.cpu().numpy(), "b1": b1.cpu().numpy(),
            "W2": lin2.weight.double().cpu().numpy(), "b2": lin2.bias.double().cpu().numpy()}


def forward_numpy(w: dict, x: np.ndarray) -> np.ndarray:
    """Aktarılan ağırlıklarla şifresiz başvuru çıkarımı; logit döner."""
    h = np.asarray(x, dtype=np.float64) @ w["W1"].T + w["b1"]
    if str(w["kind"]) == "D2":
        h = h * h
    return h @ w["W2"].T + w["b2"]


def save_weights(w: dict, path) -> None:
    np.savez(path, **w)


def load_weights(path) -> dict:
    with np.load(path) as z:
        return {k: (str(z[k]) if k == "kind" else z[k]) for k in z.files}
