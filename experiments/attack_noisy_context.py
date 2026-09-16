"""Adım 5 (çözüm, rakip): gürültülü ya da bulanık bağlam saldırısı (Bi-CryptoNets tarzı).

Bi-CryptoNets (arXiv 2402.01296) hassas bölgeyi şifreler, geri kalanı gürültü ekleyerek açık işler. Burada ROI gizli
(şifreli), bağlam bozulmuş ama açık bir sunucu görüşüne Saldırı B'nin uyarlanabilir saldırganı uygulanır: saldırgan aynı
bozulmayla eğitilir. Aynı bozulma düzeyinde fayda da ölçülür: ROI temiz + bağlam bozuk görüntüyle eğitilen teşhis modeli
(Bi-CryptoNets'in sunucusunun görebildiği bilginin şifresiz üst sınırı).

Bozulmalar (görüntü [0,1] ölçeğinde, 224 px):
- gauss σ ∈ {0.05, 0.1, 0.2, 0.4}: görüntü başına sabit gürültü (istemci bir kez gönderir); 64 sabit gürültü alanı
  bankasından görüntü indeksine göre seçilip kaydırılır. Saldırgan her epoch aynı gürültüyü görür (savunmanın lehine).
- bulanik σ ∈ {2, 4, 8} piksel: Gauss bulanıklığı.
Bozulma artırmadan (kaydırma/ölçekleme) önce, orijinal ızgarada uygulanır; ROI içi temiz kalır (fayda görüşü) ya da
gizlenir (saldırgan görüşü "baglam").

`--norm gorunur` (yalnız beyin, inceleme S1): istemci ham kesiti yalnızca açık gönderilen bağlam piksellerinin (ROI dışı)
yüzdelikleriyle normalize eder, sonra bağlamı bozar; ROI içeriği açık piksellerin ölçeğine girmez. Çıktılar `_gnorm` ekli.

Çıktılar: results/tables/saldiri_B_gurultu.csv|md (satırlar her koşudan sonra yazılır, biten atlanır).
Çalıştırma: .venv\\Scripts\\python -m experiments.attack_noisy_context [--dataset brain covidqu] [--seeds 0]
            [--norm kesit|gorunur] [--quick]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import config
from attacks.context_cnn import CpuCache, normalize_visible, random_affine, server_view, train_and_predict
from common.evaluation import auc_score, bootstrap_ci
from common.report import write_markdown_table
from experiments.attack_context import build_cache
from foveahe.data import load_dataset

LOG = config.LOGS / "attack_noisy_context.log"
PREDS = config.RESULTS / "preds"
PERTURBATIONS = [("gauss", 0.05), ("gauss", 0.1), ("gauss", 0.2), ("gauss", 0.4), ("bulanik", 2.0), ("bulanik", 4.0),
                 ("bulanik", 8.0)]
VIEWS = {"baglam": "saldırgan: ROI gizli, bağlam bozuk", "tam": "fayda: ROI temiz, bağlam bozuk"}
NOISE_BANK = 64


def log(msg: str):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = int(np.ceil(3 * sigma))
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), k.view(1, 1, -1, 1))


class PerturbedContextCache(CpuCache):
    """CpuCache + bağlam bozulması. Bozulma yalnızca ROI dışına, artırmadan önce uygulanır."""

    def __init__(self, images, masks, labels, kind: str, level: float, device: str = "cuda"):
        super().__init__(images, masks, labels, device)
        self.kind, self.level = kind, level
        size = images.shape[1]
        gen = torch.Generator(device="cpu").manual_seed(2402)
        self.bank = torch.randn((NOISE_BANK, 1, size, size), generator=gen).to(device)

    def perturb(self, x: torch.Tensor, m: torch.Tensor, idx_t: torch.Tensor) -> torch.Tensor:
        """Bağlam bozulması. Bulanıklık ROI dışarıda bırakılarak normalize edilir: ROI pikselleri bağlama yayılmaz."""
        if self.kind == "bulanik":
            keep = (~m).float()
            return gaussian_blur(x * keep, self.level) / gaussian_blur(keep, self.level).clamp_min(1e-3)
        n, _, h, w = x.shape
        idx = idx_t.to(self.device)
        noise = self.bank[idx % NOISE_BANK]  # görüntü başına sabit alan, satır ve sütun kaydırmalı
        rows = (torch.arange(h, device=self.device)[None, :] + ((idx // NOISE_BANK) % h)[:, None]) % h
        cols = (torch.arange(w, device=self.device)[None, :] + ((idx // (NOISE_BANK * h)) % w)[:, None]) % w
        noise = noise.gather(2, rows[:, None, :, None].expand(n, 1, h, w))
        noise = noise.gather(3, cols[:, None, None, :].expand(n, 1, h, w))
        return (x + self.level * noise).clamp_(0, 1)

    def batch(self, idx, view: str, train: bool, extra_hidden_fn=None):
        idx_t = torch.from_numpy(np.asarray(idx, dtype=np.int64))
        n = len(idx_t)
        torch.index_select(self.img, 0, idx_t, out=self._img_buf[:n])
        torch.index_select(self.mask, 0, idx_t, out=self._mask_buf[:n])
        x = self._img_buf[:n].to(self.device, non_blocking=True).float().div_(255).unsqueeze(1)
        m = self._mask_buf[:n].to(self.device, non_blocking=True).unsqueeze(1)
        x = torch.where(m, x, self.perturb(x, m, idx_t))  # ROI içi temiz, bağlam bozuk
        if train:
            x, m = random_affine(x, m)
        extra = extra_hidden_fn(m) if extra_hidden_fn is not None else None
        return server_view(x, m, view, extra), self.y[idx_t].to(self.device, non_blocking=True)


class PerturbedVisibleNormCache(PerturbedContextCache):
    """Ham yoğunluklar; normalizasyon yalnızca bağlam (ROI dışı) piksellerinden, bozulmadan önce (S1)."""

    def batch(self, idx, view: str, train: bool, extra_hidden_fn=None):
        idx_t = torch.from_numpy(np.asarray(idx, dtype=np.int64))
        n = len(idx_t)
        torch.index_select(self.img, 0, idx_t, out=self._img_buf[:n])
        torch.index_select(self.mask, 0, idx_t, out=self._mask_buf[:n])
        x = self._img_buf[:n].to(self.device, non_blocking=True).float().unsqueeze(1)
        m = self._mask_buf[:n].to(self.device, non_blocking=True).unsqueeze(1)
        x = normalize_visible(x, m)
        x = torch.where(m, x, self.perturb(x, m, idx_t))
        if train:
            x, m = random_affine(x, m)
        extra = extra_hidden_fn(m) if extra_hidden_fn is not None else None
        return server_view(x, m, view, extra), self.y[idx_t].to(self.device, non_blocking=True)


def save_row(row: dict, out_csv):
    keys = ["veri", "bozulma", "duzey", "gorus", "tohum"]
    if out_csv.exists():
        df = pd.read_csv(out_csv)
        same = np.logical_and.reduce([df[k] == row[k] for k in keys])
        df = pd.concat([df[~same], pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(out_csv, index=False)
    write_markdown_table(df, out_csv.with_suffix(".md"), floatfmt="{:.4f}")


def run(name: str, seeds, epochs: int, quick: bool, out_csv, tag: str, norm: str = "kesit"):
    ds = load_dataset(name)
    imgs, masks = build_cache(name, ds.img_paths, ds.mask_paths)
    cache_cls = PerturbedContextCache
    if norm == "gorunur":
        from experiments.brain_visible_norm import load_raw
        imgs, cache_cls = load_raw(), PerturbedVisibleNormCache
    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = set(zip(prev.veri, prev.bozulma, prev.duzey.astype(float), prev.gorus, prev.tohum.astype(int)))
    n_cls = len(ds.labels)
    perturbations = PERTURBATIONS[:1] + PERTURBATIONS[4:5] if quick else PERTURBATIONS
    for kind, level in perturbations:
        cache = cache_cls(imgs, masks, ds.y, kind, level)
        for view in VIEWS:
            for seed in seeds:
                if (ds.display, kind, float(level), view, seed) in done:
                    continue
                t0 = time.perf_counter()
                if name == "brain":
                    folds = np.unique(ds.folds)
                    probs = np.full((len(ds.y), n_cls), np.nan)
                    for k in folds[:1] if quick else folds:
                        tr, te = np.flatnonzero(ds.folds != k), np.flatnonzero(ds.folds == k)
                        probs[te], _ = train_and_predict(cache, tr, te, view, n_cls, epochs=epochs, seed=seed,
                                                         log=lambda *_: None)
                    te = np.flatnonzero(~np.isnan(probs[:, 0]))
                    p, groups = probs[te], ds.groups[te]
                else:
                    tr, te = ds.train_idx, ds.test_idx
                    if quick:
                        rng = np.random.default_rng(0)
                        tr, te = rng.choice(tr, 3000, replace=False), rng.choice(te, 1000, replace=False)
                    probs, _ = train_and_predict(cache, tr, te, view, n_cls, epochs=epochs, seed=seed,
                                                 log=lambda *_: None)
                    p, groups = probs, None
                yt = ds.y[te]
                auc = auc_score(yt, p)
                lo, hi = bootstrap_ci(yt, p, groups=groups, n_boot=500)
                np.save(PREDS / f"gurultu_{name}_{kind}{level:g}_{view}_s{seed}{tag}.npy", probs)
                row = {"veri": ds.display, "bozulma": kind, "duzey": float(level), "gorus": view,
                       "aciklama": VIEWS[view], "tohum": seed, "auc": auc, "ci95_alt": lo, "ci95_ust": hi,
                       "n": len(te), "epoch": epochs, "sure_s": time.perf_counter() - t0, "normalizasyon": norm}
                save_row(row, out_csv)
                log(f"[{name}{tag}] {kind} {level:g} {view:6s} tohum={seed} AUC={auc:.4f} (%95 GA {lo:.4f}-{hi:.4f}) "
                    f"{row['sure_s']:.0f} s")
        del cache
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="*", default=["brain", "covidqu"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--epochs-brain", type=int, default=12)
    ap.add_argument("--epochs-cxr", type=int, default=5)
    ap.add_argument("--norm", choices=["kesit", "gorunur"], default="kesit")
    ap.add_argument("--quick", action="store_true", help="bir gürültü ve bir bulanıklık düzeyi, küçük alt küme, 1 epoch")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "CUDA bulunamadı"
    if args.norm == "gorunur" and args.dataset != ["brain"]:
        raise SystemExit("--norm gorunur yalnız beyin MR için: --dataset brain")
    tag = ("_hizli" if args.quick else "") + ("_gnorm" if args.norm == "gorunur" else "")
    out_csv = config.TABLES / f"saldiri_B_gurultu{tag}.csv"
    for name in args.dataset:
        epochs = 1 if args.quick else (args.epochs_brain if name == "brain" else args.epochs_cxr)
        run(name, args.seeds, epochs, args.quick, out_csv, tag, args.norm)
    if out_csv.exists():
        print(pd.read_csv(out_csv).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
