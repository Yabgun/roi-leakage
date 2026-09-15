"""Adım: Kök neden — saldırgan görüntünün neresine bakıyor, hangi önişleme sızıntıyı ne kadar azaltıyor?

1) Grad-CAM: 'baglam' görüşüyle eğitilmiş saldırganın dikkat kütlesinin bölgelere dağılımı.
   Göğüs: kenar çerçeve, akciğer üstü (boyun/omuz), akciğer altı (karın), akciğerler arası (kalp/mediasten),
   yanlar (göğüs duvarı/kollar), akciğer (gizli). Beyin: tümör (gizli), tümör çevresi halka, beynin geri kalanı, arka plan.
2) Önişleme ablasyonu (yalnızca göğüs): yok, histogram eşitleme, %5/%10 kenar gizleme, eşitleme + %10 kenar.

Çalıştırma: .venv\\Scripts\\python -m experiments.root_cause [--dataset covidqu brain] [--quick]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import config
from attacks.context_cnn import CpuCache, train_and_predict
from common.evaluation import auc_score
from common.report import write_markdown_table
from defenses.roi_expansion import dilate_np
from experiments.attack_context import RES, build_cache, log

CXR_ZONES = ["akciger (gizli)", "kenar cerceve", "akciger ustu", "akciger alti", "akcigerler arasi", "yanlar"]
BRAIN_ZONES = ["tumor (gizli)", "tumor cevresi", "beynin geri kalani", "arka plan"]


def gradcam(model, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    feats = {}
    h1 = model.layer4.register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))
    model.zero_grad(set_to_none=True)
    logits = model(x)
    a = feats["a"]
    a.retain_grad()
    logits.gather(1, target.view(-1, 1)).sum().backward()
    h1.remove()
    w = a.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    return cam / (cam.sum(dim=(1, 2), keepdim=True) + 1e-12)


def cxr_zones(m: np.ndarray) -> dict:
    h, w = m.shape
    frame = np.zeros_like(m)
    b = int(0.10 * h)
    frame[:b, :] = frame[-b:, :] = frame[:, :b] = frame[:, -b:] = True
    zones = {"akciger (gizli)": m}
    rows, cols = np.nonzero(m)
    if len(rows) == 0:
        return {**zones, "kenar cerceve": frame & ~m}
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    rr = np.arange(h)[:, None]
    cc = np.arange(w)[None, :]
    inside_rows = (rr >= r0) & (rr <= r1)
    between = np.zeros_like(m)
    for r in range(r0, r1 + 1):
        lc = np.flatnonzero(m[r])
        if len(lc) >= 2:
            gaps = np.flatnonzero(np.diff(lc) > 1)
            if len(gaps):
                g = gaps[np.argmax(np.diff(lc)[gaps])]
                between[r, lc[g] + 1:lc[g + 1]] = True
    rest = ~m & ~frame
    zones["kenar cerceve"] = frame & ~m
    zones["akciger ustu"] = rest & (rr < r0)
    zones["akciger alti"] = rest & (rr > r1)
    zones["akcigerler arasi"] = rest & between
    zones["yanlar"] = rest & inside_rows & ~between
    return zones


def brain_zones(m: np.ndarray, img: np.ndarray) -> dict:
    head = img > 20
    ring = dilate_np(m, 16) & ~m
    return {"tumor (gizli)": m, "tumor cevresi": ring & head, "beynin geri kalani": head & ~m & ~ring,
            "arka plan": ~head & ~m}


def cam_zone_table(dataset, model, cache, idx, y, masks, imgs, labels, zone_fn, zone_names, batch=32):
    shares = {z: [] for z in zone_names}
    classes = []
    maps = np.zeros((len(labels), RES, RES))
    counts = np.zeros(len(labels))
    model.eval()
    for i in range(0, len(idx), batch):
        b = idx[i:i + batch]
        x, yt = cache.batch(b, "baglam", train=False)
        x = x.float().requires_grad_(False)
        with torch.enable_grad():
            cam = gradcam(model, x, yt).detach().cpu().numpy()
        for j, k in enumerate(b):
            zones = zone_fn(masks[k]) if zone_fn is cxr_zones else zone_fn(masks[k], imgs[k])
            for z in zone_names:
                shares[z].append(float(cam[j][zones[z]].sum()) if z in zones else 0.0)
            classes.append(int(y[k]))
            maps[y[k]] += cam[j]
            counts[y[k]] += 1
    df = pd.DataFrame(shares)
    df["sinif"] = [labels[c] for c in classes]
    table = df.groupby("sinif").mean().reset_index()
    overall = df.drop(columns="sinif").mean().to_frame().T
    overall.insert(0, "sinif", "hepsi")
    table = pd.concat([table, overall], ignore_index=True)
    table.insert(0, "veri", dataset)
    return table, maps / np.maximum(counts, 1)[:, None, None]


def plot_maps(dataset, maps, labels, mean_mask):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 4))
    for ax, lab, mp in zip(np.atleast_1d(axes), labels, maps):
        ax.imshow(mp, cmap="inferno")
        ax.contour(mean_mask, levels=[0.5], colors="cyan", linewidths=0.8)
        ax.set_title(f"{lab}: ortalama Grad-CAM")
        ax.axis("off")
    fig.suptitle(f"Saldırgan nereye bakıyor? ({dataset}, ROI gizli; camgöbeği = ortalama ROI sınırı)")
    fig.tight_layout()
    fig.savefig(config.FIGURES / f"kok_neden_gradcam_{dataset}.png", dpi=150)


def equalize_cache(imgs: np.ndarray) -> np.ndarray:
    out = np.empty_like(imgs)
    for i, im in enumerate(imgs):
        hist = np.bincount(im.ravel(), minlength=256)
        cdf = hist.cumsum()
        cdf = (cdf - cdf[cdf > 0].min()) / max(cdf[-1] - cdf[cdf > 0].min(), 1) * 255
        out[i] = cdf[im].astype(np.uint8)
    return out


def frame_fn(frac: float):
    b = int(frac * RES)
    fr = torch.zeros((1, 1, RES, RES), dtype=torch.bool, device="cuda")
    fr[..., :b, :] = fr[..., -b:, :] = fr[..., :, :b] = fr[..., :, -b:] = True
    return lambda m: fr.expand_as(m)


def run_covidqu(quick, epochs, rows_cam, rows_abl):
    man = pd.read_csv(config.DATA_PROC / "covidqu_manifest.csv")
    imgs, masks = build_cache("covidqu", man.img_path, man.lung_mask_path)
    labels = ["COVID-19", "Non-COVID", "Normal"]
    y = man.label.map({l: i for i, l in enumerate(labels)}).to_numpy()
    rng = np.random.default_rng(0)
    tr = np.flatnonzero(man.split.isin(["Train", "Val"]).to_numpy())
    te = np.flatnonzero((man.split == "Test").to_numpy())
    if quick:
        tr, te = rng.choice(tr, 3000, replace=False), rng.choice(te, 600, replace=False)
    cache = CpuCache(imgs, masks, y)
    log("[kök neden] covidqu: baglam saldırganı eğitiliyor")
    probs, model = train_and_predict(cache, tr, te, "baglam", 3, epochs=epochs, seed=0, log=log)
    cam_idx = rng.choice(te, min(1500, len(te)), replace=False)
    table, maps = cam_zone_table("covidqu", model, cache, cam_idx, y, masks, imgs, labels, cxr_zones, CXR_ZONES)
    rows_cam.append(table)
    plot_maps("covidqu", maps, labels, masks[cam_idx].mean(0))

    eq = equalize_cache(imgs)
    cache_eq = CpuCache(eq, masks, y)
    variants = [("yok", cache, None, auc_score(y[te], probs)), ("histogram_esitleme", cache_eq, None, None),
                ("kenar_gizle_5", cache, frame_fn(0.05), None), ("kenar_gizle_10", cache, frame_fn(0.10), None),
                ("esitleme+kenar_10", cache_eq, frame_fn(0.10), None)]
    for name, c, fn, auc in variants:
        if auc is None:
            log(f"[kök neden] önişleme={name}")
            p, _ = train_and_predict(c, tr, te, "baglam", 3, epochs=epochs, seed=0, extra_hidden_fn=fn, log=log)
            auc = auc_score(y[te], p)
        rows_abl.append({"veri": "covidqu", "onisleme": name, "auc_3sinif": auc})
        log(f"[kök neden] önişleme={name:20s} AUC={auc:.3f}")


def run_brain(quick, epochs, rows_cam):
    meta = pd.read_csv(config.DATA_PROC / "brain" / "meta.csv")
    base = config.DATA_PROC / "brain"
    imgs, masks = build_cache("brain", [base / p for p in meta.img_path], [base / p for p in meta.mask_path])
    y = meta["label"].to_numpy() - 1
    folds = meta["fold"].to_numpy()
    last = np.unique(folds)[-1]
    tr, te = np.flatnonzero(folds != last), np.flatnonzero(folds == last)
    labels = ["meningioma", "glioma", "pituitary"]
    cache = CpuCache(imgs, masks, y)
    log("[kök neden] beyin: baglam saldırganı eğitiliyor")
    _, model = train_and_predict(cache, tr, te, "baglam", 3, epochs=1 if quick else epochs, seed=0, log=log)
    table, maps = cam_zone_table("brain", model, cache, te, y, masks, imgs, labels, brain_zones, BRAIN_ZONES)
    rows_cam.append(table)
    plot_maps("brain", maps, labels, masks[te].mean(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="*", default=["covidqu", "brain"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    rows_cam, rows_abl = [], []
    if "covidqu" in args.dataset:
        run_covidqu(args.quick, 1 if args.quick else 5, rows_cam, rows_abl)
    if "brain" in args.dataset:
        run_brain(args.quick, 12, rows_cam)
    tag = "_hizli" if args.quick else ""
    if rows_cam:
        cam = pd.concat(rows_cam, ignore_index=True)
        cam.to_csv(config.TABLES / f"kok_neden_gradcam{tag}.csv", index=False)
        write_markdown_table(cam, config.TABLES / f"kok_neden_gradcam{tag}.md", floatfmt="{:.3f}")
        print(cam.round(3).to_string(index=False))
    if rows_abl:
        abl = pd.DataFrame(rows_abl)
        abl.to_csv(config.TABLES / f"kok_neden_onisleme{tag}.csv", index=False)
        write_markdown_table(abl, config.TABLES / f"kok_neden_onisleme{tag}.md", floatfmt="{:.3f}")
        print(abl.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
