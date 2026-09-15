"""Adım: Saldırı B — bağlam saldırısı (CNN).

Sunucunun görüşleri: tam (şifresiz, üst sınır), yalniz_roi (referans), baglam (Π_ROI: ROI gizli),
baglam_genisK (ROI K piksel genişletilmiş), kutu (ROI'nin sınırlayıcı kutusu gizli).

Beyin MR (Cheng): tümör ROI'si, 3 sınıf, hasta bazlı 5 kat.
Akciğer grafisi (COVID-QU-Ex): akciğer ROI'si, 3 sınıf + ikili karşılaştırmalar, resmi Test bölmesi.

Çalıştırma: .venv\\Scripts\\python -m experiments.attack_context [--dataset brain covidqu] [--views ...]
            [--seeds 0] [--quick]
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupKFold

import config
from attacks.context_cnn import VIEWS, CpuCache, hidden_region, train_and_predict
from common.evaluation import auc_score, bootstrap_ci
from common.report import write_markdown_table

RES = 224
PREDS = config.RESULTS / "preds"
PREDS.mkdir(parents=True, exist_ok=True)
LOG = config.LOGS / "attack_context.log"


def log(msg: str):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _load_pair(paths):
    img_path, mask_path = paths
    with Image.open(img_path) as im:
        g = np.asarray(im.convert("L").resize((RES, RES), Image.BILINEAR), dtype=np.uint8)
    with Image.open(mask_path) as mm:
        m = np.asarray(mm.convert("L").resize((RES, RES), Image.NEAREST)) > 127
    return g, m


def build_cache(name: str, img_paths, mask_paths):
    ci, cm = config.DATA_PROC / f"{name}_{RES}_img.npy", config.DATA_PROC / f"{name}_{RES}_mask.npy"
    if ci.exists() and cm.exists():
        return np.load(ci), np.unpackbits(np.load(cm), axis=-1).astype(bool)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(8) as ex:
        pairs = list(ex.map(_load_pair, zip(img_paths, mask_paths)))
    imgs = np.stack([p[0] for p in pairs])
    masks = np.stack([p[1] for p in pairs])
    np.save(ci, imgs)
    np.save(cm, np.packbits(masks, axis=-1))
    log(f"[önbellek] {name}: {imgs.shape} ({time.perf_counter() - t0:.0f} s)")
    return imgs, masks


def hidden_fraction(masks: np.ndarray, view: str) -> float:
    fr = []
    for i in range(0, len(masks), 512):
        m = torch.from_numpy(masks[i:i + 512]).cuda().unsqueeze(1)
        fr.append(hidden_region(m, view).float().mean(dim=(1, 2, 3)).cpu())
    return float(torch.cat(fr).mean())


def run_brain(views, seeds, epochs, rows, quick):
    meta = pd.read_csv(config.DATA_PROC / "brain" / "meta.csv")
    base = config.DATA_PROC / "brain"
    imgs, masks = build_cache("brain", [base / p for p in meta.img_path], [base / p for p in meta.mask_path])
    y = meta["label"].to_numpy() - 1
    groups = meta["pid"].to_numpy()
    folds = meta["fold"].to_numpy()
    if (meta.groupby("pid")["fold"].nunique() > 1).any() or (folds < 0).any():
        folds = np.zeros(len(y), dtype=int)
        for k, (_, te) in enumerate(GroupKFold(5).split(imgs, y, groups)):
            folds[te] = k
    fold_ids = np.unique(folds)[:1] if quick else np.unique(folds)
    cache = CpuCache(imgs, masks, y)
    for view in views:
        frac = hidden_fraction(masks, view)
        for seed in seeds:
            t0 = time.perf_counter()
            oof = np.full((len(y), 3), np.nan)
            for k in fold_ids:
                tr, te = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
                log(f"[beyin] görüş={view} tohum={seed} kat={k} (eğitim {len(tr)}, test {len(te)})")
                p, _ = train_and_predict(cache, tr, te, view, 3, epochs=epochs, seed=seed, log=log)
                oof[te] = p
            done = ~np.isnan(oof[:, 0])
            auc = auc_score(y[done], oof[done])
            lo, hi = bootstrap_ci(y[done], oof[done], groups=groups[done], n_boot=500)
            np.save(PREDS / f"brain_{view}_s{seed}.npy", oof)
            rows.append({"veri": "beyin MR (Cheng)", "hedef": "tümör tipi (3 sınıf)", "gorus": view, "tohum": seed,
                         "gizli_alan_ort": frac, "auc": auc, "ci95_alt": lo, "ci95_ust": hi, "n": int(done.sum()),
                         "sure_s": time.perf_counter() - t0})
            log(f"[beyin] görüş={view:15s} tohum={seed} makro AUC={auc:.3f} (%95 GA {lo:.3f}-{hi:.3f}) "
                f"gizli alan={frac:.3f}")


def run_covidqu(views, seeds, epochs, rows, quick):
    man = pd.read_csv(config.DATA_PROC / "covidqu_manifest.csv")
    man = man[man.lung_mask_path.notna() & (man.lung_mask_path != "")].reset_index(drop=True)
    imgs, masks = build_cache("covidqu", man.img_path, man.lung_mask_path)
    labels = ["COVID-19", "Non-COVID", "Normal"]
    y = man.label.map({l: i for i, l in enumerate(labels)}).to_numpy()
    tr = np.flatnonzero(man.split.isin(["Train", "Val"]).to_numpy())
    te = np.flatnonzero((man.split == "Test").to_numpy())
    if quick:
        rng = np.random.default_rng(0)
        tr, te = rng.choice(tr, 3000, replace=False), rng.choice(te, 1000, replace=False)
    cache = CpuCache(imgs, masks, y)
    pairs = [("Normal", "Non-COVID"), ("Normal", "COVID-19"), ("Non-COVID", "COVID-19")]
    for view in views:
        frac = hidden_fraction(masks, view)
        for seed in seeds:
            t0 = time.perf_counter()
            log(f"[covidqu] görüş={view} tohum={seed} (eğitim {len(tr)}, test {len(te)})")
            p, _ = train_and_predict(cache, tr, te, view, 3, epochs=epochs, seed=seed, log=log)
            np.save(PREDS / f"covidqu_{view}_s{seed}.npy", p)
            yt = y[te]
            auc3 = auc_score(yt, p)
            lo, hi = bootstrap_ci(yt, p, n_boot=500)
            row = {"veri": "akciğer grafisi (COVID-QU-Ex)", "hedef": "teşhis (3 sınıf)", "gorus": view, "tohum": seed,
                   "gizli_alan_ort": frac, "auc": auc3, "ci95_alt": lo, "ci95_ust": hi, "n": len(te),
                   "sure_s": time.perf_counter() - t0}
            for a, b in pairs:
                ia, ib = labels.index(a), labels.index(b)
                sel = np.isin(yt, [ia, ib])
                score = p[sel, ib] / (p[sel, ia] + p[sel, ib] + 1e-12)
                row[f"auc_{a}_vs_{b}"] = auc_score((yt[sel] == ib).astype(int), score)
            rows.append(row)
            log(f"[covidqu] görüş={view:15s} tohum={seed} 3 sınıf AUC={auc3:.3f} (%95 GA {lo:.3f}-{hi:.3f}) "
                f"Normal/Pnömoni={row['auc_Normal_vs_Non-COVID']:.3f} gizli alan={frac:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="*", default=["brain", "covidqu"])
    ap.add_argument("--views", nargs="*", default=list(VIEWS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--epochs-brain", type=int, default=12)
    ap.add_argument("--epochs-cxr", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="küçük alt küme ve 1 epoch ile boru hattı testi")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "CUDA bulunamadı"
    rows = []
    tag = "_hizli" if args.quick else ""
    out_csv = config.TABLES / f"saldiri_B_baglam{tag}.csv"
    previous = pd.read_csv(out_csv) if out_csv.exists() and not args.quick else None
    if "brain" in args.dataset:
        run_brain(args.views, args.seeds, 1 if args.quick else args.epochs_brain, rows, args.quick)
    if "covidqu" in args.dataset:
        run_covidqu(args.views, args.seeds, 1 if args.quick else args.epochs_cxr, rows, args.quick)
    df = pd.DataFrame(rows)
    if previous is not None:
        keys = ["veri", "gorus", "tohum"]
        previous = previous.merge(df[keys], on=keys, how="left", indicator=True)
        previous = previous[previous["_merge"] == "left_only"].drop(columns="_merge")
        df = pd.concat([previous, df], ignore_index=True)
    df.to_csv(out_csv, index=False)
    write_markdown_table(df, config.TABLES / f"saldiri_B_baglam{tag}.md", floatfmt="{:.3f}")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
