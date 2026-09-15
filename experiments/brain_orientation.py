"""Kontrol: beyin MR kesit yönü (aksiyel / koronal / sagital) karıştırıcı mı?

Veri setinde yön etiketi yok. Kesitler görünüşlerine göre kümelenir (32×32 küçük resim + PCA + k-ortalamalar),
her kümeden örnek kolajı çizilir (yönler gözle etiketlenir), sonra meta veri saldırısı her küme içinde ayrı ölçülür.

Çalıştırma: .venv\\Scripts\\python -m experiments.brain_orientation [--k 3] [--within]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import config
from attacks.metadata import ALL_FEATURES
from common.evaluation import auc_score
from experiments.attack_metadata import brain_features, oof_predict


def cluster(k: int):
    imgs = np.load(config.DATA_PROC / "brain_224_img.npy")
    small = np.stack([np.asarray(Image.fromarray(im).resize((32, 32), Image.BILINEAR), dtype=np.float32) / 255
                      for im in imgs]).reshape(len(imgs), -1)
    z = PCA(n_components=40, random_state=0).fit_transform(small - small.mean(0))
    labels = KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(z)
    return imgs, labels


def orientation_features(im: np.ndarray):
    """(sağ-sol simetri korelasyonu, alt kenar doluluğu, baş kutusu en-boy oranı)."""
    from scipy import ndimage
    head = im > 25
    lab, n = ndimage.label(head)
    if n:
        sizes = ndimage.sum(head, lab, range(1, n + 1))
        head = lab == (int(np.argmax(sizes)) + 1)
    rows, cols = np.flatnonzero(head.any(1)), np.flatnonzero(head.any(0))
    if len(rows) == 0:
        return 0.0, 0.0, 1.0
    aspect = (rows.max() - rows.min() + 1) / (cols.max() - cols.min() + 1)
    bottom = float(head[-12:, :].mean())
    cc = int(round(np.nonzero(head)[1].mean()))
    half = min(cc, im.shape[1] - 1 - cc)
    win = im[:, cc - half:cc + half + 1].astype(np.float64)
    sym = float(np.corrcoef(win.ravel(), win[:, ::-1].ravel())[0, 1]) if half > 4 else 0.0
    return sym, bottom, float(aspect)


def heuristic_orientation(imgs: np.ndarray):
    feats = np.array([orientation_features(im) for im in imgs])
    z = (feats - feats.mean(0)) / feats.std(0)
    km = KMeans(n_clusters=3, n_init=20, random_state=0).fit(z)
    centers = feats[:, :2]
    means = np.array([centers[km.labels_ == c].mean(0) for c in range(3)])  # (simetri, alt doluluk)
    sag = int(np.argmin(means[:, 0]))
    rest = [c for c in range(3) if c != sag]
    cor = max(rest, key=lambda c: means[c, 1])
    names = {sag: "sagital", cor: "koronal"}
    names.update({c: "aksiyel" for c in rest if c != cor})
    return np.array([names[c] for c in km.labels_]), feats


def montage_named(imgs, names, per=12, path=None):
    rng = np.random.default_rng(1)
    rows = []
    for name in ["aksiyel", "koronal", "sagital"]:
        idx = rng.choice(np.flatnonzero(names == name), min(per, int((names == name).sum())), replace=False)
        tiles = [np.asarray(Image.fromarray(imgs[i]).resize((80, 80))) for i in idx]
        tiles += [np.zeros((80, 80), np.uint8)] * (per - len(tiles))
        rows.append(np.concatenate(tiles, axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def montage(imgs, labels, k, per=10):
    rng = np.random.default_rng(0)
    rows = []
    for c in range(k):
        idx = rng.choice(np.flatnonzero(labels == c), min(per, int((labels == c).sum())), replace=False)
        tiles = [np.asarray(Image.fromarray(imgs[i]).resize((96, 96))) for i in idx]
        tiles += [np.zeros((96, 96), np.uint8)] * (per - len(tiles))
        rows.append(np.concatenate(tiles, axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(config.FIGURES / f"beyin_kesit_kumeleri_k{k}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--names", nargs="*", default=None, help="küme sırasına göre yön adları (gözle belirlenir)")
    ap.add_argument("--heuristic", action="store_true", help="simetri / alt kenar / en-boy ipuçlarıyla yön tahmini")
    args = ap.parse_args()
    meta = pd.read_csv(config.DATA_PROC / "brain" / "meta.csv")
    if args.heuristic:
        imgs = np.load(config.DATA_PROC / "brain_224_img.npy")
        names, feats = heuristic_orientation(imgs)
        meta["yon"] = names
        meta["simetri"], meta["alt_doluluk"], meta["en_boy"] = feats[:, 0], feats[:, 1], feats[:, 2]
        montage_named(imgs, names, path=config.FIGURES / "beyin_kesit_yonu_tahmini.png")
        meta[["id", "yon", "simetri", "alt_doluluk", "en_boy"]].to_csv(
            config.DATA_PROC / "brain" / "kesit_yonu.csv", index=False)
        print(meta.groupby("yon")[["simetri", "alt_doluluk", "en_boy"]].mean().round(3).to_string())
        tag = "heuristik"
    else:
        imgs, labels = cluster(args.k)
        montage(imgs, labels, args.k)
        meta["kume"] = labels
        if args.names:
            meta["yon"] = [args.names[c] for c in labels]
        meta[["id", "kume"] + (["yon"] if args.names else [])].to_csv(
            config.DATA_PROC / "brain" / f"kesit_yonu_k{args.k}.csv", index=False)
        print("küme × tümör tipi:\n", pd.crosstab(meta["kume"], meta["label_name"]).to_string())
        if not args.names:
            print(f"kolaj: results/figures/beyin_kesit_kumeleri_k{args.k}.png "
                  "(yönleri gözle belirleyip --names ile tekrar çalıştırın)")
            return
        tag = f"k{args.k}"

    table = pd.crosstab(meta["yon"], meta["label_name"])
    print("yön × tümör tipi:\n", table.to_string())
    _, F = brain_features()
    y = meta["label"].to_numpy() - 1
    folds = meta["fold"].to_numpy()

    # 1) Yalnızca kesit yönü ne kadar sızdırıyor, meta veri bunun üstüne ne ekliyor?
    onehot = pd.get_dummies(meta["yon"], prefix="yon").astype(float)
    auc_dir = auc_score(y, oof_predict(onehot, y, folds, seed=0))
    auc_meta = auc_score(y, oof_predict(F[ALL_FEATURES], y, folds, seed=0))
    auc_both = auc_score(y, oof_predict(pd.concat([F[ALL_FEATURES], onehot], axis=1), y, folds, seed=0))
    print(f"yalnızca kesit yönü AUC={auc_dir:.3f} | yalnızca meta veri AUC={auc_meta:.3f} | ikisi birlikte AUC={auc_both:.3f}")

    # 2) Her yön içinde meta veri saldırısı (en az 20 örneği olan sınıflarla)
    results = {}
    for name in sorted(meta["yon"].unique()):
        sel = (meta["yon"] == name).to_numpy()
        counts = pd.Series(y[sel]).value_counts()
        keep_classes = counts[counts >= 20].index.to_numpy()
        sel = sel & np.isin(y, keep_classes)
        if len(keep_classes) < 2:
            continue
        remap = {c: i for i, c in enumerate(sorted(keep_classes))}
        yr = np.array([remap[v] for v in y[sel]])
        probs = oof_predict(F.loc[sel, ALL_FEATURES].reset_index(drop=True), yr, folds[sel], seed=0)
        results[name] = {"n": int(sel.sum()),
                         "siniflar": {meta.label_name[meta.label - 1 == c].iloc[0]: int((y[sel] == c).sum())
                                      for c in sorted(keep_classes)},
                         "meta_veri_auc": auc_score(yr, probs if len(keep_classes) > 2 else probs[:, 1])}
        print(f"[{name}] n={results[name]['n']} sınıflar={results[name]['siniflar']} "
              f"yön içinde meta veri AUC={results[name]['meta_veri_auc']:.3f}")
    with open(config.TABLES / f"saldiri_A_beyin_yon_kontrolu_{tag}.json", "w", encoding="utf-8") as f:
        json.dump({"yon_tumor_tablosu": table.to_dict(), "yalniz_yon_auc": auc_dir, "yalniz_meta_auc": auc_meta,
                   "yon_arti_meta_auc": auc_both, "yon_icinde": results}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
