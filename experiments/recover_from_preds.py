"""Yardımcı adım: Saldırı B satırlarını kayıtlı tahmin dosyalarından kurtarma.

attack_context CSV'yi yalnızca çalışmanın sonunda yazar. Kuyruk yarıda kesilirse biten tohumların
tahminleri results/preds altında kalır ama CSV'ye girmez. Bu betik o tahminlerden satırları
attack_context ile aynı ölçüm koduyla (aynı bootstrap tohumu, n_boot=500) yeniden hesaplar ve
saldiri_B_baglam.csv'ye ekler. Düzeltilmiş değerlendirmeden (fp32 + karışık sıra) önce üretilmiş
tahmin dosyaları kullanılmaz.

Uyarı: attack_context çalışırken bu betiği çalıştırma; attack_context CSV'yi başlangıçta okuyup
sonda yazdığı için buradaki satırların üzerine yazar.

Çalıştırma: .venv\\Scripts\\python -m experiments.recover_from_preds --dataset brain --views tam baglam
            [--seeds 0 1 2 3 4]
"""
from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

import config
from common.evaluation import auc_score, bootstrap_ci
from common.report import write_markdown_table

PREDS = config.RESULTS / "preds"
OUT_CSV = config.TABLES / "saldiri_B_baglam.csv"
FIX_TIME = datetime(2026, 9, 14, 17, 46)  # düzeltilmiş değerlendirmeyle kuyruğun başladığı an
VERI = {"brain": "beyin MR (Cheng)", "covidqu": "akciğer grafisi (COVID-QU-Ex)"}
HEDEF = {"brain": "tümör tipi (3 sınıf)", "covidqu": "teşhis (3 sınıf)"}
LABELS = ["COVID-19", "Non-COVID", "Normal"]
PAIRS = [("Normal", "Non-COVID"), ("Normal", "COVID-19"), ("Non-COVID", "COVID-19")]


def load_labels(ds: str):
    if ds == "brain":
        meta = pd.read_csv(config.DATA_PROC / "brain" / "meta.csv")
        return meta["label"].to_numpy() - 1, meta["pid"].to_numpy(), None
    man = pd.read_csv(config.DATA_PROC / "covidqu_manifest.csv")
    man = man[man.lung_mask_path.notna() & (man.lung_mask_path != "")].reset_index(drop=True)
    y = man.label.map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    return y, None, np.flatnonzero((man.split == "Test").to_numpy())


def make_row(ds, view, seed, frac, p, y, groups, te):
    row = {"veri": VERI[ds], "hedef": HEDEF[ds], "gorus": view, "tohum": seed, "gizli_alan_ort": frac}
    if ds == "brain":
        done = ~np.isnan(p[:, 0])
        row["auc"] = auc_score(y[done], p[done])
        row["ci95_alt"], row["ci95_ust"] = bootstrap_ci(y[done], p[done], groups=groups[done], n_boot=500)
        row["n"], row["sure_s"] = int(done.sum()), np.nan
        return row
    assert len(p) == len(te), f"tahmin sayısı ({len(p)}) test bölmesiyle ({len(te)}) uyuşmuyor"
    yt = y[te]
    row["auc"] = auc_score(yt, p)
    row["ci95_alt"], row["ci95_ust"] = bootstrap_ci(yt, p, n_boot=500)
    row["n"], row["sure_s"] = len(te), np.nan
    for a, b in PAIRS:
        ia, ib = LABELS.index(a), LABELS.index(b)
        sel = np.isin(yt, [ia, ib])
        score = p[sel, ib] / (p[sel, ia] + p[sel, ib] + 1e-12)
        row[f"auc_{a}_vs_{b}"] = auc_score((yt[sel] == ib).astype(int), score)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="*", default=["brain", "covidqu"])
    ap.add_argument("--views", nargs="*", required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    df = pd.read_csv(OUT_CSV)
    rows = []
    for ds in args.dataset:
        y, groups, te = load_labels(ds)
        for view in args.views:
            known = df[(df.veri == VERI[ds]) & (df.gorus == view)]
            if known.empty:
                print(f"[{ds}] {view}: CSV'de gizli alan oranı yok, attack_context ile çalıştırılmalı")
                continue
            frac = float(known.gizli_alan_ort.iloc[0])
            for seed in args.seeds:
                f = PREDS / f"{ds}_{view}_s{seed}.npy"
                if not f.exists() or datetime.fromtimestamp(f.stat().st_mtime) < FIX_TIME:
                    print(f"atlandı (yok ya da düzeltme öncesi): {f.name}")
                    continue
                row = make_row(ds, view, seed, frac, np.load(f), y, groups, te)
                rows.append(row)
                print(f"{f.name}: AUC={row['auc']:.3f} (%95 GA {row['ci95_alt']:.3f}-{row['ci95_ust']:.3f})")
    if not rows:
        print("kurtarılacak satır yok")
        return
    new = pd.DataFrame(rows)
    keys = ["veri", "gorus", "tohum"]
    old = df.merge(new[keys], on=keys, how="left", indicator=True)
    old = old[old["_merge"] == "left_only"].drop(columns="_merge")
    out = pd.concat([old, new], ignore_index=True)
    out.to_csv(OUT_CSV, index=False)
    write_markdown_table(out, config.TABLES / "saldiri_B_baglam.md", floatfmt="{:.3f}")
    print(f"{len(new)} satır eklendi/güncellendi -> {OUT_CSV.name}")


if __name__ == "__main__":
    main()
