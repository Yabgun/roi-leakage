"""Adım (gerçekçilik testi): saldırgan bir veri kaynağında eğitilip başka kaynakta denenir.

Soru: bulut sağlayıcı, hastanenin kendi verisine erişmeden, herkese açık başka bir veriyle eğittiği saldırganla
teşhisi yine çıkarabilir mi?
- Yön A: COVID-QU-Ex (Normal / Non-COVID) ile eğit -> Kaggle CXR (NORMAL / PNEUMONIA) üzerinde test
- Yön B: Kaggle CXR train ile eğit -> COVID-QU-Ex Test üzerinde test
Kaggle görüntülerinden COVID-QU-Ex'te kopyası olanlar (dHash + korelasyon) dışarıda bırakılır.
Kaggle akciğer maskeleri `experiments.lung_segmenter` ile üretilmiş olmalıdır.

Çalıştırma: .venv\\Scripts\\python -m experiments.attack_context_transfer [--views tam baglam] [--quick]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import config
from attacks.context_cnn import CpuCache, train_and_predict
from common.evaluation import auc_score, bootstrap_ci
from common.report import write_markdown_table
from experiments.attack_context import build_cache, hidden_fraction, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="*", default=["tam", "baglam"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cq = pd.read_csv(config.DATA_PROC / "covidqu_manifest.csv")
    kg = pd.read_csv(config.DATA_PROC / "kaggle_cxr_manifest.csv")
    assert "lung_mask_path" in kg.columns, "önce experiments.lung_segmenter çalıştırılmalı"
    dup = pd.read_csv(config.DATA_PROC / "cxr_capraz_tekrarlar.csv")
    kg["kopya"] = kg.index.isin(dup.kaggle_idx.unique())

    cq_img, cq_mask = build_cache("covidqu", cq.img_path, cq.lung_mask_path)
    kg_img, kg_mask = build_cache("kaggle", kg.img_path, kg.lung_mask_path)

    cq_sel = np.flatnonzero(cq.label.isin(["Normal", "Non-COVID"]).to_numpy())
    kg_sel = np.flatnonzero((kg.label.isin(["NORMAL", "PNEUMONIA"]) & ~kg.kopya).to_numpy())
    y_cq = (cq.label == "Non-COVID").astype(int).to_numpy()
    y_kg = (kg.label == "PNEUMONIA").astype(int).to_numpy()

    offset = len(cq)
    cache = CpuCache(np.concatenate([cq_img, kg_img]), np.concatenate([cq_mask, kg_mask]),
                     np.concatenate([y_cq, y_kg]))
    y_all = np.concatenate([y_cq, y_kg])

    cq_train = cq_sel[cq.split.to_numpy()[cq_sel] != "Test"]
    cq_test = cq_sel[cq.split.to_numpy()[cq_sel] == "Test"]
    kg_train = kg_sel[kg.split.to_numpy()[kg_sel] == "train"] + offset
    kg_all = kg_sel + offset
    if args.quick:
        rng = np.random.default_rng(0)
        cq_train = rng.choice(cq_train, 3000, replace=False)
        args.epochs = 1

    log(f"[transfer] Kaggle kopya olmayan NORMAL/PNEUMONIA: {len(kg_sel)} "
        f"(NORMAL {int((y_kg[kg_sel] == 0).sum())}, PNEUMONIA {int((y_kg[kg_sel] == 1).sum())}); "
        f"COVID-QU-Ex eğitim {len(cq_train)}, test {len(cq_test)}")
    directions = [("COVID-QU-Ex -> Kaggle", cq_train, kg_all), ("Kaggle -> COVID-QU-Ex", kg_train, cq_test),
                  ("COVID-QU-Ex -> COVID-QU-Ex (referans)", cq_train, cq_test)]
    rows = []
    masks_all = np.concatenate([cq_mask, kg_mask])
    for view in args.views:
        for name, tr, te in directions:
            for seed in args.seeds:
                log(f"[transfer] {name} görüş={view} tohum={seed} (eğitim {len(tr)}, test {len(te)})")
                p, _ = train_and_predict(cache, tr, te, view, 2, epochs=args.epochs, seed=seed, log=log)
                auc = auc_score(y_all[te], p)
                lo, hi = bootstrap_ci(y_all[te], p, n_boot=500)
                rows.append({"yon": name, "gorus": view, "tohum": seed, "auc": auc, "ci95_alt": lo, "ci95_ust": hi,
                             "n_egitim": len(tr), "n_test": len(te),
                             "gizli_alan_test": hidden_fraction(masks_all[te], view)})
                log(f"[transfer] {name:38s} görüş={view:7s} AUC={auc:.3f} (%95 GA {lo:.3f}-{hi:.3f})")
    df = pd.DataFrame(rows)
    tag = "_hizli" if args.quick else ""
    df.to_csv(config.TABLES / f"saldiri_B_transfer{tag}.csv", index=False)
    write_markdown_table(df, config.TABLES / f"saldiri_B_transfer{tag}.md", floatfmt="{:.3f}")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
