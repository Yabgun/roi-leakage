"""Adım: Saldırı A — ROI meta verisinden teşhis çıkarımı.

Beyin MR (Cheng): tümör ROI'sinin konum/büyüklük/şekli -> tümör tipi (3 sınıf, makro AUC), hasta bazlı 5 kat.
Akciğer grafisi (COVID-QU-Ex): akciğer ROI'sinin geometrisi -> teşhis (3 sınıf ve ikili karşılaştırmalar), resmi bölme.
Akciğer grafisi (Kaggle): yalnızca girdi boyutu -> teşhis (orijinal çözünürlükler korunmuş set).

Çalıştırma: .venv\\Scripts\\python -m experiments.attack_metadata
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

import config
from attacks.metadata import ALL_FEATURES, FEATURE_GROUPS, mask_features
from common.evaluation import auc_score, bootstrap_ci
from common.images import load_mask
from common.report import write_markdown_table

MASK_RES = 256


def gbm(seed: int):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, random_state=seed)


def oof_predict(X: pd.DataFrame, y: np.ndarray, folds: np.ndarray, seed: int) -> np.ndarray:
    n_classes = len(np.unique(y))
    proba = np.zeros((len(y), n_classes))
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        model = gbm(seed).fit(X[tr], y[tr])
        proba[te] = model.predict_proba(X[te])
    return proba


def brain_features():
    meta = pd.read_csv(config.DATA_PROC / "brain" / "meta.csv")

    def one(row):
        m = load_mask(config.DATA_PROC / "brain" / row.mask_path, MASK_RES)
        return mask_features(m, row.h, row.w)

    with ThreadPoolExecutor(8) as ex:
        feats = list(ex.map(one, meta.itertuples(index=False)))
    return meta, pd.DataFrame(feats)


def run_brain(rows, importances):
    meta, F = brain_features()
    y = meta["label"].to_numpy() - 1
    groups = meta["pid"].to_numpy()
    folds = meta["fold"].to_numpy()
    multi = (meta.groupby("pid")["fold"].nunique() > 1).any() or (folds < 0).any()
    if multi:  # resmi katlar hasta bazlı değilse hasta bazlı kat oluştur
        folds = np.zeros(len(y), dtype=int)
        for k, (_, te) in enumerate(GroupKFold(5).split(F, y, groups)):
            folds[te] = k
    for group_name, cols in list(FEATURE_GROUPS.items()) + [("tumu", ALL_FEATURES)]:
        aucs, last = [], None
        for seed in config.SEEDS:
            proba = oof_predict(F[cols], y, folds, seed)
            aucs.append(auc_score(y, proba))
            last = proba
        lo, hi = bootstrap_ci(y, last, groups=groups, n_boot=500)
        rows.append({"veri": "beyin MR (Cheng)", "hedef": "tümör tipi (3 sınıf)", "roi": "tümör maskesi",
                     "oznitelik_grubu": group_name, "auc_ort": np.mean(aucs), "auc_std": np.std(aucs),
                     "ci95_alt": lo, "ci95_ust": hi, "n": len(y), "hasta_bazli_kat": True})
        print(f"[beyin] {group_name:13s} makro AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f} "
              f"(%95 GA {lo:.3f}-{hi:.3f})", flush=True)
    # hangi öznitelik ne kadar sızdırıyor: tüm öznitelikli modelde permütasyon önemi (tek kat üzerinde)
    first_fold = np.unique(folds)[0]  # resmi katlar 1..5 ile numaralı
    tr, te = folds != first_fold, folds == first_fold
    model = gbm(0).fit(F[tr], y[tr])
    pi = permutation_importance(model, F[te], y[te], scoring="roc_auc_ovr", n_repeats=10, random_state=0)
    for name, mean, std in zip(F.columns, pi.importances_mean, pi.importances_std):
        importances.append({"veri": "beyin MR (Cheng)", "oznitelik": name, "auc_dususu": mean, "std": std})
    # sınıf bazında tek tek (bire karşı hepsi) AUC, tüm özniteliklerle
    proba = oof_predict(F[ALL_FEATURES], y, folds, 0)
    per_class = {meta.label_name[meta.label - 1 == c].iloc[0]: auc_score((y == c).astype(int), proba[:, c])
                 for c in range(3)}
    return per_class


def covidqu_features():
    man = pd.read_csv(config.DATA_PROC / "covidqu_manifest.csv")
    man = man[man.lung_mask_path.notna() & (man.lung_mask_path != "")].reset_index(drop=True)

    def one(row):
        m = load_mask(row.lung_mask_path, MASK_RES)
        return mask_features(m, row.height, row.width)

    with ThreadPoolExecutor(8) as ex:
        feats = list(ex.map(one, man.itertuples(index=False)))
    return man, pd.DataFrame(feats)


def run_covidqu(rows, importances):
    man, F = covidqu_features()
    labels = sorted(man.label.unique())
    y_all = man.label.map({l: i for i, l in enumerate(labels)}).to_numpy()
    train = man.split.isin(["Train", "Val"]).to_numpy()
    test = (man.split == "Test").to_numpy()
    cols_no_size = [c for c in ALL_FEATURES if c not in FEATURE_GROUPS["girdi_boyutu"]]  # tüm görüntüler 256×256
    tasks = [("teşhis (3 sınıf)", labels)] + [(f"{a} / {b}", [a, b]) for a, b in
                                              [("Normal", "Non-COVID"), ("Normal", "COVID-19"),
                                               ("Non-COVID", "COVID-19")]]
    for task_name, task_labels in tasks:
        sel = man.label.isin(task_labels).to_numpy()
        y = man.label[sel].map({l: i for i, l in enumerate(task_labels)}).to_numpy()
        tr, te = train[sel], test[sel]
        for group_name, cols in [(g, c) for g, c in FEATURE_GROUPS.items() if g != "girdi_boyutu"] + \
                                [("tumu", cols_no_size)]:
            aucs, last = [], None
            for seed in config.SEEDS:
                model = gbm(seed).fit(F.loc[sel, cols][tr], y[tr])
                proba = model.predict_proba(F.loc[sel, cols][te])
                aucs.append(auc_score(y[te], proba))
                last = proba
            lo, hi = bootstrap_ci(y[te], last, n_boot=500)
            rows.append({"veri": "akciğer grafisi (COVID-QU-Ex)", "hedef": task_name, "roi": "akciğer maskesi",
                         "oznitelik_grubu": group_name, "auc_ort": np.mean(aucs), "auc_std": np.std(aucs),
                         "ci95_alt": lo, "ci95_ust": hi, "n": int(te.sum()), "hasta_bazli_kat": False})
            print(f"[covidqu] {task_name:22s} {group_name:9s} AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}",
                  flush=True)
    model = gbm(0).fit(F.loc[train, cols_no_size], y_all[train])
    pi = permutation_importance(model, F.loc[test, cols_no_size], y_all[test], scoring="roc_auc_ovr",
                                n_repeats=5, random_state=0)
    for name, mean, std in zip(cols_no_size, pi.importances_mean, pi.importances_std):
        importances.append({"veri": "akciğer grafisi (COVID-QU-Ex)", "oznitelik": name, "auc_dususu": mean, "std": std})
    # uç durum: ROI = enfeksiyon (lezyon) maskesi ise ROI'nin varlığı/boyutu
    inf = man[man.inf_mask_path.notna() & (man.inf_mask_path != "")].reset_index(drop=True)
    if len(inf):
        area = np.array([load_mask(p, MASK_RES).mean() for p in inf.inf_mask_path])
        yb = (inf.label == "COVID-19").astype(int).to_numpy()
        rows.append({"veri": "akciğer grafisi (COVID-QU-Ex, enfeksiyon alt kümesi)", "hedef": "COVID-19 / diğer",
                     "roi": "enfeksiyon (lezyon) maskesi", "oznitelik_grubu": "yalnızca ROI alanı",
                     "auc_ort": auc_score(yb, area), "auc_std": 0.0, "ci95_alt": np.nan, "ci95_ust": np.nan,
                     "n": len(yb), "hasta_bazli_kat": False})


def run_kaggle_size(rows):
    man = pd.read_csv(config.DATA_PROC / "kaggle_cxr_manifest.csv")
    for a, b in [("NORMAL", "PNEUMONIA"), ("NORMAL", "COVID19")]:
        sel = man.label.isin([a, b]).to_numpy()
        y = (man.label[sel] == b).astype(int).to_numpy()
        F = pd.DataFrame({"img_h": man.height[sel], "img_w": man.width[sel],
                          "img_aspect": man.width[sel] / man.height[sel]})
        tr, te = (man.split[sel] == "train").to_numpy(), (man.split[sel] == "test").to_numpy()
        aucs = [auc_score(y[te], gbm(s).fit(F[tr], y[tr]).predict_proba(F[te])) for s in config.SEEDS]
        rows.append({"veri": "akciğer grafisi (Kaggle, orijinal çözünürlük)", "hedef": f"{a} / {b}",
                     "roi": "(ROI'den bağımsız)", "oznitelik_grubu": "girdi_boyutu", "auc_ort": np.mean(aucs),
                     "auc_std": np.std(aucs), "ci95_alt": np.nan, "ci95_ust": np.nan, "n": int(te.sum()),
                     "hasta_bazli_kat": False})
        print(f"[kaggle] {a}/{b} yalnızca girdi boyutu AUC = {np.mean(aucs):.3f}", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["brain", "covidqu", "kaggle"])
    args = ap.parse_args()
    rows, importances, per_class = [], [], {}
    if "brain" in args.datasets:
        per_class = run_brain(rows, importances)
    if "covidqu" in args.datasets:
        run_covidqu(rows, importances)
    if "kaggle" in args.datasets:
        run_kaggle_size(rows)
    df = pd.DataFrame(rows)
    out_csv = config.TABLES / "saldiri_A_meta_veri.csv"
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        old = old[~old["veri"].isin(df["veri"].unique())]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(config.TABLES / "saldiri_A_meta_veri.csv", index=False)
    write_markdown_table(df, config.TABLES / "saldiri_A_meta_veri.md", floatfmt="{:.3f}")
    imp = pd.DataFrame(importances).sort_values(["veri", "auc_dususu"], ascending=[True, False])
    imp.to_csv(config.TABLES / "saldiri_A_oznitelik_onemi.csv", index=False)
    with open(config.TABLES / "saldiri_A_beyin_sinif_auc.json", "w", encoding="utf-8") as f:
        json.dump(per_class, f, ensure_ascii=False, indent=2)
    print(df.round(3).to_string(index=False))
    print("beyin sınıf bazında AUC:", {k: round(v, 3) for k, v in per_class.items()})


if __name__ == "__main__":
    main()
