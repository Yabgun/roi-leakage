"""Saldırı A (makale Tablo II) için görüntü başına tahminlerin kaydı.

Özgün betik (`experiments.attack_metadata`) yalnızca AUC tablosunu kaydeder; kesinlik, duyarlılık, F1 ve karışıklık
matrisi için görüntü başına tahmin gerekir. Bu betik aynı öznitelikleri, aynı sınıflandırıcıyı (HistGradientBoosting),
aynı bölmeleri ve aynı 5 tohumu kullanarak Tablo II'nin her satırının tahminlerini `results/preds/saldiri_A_*.npz`
dosyalarına yazar. Her satırda AUC ortalaması, standart sapması ve güven aralığı `saldiri_A_meta_veri.csv`'deki
değerle karşılaştırılır; tutmazsa betik hata verir. Özgün tablolar değiştirilmez.

Öznitelikler `data/processed/saldiri_A_oznitelikler_{beyin,covidqu}.csv` dosyalarına da yazılır; veri dağılımı
şekilleri ROI alanını buradan okur.

Donanım: yalnızca CPU; OpenMP iş parçacığı sayısı 4 ile sınırlıdır (bilgisayar kilitlenmesin diye).

Çalıştırma: .venv\\Scripts\\python -m analysis.saldiri_a_tahmin
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from common.evaluation import auc_score, bootstrap_ci  # noqa: E402
from experiments.attack_metadata import (ALL_FEATURES, FEATURE_GROUPS, brain_features, covidqu_features,  # noqa: E402
                                         gbm, oof_predict)

PREDS = config.RESULTS / "preds"
TOL = 1e-9


def kaydet(ad: str, seed: int, proba, y, idx, siniflar) -> None:
    np.savez(PREDS / f"saldiri_A_{ad}_s{seed}.npz", proba=np.asarray(proba, dtype=np.float64),
             y=np.asarray(y, dtype=np.int64), idx=np.asarray(idx, dtype=np.int64), siniflar=np.asarray(siniflar))


def denetle(ref: pd.DataFrame, veri: str, hedef: str, grup: str, aucs, ci) -> dict:
    """Yeniden hesaplanan değerleri özgün tablo satırıyla karşılaştırır."""
    row = ref[(ref.veri == veri) & (ref.hedef == hedef) & (ref.oznitelik_grubu == grup)]
    if len(row) != 1:
        raise RuntimeError(f"özgün tabloda satır bulunamadı: {veri} / {hedef} / {grup}")
    row = row.iloc[0]
    yeni = {"auc_ort": float(np.mean(aucs)), "auc_std": float(np.std(aucs)), "ci95_alt": ci[0], "ci95_ust": ci[1]}
    farklar = {}
    for k, v in yeni.items():
        eski = row[k]
        if pd.isna(eski) and (v is None or (isinstance(v, float) and np.isnan(v))):
            continue
        farklar[k] = abs(float(eski) - float(v))
    en_buyuk = max(farklar.values()) if farklar else 0.0
    durum = "aynı" if en_buyuk <= TOL else "FARKLI"
    print(f"  {veri[:24]:24s} | {hedef:18s} | {grup:13s} AUC {yeni['auc_ort']:.4f} (özgün {row.auc_ort:.4f}) "
          f"en büyük fark {en_buyuk:.1e} -> {durum}", flush=True)
    if durum != "aynı":
        raise RuntimeError(f"Saldırı A yeniden üretilemedi: {veri} / {hedef} / {grup}: {farklar}")
    return {"veri": veri, "hedef": hedef, "oznitelik_grubu": grup, **yeni, "ozgun_auc_ort": float(row.auc_ort),
            "en_buyuk_fark": en_buyuk}


def beyin(ref, rapor):
    meta, F = brain_features()
    F.insert(0, "id", meta["id"].to_numpy())
    F.to_csv(config.DATA_PROC / "saldiri_A_oznitelikler_beyin.csv", index=False)
    F = F.drop(columns="id")
    y = meta["label"].to_numpy() - 1
    groups, folds = meta["pid"].to_numpy(), meta["fold"].to_numpy()
    if (meta.groupby("pid")["fold"].nunique() > 1).any() or (folds < 0).any():
        raise RuntimeError("resmi katlar hasta bazlı değil; özgün betikteki yedek kat kuralı gerekir")
    siniflar = meta.groupby("label")["label_name"].first().tolist()
    for grup, cols in list(FEATURE_GROUPS.items()) + [("tumu", ALL_FEATURES)]:
        aucs, son = [], None
        for seed in config.SEEDS:
            proba = oof_predict(F[cols], y, folds, seed)
            kaydet(f"beyin_tumor_tipi_{grup}", seed, proba, y, np.arange(len(y)), siniflar)
            aucs.append(auc_score(y, proba))
            son = proba
        ci = bootstrap_ci(y, son, groups=groups, n_boot=500)
        rapor.append(denetle(ref, "beyin MR (Cheng)", "tümör tipi (3 sınıf)", grup, aucs, ci))


def covidqu(ref, rapor):
    man, F = covidqu_features()
    F.insert(0, "file", man["file"].to_numpy())
    F.insert(1, "split", man["split"].to_numpy())
    F.insert(2, "label", man["label"].to_numpy())
    F.to_csv(config.DATA_PROC / "saldiri_A_oznitelikler_covidqu.csv", index=False)
    F = F.drop(columns=["file", "split", "label"])
    train = man.split.isin(["Train", "Val"]).to_numpy()
    test = (man.split == "Test").to_numpy()
    cols_no_size = [c for c in ALL_FEATURES if c not in FEATURE_GROUPS["girdi_boyutu"]]
    labels = sorted(man.label.unique())
    gorevler = [("teşhis (3 sınıf)", labels, "teshis",
                 [(g, c) for g, c in FEATURE_GROUPS.items() if g != "girdi_boyutu"] + [("tumu", cols_no_size)]),
                ("Normal / Non-COVID", ["Normal", "Non-COVID"], "normal_pnomoni", [("tumu", cols_no_size)])]
    for hedef, task_labels, kisa, gruplar in gorevler:
        sel = man.label.isin(task_labels).to_numpy()
        y = man.label[sel].map({l: i for i, l in enumerate(task_labels)}).to_numpy()
        tr, te = train[sel], test[sel]
        idx_te = np.flatnonzero(sel)[te]
        for grup, cols in gruplar:
            aucs, son = [], None
            for seed in config.SEEDS:
                proba = gbm(seed).fit(F.loc[sel, cols][tr], y[tr]).predict_proba(F.loc[sel, cols][te])
                kaydet(f"covidqu_{kisa}_{grup}", seed, proba, y[te], idx_te, task_labels)
                aucs.append(auc_score(y[te], proba))
                son = proba
            ci = bootstrap_ci(y[te], son, n_boot=500)
            rapor.append(denetle(ref, "akciğer grafisi (COVID-QU-Ex)", hedef, grup, aucs, ci))


def kaggle(ref, rapor):
    man = pd.read_csv(config.DATA_PROC / "kaggle_cxr_manifest.csv")
    a, b = "NORMAL", "PNEUMONIA"
    sel = man.label.isin([a, b]).to_numpy()
    y = (man.label[sel] == b).astype(int).to_numpy()
    F = pd.DataFrame({"img_h": man.height[sel], "img_w": man.width[sel], "img_aspect": man.width[sel] / man.height[sel]})
    tr, te = (man.split[sel] == "train").to_numpy(), (man.split[sel] == "test").to_numpy()
    idx_te = np.flatnonzero(sel)[te]
    aucs = []
    for seed in config.SEEDS:
        proba = gbm(seed).fit(F[tr], y[tr]).predict_proba(F[te])
        kaydet("kaggle_normal_pnomoni_girdi_boyutu", seed, proba, y[te], idx_te, [a, b])
        aucs.append(auc_score(y[te], proba))
    rapor.append(denetle(ref, "akciğer grafisi (Kaggle, orijinal çözünürlük)", f"{a} / {b}", "girdi_boyutu", aucs,
                         (np.nan, np.nan)))


def main():
    t0 = time.perf_counter()
    ref = pd.read_csv(config.TABLES / "saldiri_A_meta_veri.csv")
    rapor = []
    print("[beyin] öznitelikler ve tahminler", flush=True)
    beyin(ref, rapor)
    print("[covidqu] öznitelikler ve tahminler", flush=True)
    covidqu(ref, rapor)
    print("[kaggle] yalnızca girdi boyutu", flush=True)
    kaggle(ref, rapor)
    pd.DataFrame(rapor).to_csv(config.TABLES / "saldiri_A_yeniden_uretim.csv", index=False)
    print(f"Tamam: {len(rapor)} satırın hepsi özgün tabloyla aynı ({time.perf_counter() - t0:.0f} s).", flush=True)


if __name__ == "__main__":
    main()
