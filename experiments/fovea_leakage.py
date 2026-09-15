"""Adım 4 (çözüm): sızıntı denetimi. Sunucunun gördüğü her şeye saldırı; FoveaHE'de sonuç şans düzeyinde olmalı.

Sınıflandırıcı Saldırı A ile aynıdır (`experiments.attack_metadata.gbm`, 5 tohum). Katlar sınıf oranı dengelidir,
beyinde ayrıca hasta bazlıdır (StratifiedGroupKFold; COVID-QU-Ex'te StratifiedKFold). Ana ölçü kat ortalaması AUC'dir:
bilgisiz özniteliklerde tam 0.5 verir. Havuzlanmış kat dışı AUC yalnızca ek sütundur; eğitim katlarının sınıf öncülleri
test katıyla ters ilişkili olduğundan küçük örneklemde 0.5'in altına sapar (hızlı denemede 0.27-0.40 görüldü).
Şans düzeyi etiket permütasyonuyla sınanır: boş dağılımın %95'lik değeri ve tek yönlü p değeri.

Satırlar (her FoveaHE ölçümünün yanında aynı örneklem ve protokolle Π_ROI pozitif kontrolü):
- Π_ROI, paket meta verisi: ROI piksellerini taşıyan ciphertext sayısı ve yükleme boyutu (orijinal çözünürlükte
  maskeden). ROI şekli gizlense bile paket sayısı ROI boyutunu sızdırır. Π_ROI sunucu süresi ciphertext sayısıyla
  doğrusal olduğundan (`piroi_maliyet.csv`) süre kanalı bu özniteliklere indirgenir.
- FoveaHE, paket meta verisi: gerçek şifrelemede ölçülen ciphertext ve slot sayısı, yükleme baytı.
- FoveaHE, yan kanal: gerçek şifreli çıkarımda sunucu süresi ve indirme baytı; görüntüler sınıflar arasında karışık
  sırada işlenir. Süre ölçümü için makine boşken çalıştırılmalı.
Referans satırları (saldırı kısmının kesin tablolarından): Π_ROI Saldırı A (ROI meta verisi) ve Saldırı B (açık bağlam);
FoveaHE Saldırı B tanımsal olarak uygulanamaz; akıl sağlığı: Adım 1 "sabit" (bilgisiz girdi).
Çıktılar: results/tables/cozum_sizinti.csv|md (sızıntı fonksiyonu tablosu ve IND-CPA-D kuralıyla),
cozum_sizinti_olcum.csv (görüntü başına ölçümler).

Çalıştırma: .venv\\Scripts\\python -m experiments.fovea_leakage [--dataset brain covidqu] [--meta-per-class 1000]
            [--timing-per-class 150] [--permutations 100] [--config F64_G32] [--quick]
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

import config
from common.evaluation import auc_score
from common.images import load_mask
from experiments.attack_metadata import gbm
from experiments.fovea_cost import checkpoint
from experiments.fovea_info import fill_fold_auc
from experiments.fovea_models import VectorData
from foveahe.data import Dataset, load_dataset
from foveahe.he_infer import FoveaHEInference
from foveahe.he_models import load_weights
from foveahe.representation import SLOTS, FoveaSpec
from he.piroi import make_context

LOG = config.LOGS / "fovea_leakage.log"
HIDDEN = 20

LEAKAGE_FUNCTIONS = [
    ("Π_ROI (ePrint 2026/103)", "model ağırlıkları; ROI dışındaki tüm açık pikseller; ROI konumu, boyutu ve şekli; "
                                "girdi boyutu; ROI ciphertext sayısı ve yükleme boyutu (ROI alanıyla değişir)"),
    ("Encrypt What Matters (arXiv 2609.09357)", "ROI dışı açık bölge ve ROI yerleşimi (ROI dışı açık kabul edilir)"),
    ("Bi-CryptoNets (arXiv 2402.01296)", "hassas olmayan kısım (gürültü eklenmiş ama açık)"),
    ("FoveaHE (odaklı tam şifreleme)", "yalnızca herkese açık sabitler: CKKS parametreleri (N = 16384), slot sayısı "
                                       "(8.192), ciphertext sayısı (1), model mimarisi; her hasta için aynı"),
]


def log(msg: str):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def balanced_subset(ds: Dataset, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = []
    for c in np.unique(ds.y):
        pool = np.flatnonzero(ds.y == c)
        parts.append(rng.choice(pool, min(per_class, len(pool)), replace=False))
    return np.sort(np.concatenate(parts))


def folds_for(ds: Dataset, idx: np.ndarray) -> np.ndarray:
    """Sınıf oranı dengeli 5 kat; beyinde hasta bazlı."""
    y = ds.y[idx]
    if ds.name == "brain":
        splits = StratifiedGroupKFold(5, shuffle=True, random_state=0).split(idx, y, ds.groups[idx])
    else:
        splits = StratifiedKFold(5, shuffle=True, random_state=0).split(idx, y)
    folds = np.zeros(len(idx), dtype=int)
    for k, (_, te) in enumerate(splits):
        folds[te] = k
    return folds


def fold_auc(X: np.ndarray, y: np.ndarray, folds: np.ndarray, seed: int) -> tuple[float, np.ndarray]:
    """(kat ortalaması AUC, kat dışı olasılıklar). Tüm sınıfları içermeyen test katı ortalamaya girmez."""
    n_cls = len(np.unique(y))
    proba = np.zeros((len(y), n_cls))
    per = []
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        model = gbm(seed).fit(X[tr], y[tr])
        p = np.zeros((int(te.sum()), n_cls))
        p[:, model.classes_] = model.predict_proba(X[te])  # eğitim katında eksik sınıf olsa da sütunlar doğru
        proba[te] = p
        if len(np.unique(y[te])) == n_cls:
            per.append(auc_score(y[te], p))
    return float(np.mean(per)), proba


def audit(X: np.ndarray, y: np.ndarray, folds: np.ndarray, n_perm: int) -> dict:
    means, pooled = [], []
    for seed in config.SEEDS:
        m, proba = fold_auc(X, y, folds, seed)
        means.append(m)
        pooled.append(auc_score(y, proba))
    rng = np.random.default_rng(0)
    null = np.array([fold_auc(X, rng.permutation(y), folds, 0)[0] for _ in range(n_perm)])
    obs = float(np.mean(means))
    p95 = float(np.percentile(null, 95))
    return {"auc_kat_ort": obs, "auc_std": float(np.std(means)), "auc_havuz": float(np.mean(pooled)),
            "bos_p95": p95, "p_degeri": float((1 + np.sum(null >= obs)) / (1 + n_perm)),
            "sans_duzeyinde": "evet" if obs <= p95 else "hayır"}


def piroi_packet_features(ds: Dataset, idx: np.ndarray, ct_mb: float) -> pd.DataFrame:
    """Π_ROI'de sunucunun gördüğü paket bilgisi: ROI piksellerini taşıyan ciphertext sayısı ve yükleme boyutu."""
    def one(i):
        return int(load_mask(ds.mask_paths[i]).sum())

    with ThreadPoolExecutor(8) as ex:
        roi = np.array(list(ex.map(one, idx)))
    n_ct = np.ceil(roi / SLOTS).astype(int)
    return pd.DataFrame({"piroi_ciphertext": n_ct, "piroi_yukleme_MB": n_ct * ct_mb, "piroi_roi_piksel": roi})


def fovea_measure(infer: FoveaHEInference, data: VectorData, idx: np.ndarray, run_server: bool, seed: int):
    """Görüntü başına gerçek şifreleme (ve istenirse sunucu); sıra sınıflar arasında karışık."""
    recs = [None] * len(idx)
    for j in np.random.default_rng(seed).permutation(len(idx)):
        x = data.get([idx[j]]).double().numpy()[0]
        t0 = time.perf_counter()
        enc = infer.encrypt(x)
        rec = {"sifreleme_s": time.perf_counter() - t0, "fovea_ciphertext": len(enc),
               "fovea_slot": int(sum(v.size() for v in enc)),
               "fovea_yukleme_bayt": sum(len(v.serialize()) for v in enc)}
        if run_server:
            t0 = time.perf_counter()
            outs = infer.server(enc)
            rec["fovea_sunucu_s"] = time.perf_counter() - t0
            rec["fovea_indirme_bayt"] = sum(len(o.serialize()) for o in outs)
        recs[j] = rec
    return pd.DataFrame(recs)


def load_infer(ctx, ds: Dataset, cfg: str, tag: str) -> tuple[FoveaHEInference, str]:
    """Adım 2'nin D ağırlıkları varsa onlar, yoksa aynı boyutta rastgele ağırlıklar (CKKS süresi değerden bağımsız)."""
    spec = FoveaSpec.parse(cfg)
    path = checkpoint(ds.name, "D", cfg, tag)
    if path.exists():
        return FoveaHEInference(ctx, load_weights(path)), path.name
    rng = np.random.default_rng(0)
    n_cls = len(ds.labels)
    w = {"kind": "D", "W1": rng.normal(0, 1 / np.sqrt(spec.n_values), (HIDDEN, spec.n_values)), "b1": np.zeros(HIDDEN),
         "W2": rng.normal(0, 1 / np.sqrt(HIDDEN), (n_cls, HIDDEN)), "b2": np.zeros(n_cls)}
    return FoveaHEInference(ctx, w), "rastgele ağırlık"


def reference_rows(ds: Dataset) -> list[dict]:
    """Saldırı kısmının kesin sonuçları (Π_ROI) ve Adım 1 bilgisiz girdi satırı; AUC kendi tablolarındaki ölçüdür."""
    rows = []
    a = config.TABLES / "saldiri_A_meta_veri.csv"
    if a.exists():
        t = pd.read_csv(a)
        hedef = "tümör tipi (3 sınıf)" if ds.name == "brain" else "teşhis (3 sınıf)"
        r = t[(t.veri == ds.display) & (t.hedef == hedef) & (t.oznitelik_grubu == "tumu")]
        if len(r):
            rows.append({"yontem": "Π_ROI", "saldiri": "A: ROI meta verisi", "oznitelikler": "konum, boyut, şekil",
                         "n": int(r.n.iloc[0]), "auc": float(r.auc_ort.iloc[0]),
                         "kaynak": "saldiri_A_meta_veri.csv (5 tohum, havuzlanmış kat dışı)"})
    b = config.TABLES / "saldiri_B_baglam.csv"
    if b.exists():
        t = pd.read_csv(b)
        r = t[(t.veri == ds.display) & (t.gorus == "baglam")]
        if len(r):
            rows.append({"yontem": "Π_ROI", "saldiri": "B: açık bağlam (ROI gizli)", "oznitelikler": "açık pikseller",
                         "n": int(r.n.iloc[0]), "auc": float(r.auc.mean()),
                         "kaynak": "saldiri_B_baglam.csv (5 tohum ortalaması)"})
    rows.append({"yontem": "FoveaHE", "saldiri": "B: açık bağlam", "oznitelikler": "açık piksel yok", "n": np.nan,
                 "auc": np.nan, "kaynak": "tanımsal olarak uygulanamaz: sunucu yalnızca ciphertext görür"})
    info = config.TABLES / "cozum_bilgi.csv"
    if info.exists():
        t = fill_fold_auc(pd.read_csv(info), "")
        r = t[(t.veri == ds.display) & (t.yapilandirma == "sabit")]
        if len(r):
            value = r.auc_kat_ort.iloc[0] if pd.notna(r.auc_kat_ort.iloc[0]) else r.auc.iloc[0]
            rows.append({"yontem": "akıl sağlığı", "saldiri": "sunucu görüşü sabit görüntü", "oznitelikler": "bilgisiz girdi",
                         "n": int(r.n.iloc[0]), "auc": float(value),
                         "kaynak": "cozum_bilgi.csv, Adım 1 sabit (beyinde kat ortalaması)"})
    return rows


def run_dataset(ctx, name: str, cfg: str, meta_per_class: int, timing_per_class: int, n_perm: int, tag: str,
                rows: list, refs: list, measures: list):
    ds = load_dataset(name)
    data = VectorData(ds, FoveaSpec.parse(cfg))
    infer, weights_src = load_infer(ctx, ds, cfg, tag)
    log(f"[{name}] FoveaHE {cfg}, ağırlıklar: {weights_src}")
    refs += [{"veri": ds.display, **r} for r in reference_rows(ds)]

    def add(method, attack, feats, X, y, folds):
        res = audit(X.to_numpy(dtype=float), y, folds, n_perm)
        rows.append({"veri": ds.display, "yontem": method, "saldiri": attack, "oznitelikler": feats, "n": len(y), **res})
        log(f"[{name}] {method:8s} {attack} [{feats}]: kat ort AUC={res['auc_kat_ort']:.4f} ± {res['auc_std']:.4f}, "
            f"boş %95={res['bos_p95']:.4f}, p={res['p_degeri']:.3f}, havuz={res['auc_havuz']:.4f}")

    # Paket meta verisi: Π_ROI (maskeden) ve FoveaHE (gerçek şifreleme), aynı örneklem ve katlar
    idx = np.arange(len(ds.y)) if meta_per_class <= 0 else balanced_subset(ds, meta_per_class, seed=0)
    folds, y = folds_for(ds, idx), ds.y[idx]
    t0 = time.perf_counter()
    fov = fovea_measure(infer, data, idx, run_server=False, seed=1)
    ct_mb = float(fov.fovea_yukleme_bayt.median()) / 1e6
    pir = piroi_packet_features(ds, idx, ct_mb)
    log(f"[{name}] paket meta verisi: {len(idx)} görüntü ({time.perf_counter() - t0:.0f} s); FoveaHE yükleme "
        f"{fov.fovea_yukleme_bayt.min()}-{fov.fovea_yukleme_bayt.max()} bayt; Π_ROI ciphertext "
        f"{pir.piroi_ciphertext.min()}-{pir.piroi_ciphertext.max()}")
    add("Π_ROI", "A: paket meta verisi", "ciphertext sayısı, yükleme boyutu",
        pir[["piroi_ciphertext", "piroi_yukleme_MB"]], y, folds)
    add("FoveaHE", "A: paket meta verisi", "ciphertext sayısı, slot sayısı, yükleme baytı",
        fov[["fovea_ciphertext", "fovea_slot", "fovea_yukleme_bayt"]], y, folds)
    measures.append(pd.concat([pd.DataFrame({"veri": ds.display, "satir": idx, "etiket": y, "kat": folds,
                                             "olcum": "paket"}), fov, pir], axis=1))

    # Yan kanal: gerçek sunucu süresi ve indirme baytı; aynı alt kümede Π_ROI paket bilgisi pozitif kontrol
    idx = balanced_subset(ds, timing_per_class, seed=2)
    folds, y = folds_for(ds, idx), ds.y[idx]
    t0 = time.perf_counter()
    fov = fovea_measure(infer, data, idx, run_server=True, seed=3)
    pir = piroi_packet_features(ds, idx, ct_mb)
    log(f"[{name}] yan kanal: {len(idx)} görüntü ({time.perf_counter() - t0:.0f} s); sunucu süresi medyan "
        f"{fov.fovea_sunucu_s.median():.3f} s (IQR {fov.fovea_sunucu_s.quantile(0.25):.3f}-"
        f"{fov.fovea_sunucu_s.quantile(0.75):.3f})")
    add("Π_ROI", "yan kanal (alt küme)", "ciphertext sayısı, yükleme boyutu (süre bunlarla doğrusal)",
        pir[["piroi_ciphertext", "piroi_yukleme_MB"]], y, folds)
    add("FoveaHE", "yan kanal (alt küme)", "sunucu süresi, indirme baytı",
        fov[["fovea_sunucu_s", "fovea_indirme_bayt"]], y, folds)
    add("FoveaHE", "yan kanal (alt küme)", "yalnızca sunucu süresi", fov[["fovea_sunucu_s"]], y, folds)
    measures.append(pd.concat([pd.DataFrame({"veri": ds.display, "satir": idx, "etiket": y, "kat": folds,
                                             "olcum": "yan_kanal"}), fov, pir], axis=1))


def write_report(df: pd.DataFrame, refs: pd.DataFrame, path):
    def fmt(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else f"{v:.4f}"
        return str(v)

    def table(frame, cols):
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        return out + ["| " + " | ".join(fmt(r.get(c, np.nan)) for c in cols) + " |" for r in frame.to_dict("records")]

    lines = ["# Çözüm Adım 4: sızıntı denetimi", "",
             "Ana ölçü kat ortalaması saldırgan AUC'si (5 tohum; sınıf dengeli katlar, beyinde hasta bazlı). "
             "`bos_p95`: etiket permütasyonuyla boş dağılımın %95'lik değeri; `sans_duzeyinde = evet` gözlenen AUC "
             "bu değeri aşmıyor demektir. Π_ROI satırları aynı örneklem ve protokolle pozitif kontroldür.", ""]
    lines += table(df, ["veri", "yontem", "saldiri", "oznitelikler", "n", "auc_kat_ort", "auc_std", "bos_p95",
                        "p_degeri", "sans_duzeyinde", "auc_havuz"])
    lines += ["", "## Referanslar (kesin tablolardan)", ""]
    lines += table(refs, ["veri", "yontem", "saldiri", "oznitelikler", "n", "auc", "kaynak"])
    lines += ["", "## Sızıntı fonksiyonları: sunucunun öğrendiği", "", "| Yöntem | Sunucunun öğrendiği |", "|---|---|"]
    lines += [f"| {m} | {s} |" for m, s in LEAKAGE_FUNCTIONS]
    lines += ["", "## Protokol kuralı (IND-CPA-D)", "",
              "- Çözülen sonuç (logit, olasılık ya da karar) sunucuya geri gönderilmez; sunucuya çözme kâhini verilmez.",
              "- Gerekçe: CKKS'de çözülmüş sonuçlara erişen sunucu gizli anahtarı kurtarabilir (IND-CPA-D; bu projedeki "
              "PoC `pilot/indcpad_poc_tenseal.py`, TenSEAL ile 0.15 s).",
              "- Model ağırlıkları sunucuda açık metindir (Π_ROI ile aynı tehdit modeli); model gizliliği kapsam dışıdır."]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="*", default=["brain", "covidqu"])
    ap.add_argument("--config", default="F64_G32")
    ap.add_argument("--meta-per-class", type=int, default=1000, help="beyinde her zaman tüm veri (3.064)")
    ap.add_argument("--timing-per-class", type=int, default=150)
    ap.add_argument("--permutations", type=int, default=100, help="p değeri çözünürlüğü 1/(1+permütasyon)")
    ap.add_argument("--quick", action="store_true", help="hızlı ağırlıklar, sınıf başına 20 (paket) ve 8 (süre)")
    args = ap.parse_args()
    tag = "_hizli" if args.quick else ""
    n_perm = 5 if args.quick else args.permutations
    ctx = make_context()
    rows, refs, measures = [], [], []
    for name in args.dataset:
        meta_pc = 20 if args.quick else (0 if name == "brain" else args.meta_per_class)
        run_dataset(ctx, name, args.config, meta_pc, 8 if args.quick else args.timing_per_class, n_perm, tag, rows,
                    refs, measures)
    df, ref_df = pd.DataFrame(rows), pd.DataFrame(refs)
    df.to_csv(config.TABLES / f"cozum_sizinti{tag}.csv", index=False)
    ref_df.to_csv(config.TABLES / f"cozum_sizinti_referans{tag}.csv", index=False)
    write_report(df, ref_df, config.TABLES / f"cozum_sizinti{tag}.md")
    pd.concat(measures, ignore_index=True).to_csv(config.TABLES / f"cozum_sizinti_olcum{tag}.csv", index=False)
    print(df.round(4).to_string(index=False))
    print(ref_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
