"""İnceleme S1: beyin MR normalizasyonunun yapay sızıntı kanalı ve ham yoğunluk önbelleği.

Ön işleme (`experiments.prepare_data.prepare_brain`) her kesiti tüm piksellerin (tümör dahil) 0.5 ve 99.5 yüzdelikleriyle
[0, 255] aralığına ölçekler; bağlam saldırısında gizli bölge bu ölçeklemeden sonra sıfırlanır. T1 kontrastlı MR'da tümör
çoğu zaman kesitin en parlak dokularındandır; üst yüzdelik tümöre bağlıysa açık piksellerin ölçeği gizli tümörün
parlaklığını taşır. Gerçek sunucu bu kanalı göremez. Bu betik:
1) ham .mat kesitlerinden 224 px ham yoğunluk önbelleği üretir (`data/processed/brain_224_raw.npy`, float32, satır sırası
   meta.csv ile aynı). Bağlam saldırıları `--norm gorunur` ile bu önbelleği kullanır: yüzdelikler görüş oluşturulurken
   yalnızca görünür piksellerden hesaplanır (`attacks.context_cnn.VisibleNormCache`).
2) kanalın tek başına taşıdığı bilgiyi ölçer: kesit başına tüm piksellerden ve yalnızca tümör dışı piksellerden hesaplanan
   yüzdelikler; ölçek oranı (üst - alt, tüm pikseller) / (üst - alt, tümör dışı) ile tümör tipi AUC'si (hasta bazlı resmi
   katlar, Saldırı A ile aynı sınıflandırıcı).
3) sağlama: tüm piksellerle normalize edilen ham önbellek eski uint8 önbellekle örtüşüyor mu (korelasyon, ortalama fark).

Çıktılar: data/processed/brain_224_raw.npy, results/tables/normalizasyon_kanali.csv (kesit başına) ve .json (özet).
Çalıştırma: .venv\\Scripts\\python -m experiments.brain_visible_norm [--force]
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

import config
from common.evaluation import auc_score
from experiments.attack_context import RES, build_cache, log
from experiments.attack_metadata import oof_predict
from experiments.prepare_data import _read_cjdata

RAW = config.DATA_PROC / f"brain_{RES}_raw.npy"
Q = (0.5, 99.5)


def _one(args):
    row_id, label, path = args
    lab, _, img, mask = _read_cjdata(path)
    if lab != label:
        raise ValueError(f"{path}: etiket meta.csv ile uyuşmuyor ({lab} / {label})")
    lo_t, hi_t = np.percentile(img, Q)
    ctx, tum = img[~mask], img[mask]
    lo_b, hi_b = np.percentile(ctx, Q) if ctx.size else (lo_t, hi_t)
    small = np.asarray(Image.fromarray(img.astype(np.float32)).resize((RES, RES), Image.BILINEAR), dtype=np.float32)
    return small, {"id": row_id, "label": label, "alt_tum": float(lo_t), "ust_tum": float(hi_t), "alt_baglam": float(lo_b),
                   "ust_baglam": float(hi_b), "tumor_medyan": float(np.median(tum)) if tum.size else np.nan,
                   "tumor_ust_asan_oran": float((tum > hi_b).mean()) if tum.size else np.nan}


def build(force: bool = False) -> pd.DataFrame:
    stats_path = config.TABLES / "normalizasyon_kanali.csv"
    if RAW.exists() and stats_path.exists() and not force:
        return pd.read_csv(stats_path)
    meta = pd.read_csv(config.DATA_PROC / "brain" / "meta.csv")
    files = {int(p.stem): p for p in (config.BRAIN / "mat").rglob("*.mat") if p.stem.isdigit()}
    t0 = time.perf_counter()
    with ThreadPoolExecutor(8) as ex:
        out = list(ex.map(_one, [(int(r.id), int(r.label), files[int(r.id)]) for r in meta.itertuples(index=False)]))
    raw = np.stack([o[0] for o in out])
    np.save(RAW, raw)
    stats = pd.DataFrame([o[1] for o in out])
    stats["label_name"] = meta["label_name"].to_numpy()
    stats["fold"] = meta["fold"].to_numpy()
    stats["olcek_orani"] = (stats.ust_tum - stats.alt_tum) / (stats.ust_baglam - stats.alt_baglam).clip(lower=1e-6)
    stats.to_csv(stats_path, index=False)
    log(f"[S1] ham önbellek {raw.shape} ve kesit istatistikleri hazır ({time.perf_counter() - t0:.0f} s)")
    return stats


def load_raw() -> np.ndarray:
    if not RAW.exists():
        build()
    return np.load(RAW)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    stats = build(args.force)
    meta = pd.read_csv(config.DATA_PROC / "brain" / "meta.csv")
    raw = np.load(RAW)
    old, _ = build_cache("brain", [config.DATA_PROC / "brain" / p for p in meta.img_path],
                         [config.DATA_PROC / "brain" / p for p in meta.mask_path])

    # Sağlama: tüm piksellerle normalize edilmiş ham önbellek ~ eski uint8 önbellek (yeniden boyutlandırma sırası farkı)
    rng = np.random.default_rng(0)
    corr, diff = [], []
    for i in rng.choice(len(raw), 200, replace=False):
        lo, hi = np.percentile(raw[i], Q)
        new = np.clip((raw[i] - lo) / max(hi - lo, 1e-6), 0, 1) * 255
        corr.append(float(np.corrcoef(new.ravel(), old[i].ravel().astype(np.float64))[0, 1]))
        diff.append(float(np.abs(new - old[i]).mean()))

    y = stats["label"].to_numpy() - 1
    folds = stats["fold"].to_numpy()
    X = pd.DataFrame({"log_olcek_orani": np.log(stats["olcek_orani"].to_numpy())})
    auc_ratio = auc_score(y, oof_predict(X, y, folds, seed=0))
    X2 = pd.DataFrame({"log_olcek_orani": X.log_olcek_orani,
                       "log_ust_tum": np.log(stats.ust_tum.clip(lower=1)), "log_ust_baglam": np.log(stats.ust_baglam.clip(lower=1))})
    auc_ratio_scale = auc_score(y, oof_predict(X2, y, folds, seed=0))
    by_class = stats.groupby("label_name").agg(n=("id", "size"), olcek_orani_medyan=("olcek_orani", "median"),
                                               olcek_orani_ort=("olcek_orani", "mean"),
                                               oran_1yuzde_ustu=("olcek_orani", lambda s: float((s > 1.01).mean())),
                                               tumor_ust_asan_oran_ort=("tumor_ust_asan_oran", "mean"))
    summary = {
        "aciklama": "ölçek oranı = (99.5 - 0.5 yüzdelik, tüm pikseller) / (aynısı, yalnız tümör dışı pikseller); "
                    "1'den büyükse eski normalizasyonda açık piksellerin ölçeği gizli tümöre bağlı",
        "kesit": int(len(stats)),
        "olcek_orani_medyan": float(stats.olcek_orani.median()),
        "olcek_orani_1yuzde_ustu_oran": float((stats.olcek_orani > 1.01).mean()),
        "olcek_orani_5yuzde_ustu_oran": float((stats.olcek_orani > 1.05).mean()),
        "yalniz_olcek_orani_auc": auc_ratio,
        "olcek_orani_ve_ust_yuzdelikler_auc": auc_ratio_scale,
        "sinif_bazinda": by_class.round(4).reset_index().to_dict(orient="records"),
        "saglama_korelasyon_ort": float(np.mean(corr)), "saglama_korelasyon_min": float(np.min(corr)),
        "saglama_ortalama_mutlak_fark_0_255": float(np.mean(diff)),
    }
    with open(config.TABLES / "normalizasyon_kanali.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"[S1] ölçek oranı medyan {summary['olcek_orani_medyan']:.4f}; %1'den büyük: "
        f"{summary['olcek_orani_1yuzde_ustu_oran']:.3f}; yalnız oranla tümör tipi AUC={auc_ratio:.3f}; "
        f"oran + üst yüzdelikler AUC={auc_ratio_scale:.3f}; sağlama korelasyonu {np.mean(corr):.4f} (en düşük {np.min(corr):.4f})")
    print(by_class.round(4).to_string())


if __name__ == "__main__":
    main()
