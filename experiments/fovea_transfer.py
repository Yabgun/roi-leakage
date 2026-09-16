"""Özgünlük bonusu: harici kaynakta genelleme (Kaggle CXR, COVID-QU-Ex'te kopyası olanlar çıkarılmış).

Soru: COVID-QU-Ex'te eğitilmiş şifreli çalışabilen modeller başka bir derlemeden gelen göğüs grafilerinde doğruluğunu
koruyor mu; odaklı temsil bu açıdan tam görüntüden (Π_ROI'nin doğruluğu) ve şifreli özetten (Adım 5a) farklı mı?
Yeniden eğitim yok: Adım 2 ve 5a'nın kayıtlı ağırlıkları (tohum başına resmi test modeli) doğrudan uygulanır; çıkarım
şifreli hesabın şifresiz başvurusudur (`forward_numpy`, `forward_numpy_c`; Adım 3'te şifreliyle AUC farkı 0).

Veri: Kaggle "Chest X-ray (COVID-19 & Pneumonia)" train + test. dHash + korelasyonla COVID-QU-Ex'te kopyası bulunan
4.665 görüntü çıkarıldıktan sonra 1.767 görüntü (PNEUMONIA 1.157, NORMAL 340, COVID19 270). Sınıf eşlemesi:
COVID19 → COVID-19, PNEUMONIA → Non-COVID, NORMAL → Normal. ROI: `experiments.lung_segmenter` akciğer maskeleri
(etiketten bağımsız, 224 px; 256 px'e en yakın komşuyla büyütülür).
Temsil: görüntü COVID-QU-Ex gibi 256×256 gri tona indirilir (`common.images.load_gray`), sonra `foveahe.data.build_layers`
ile aynı işlem: `roi_geometry`, `extract`, uint8 niceleme. Şifreli özet: aynı 224 px önbellek ve dondurulmuş ResNet-18.
Ölçüt: makro one-vs-rest AUC (üç sınıf) ve sınıf çiftleri; iç AUC (COVID-QU-Ex Test) aynı tohumların Adım 2 / 5a
satırlarından.
Uyarı: kopyalar çıkarılsa da kalan görüntüler COVID-QU-Ex'in derlendiği kaynaklarla ortak kökenli olabilir; bu, tam
bağımsız bir dış doğrulama değil, derleme kaymasına karşı sağlamlık denetimidir.

Çıktılar: results/tables/cozum_genelleme.csv (tohum satırları), cozum_genelleme.md (özet),
results/figures/cozum_genelleme.png.
Çalıştırma: .venv\\Scripts\\python -m experiments.fovea_transfer [--seeds 0 1 2 3 4] [--device cuda] [--quick]
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

import config
from common.evaluation import auc_score, bootstrap_ci
from common.images import load_gray, load_mask
from common.report import write_markdown_table
from experiments.attack_context import build_cache
from experiments.fovea_baselines import _MEAN, _STD, METHOD, encoder
from experiments.fovea_cost import softmax
from experiments.fovea_models import read_table
from foveahe.data import DISPLAY, STORE, load_dataset
from foveahe.he_cnn import forward_numpy_c
from foveahe.he_models import forward_numpy, load_weights
from foveahe.representation import FoveaSpec, extract, roi_geometry

LOG = config.LOGS / "fovea_transfer.log"
CONFIGS = ["U256", "U64", "F32_G16", "F64_G32"]
MODELS = ["D", "D2", "C"]
KAGGLE_TO_CQ = {"COVID19": "COVID-19", "PNEUMONIA": "Non-COVID", "NORMAL": "Normal"}
PAIRS = [("Normal", "Non-COVID"), ("Normal", "COVID-19"), ("Non-COVID", "COVID-19")]
SIZE = 256
CACHE = STORE / "kaggle_capraz"
SERIES = {"ic": "#2a78d6", "dis": "#eb6834"}  # referans paletin ilk iki kategorik yuvası (fovea_info ile aynı)
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
       "surface": "#fcfcfb"}


def log(msg: str):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def method_of(cfg: str) -> str:
    if cfg == "U256":
        return "tam görüntü (Π_ROI doğruluğu)"
    return "eş örnekli küçültme" if cfg.startswith("U") else "FoveaHE"


def kaggle_rows(labels, quick: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(tüm manifesto, kopyasız ve maskeli değerlendirme satırları)."""
    full = pd.read_csv(config.DATA_PROC / "kaggle_cxr_manifest.csv")
    dup = pd.read_csv(config.DATA_PROC / "cxr_capraz_tekrarlar.csv")
    full["kaggle_idx"] = np.arange(len(full))
    keep = (~full.kaggle_idx.isin(dup.kaggle_idx.unique()) & full.lung_mask_path.notna()
            & full.label.isin(list(KAGGLE_TO_CQ)))
    kg = full[keep].reset_index(drop=True)
    if quick:
        rng = np.random.default_rng(0)
        pick = np.concatenate([rng.choice(np.flatnonzero(kg.label == l), 40, replace=False) for l in KAGGLE_TO_CQ])
        kg = kg.iloc[np.sort(pick)].reset_index(drop=True)
    kg["y"] = kg.label.map(KAGGLE_TO_CQ).map({l: i for i, l in enumerate(labels)}).astype(int)
    return full, kg


def _load(pair):
    img_path, mask_path = pair
    img = np.round(load_gray(img_path, SIZE) * 255).astype(np.uint8)  # COVID-QU-Ex gibi 256 px uint8
    return img, load_mask(mask_path, SIZE)


@torch.no_grad()
def kaggle_layers(kg: pd.DataFrame, keys, device: str, tag: str) -> tuple[dict, np.ndarray]:
    path = CACHE / f"katmanlar{tag}.npz"
    if path.exists():
        with np.load(path) as z:
            if np.array_equal(z["kaggle_idx"], kg.kaggle_idx.to_numpy()) and all(k in z.files for k in keys):
                return {k: z[k] for k in keys}, z["geom"]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(8) as ex:
        pairs = list(ex.map(_load, zip(kg.img_path, kg.lung_mask_path)))
    parts, geoms = {k: [] for k in keys}, []
    for s in range(0, len(pairs), 256):
        chunk = pairs[s:s + 256]
        img = torch.from_numpy(np.stack([p[0] for p in chunk])).to(device).float().div_(255).unsqueeze(1)
        g = roi_geometry(torch.from_numpy(np.stack([p[1] for p in chunk])).to(device))
        geoms.append(g.cpu().numpy())
        for k in keys:
            lay = extract(img, g, k)[:, 0].mul_(255).round_().clamp_(0, 255)
            parts[k].append(lay.to(torch.uint8).cpu().numpy())
    layers = {k: np.concatenate(v) for k, v in parts.items()}
    geom = np.concatenate(geoms).astype(np.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, kaggle_idx=kg.kaggle_idx.to_numpy(), geom=geom, **layers)
    log(f"[genelleme] Kaggle katmanları: {len(kg)} görüntü, {' '.join(keys)} ({time.perf_counter() - t0:.0f} s)")
    return layers, geom


@torch.no_grad()
def kaggle_embeddings(full: pd.DataFrame, kg: pd.DataFrame, device: str, tag: str) -> np.ndarray:
    path = CACHE / f"ozet_r18{tag}.npz"
    if path.exists():
        with np.load(path) as z:
            if np.array_equal(z["kaggle_idx"], kg.kaggle_idx.to_numpy()):
                return z["emb"]
    t0 = time.perf_counter()
    imgs, _ = build_cache("kaggle", full.img_path, full.lung_mask_path)  # Saldırı B ile aynı 224 px önbellek
    sel = imgs[kg.kaggle_idx.to_numpy()]
    net = encoder(device)
    emb = np.zeros((len(sel), 512), dtype=np.float32)
    for i in range(0, len(sel), 128):
        x = torch.from_numpy(sel[i:i + 128]).to(device).float().div_(255).unsqueeze(1).expand(-1, 3, -1, -1)
        emb[i:i + 128] = net((x - _MEAN.to(device)) / _STD.to(device)).float().cpu().numpy()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, kaggle_idx=kg.kaggle_idx.to_numpy(), emb=emb)
    log(f"[genelleme] Kaggle özetleri: {emb.shape} ({time.perf_counter() - t0:.0f} s)")
    return emb


def fovea_logits(w: dict, layers: dict, geom: np.ndarray, spec: FoveaSpec, chunk: int = 256) -> np.ndarray:
    """`fovea_models.VectorData.get` ile aynı vektör ([katmanlar/255, geometri]) üzerinde şifresiz başvuru çıkarımı."""
    forward = forward_numpy_c if w["kind"] == "C" else forward_numpy
    out = []
    for s in range(0, len(geom), chunk):
        parts = [layers[k][s:s + chunk].reshape(len(geom[s:s + chunk]), -1).astype(np.float64) / 255
                 for k in spec.layer_keys]
        if spec.uses_geometry:
            parts.append(geom[s:s + chunk].astype(np.float64))
        out.append(forward(w, np.concatenate(parts, axis=1)))
    return np.concatenate(out)


def score_row(y: np.ndarray, p: np.ndarray, labels, n_boot: int) -> dict:
    auc = auc_score(y, p)
    lo, hi = bootstrap_ci(y, p, n_boot=n_boot)
    row = {"auc_kaggle": auc, "ci95_alt": lo, "ci95_ust": hi, "n": len(y)}
    for a, b in PAIRS:
        ia, ib = labels.index(a), labels.index(b)
        sel = np.isin(y, [ia, ib])
        row[f"auc_{a}_vs_{b}"] = auc_score((y[sel] == ib).astype(int), p[sel, ib] / (p[sel, ia] + p[sel, ib] + 1e-12))
    return row


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["yontem", "temsil", "model"], sort=False)
    out = g.agg(sifreli_deger=("sifreli_deger", "first"), tohum_sayisi=("tohum", "nunique"),
                auc_ic=("auc_ic", "mean"), auc_kaggle=("auc_kaggle", "mean"), auc_kaggle_std=("auc_kaggle", "std"),
                fark=("fark", "mean"), fark_std=("fark", "std")).reset_index()
    for a, b in PAIRS:
        out[f"auc_{a}_vs_{b}"] = g[f"auc_{a}_vs_{b}"].mean().to_numpy()
    return out


def plot(summary: pd.DataFrame, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    d = summary.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 0.34 * len(d) + 1.4))
    fig.patch.set_facecolor(INK["surface"])
    ax.set_facecolor(INK["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK["axis"])
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(colors=INK["muted"], labelcolor=INK["secondary"], labelsize=8, width=0.6, length=0)
    ax.grid(axis="x", color=INK["grid"], lw=0.5)
    ax.set_axisbelow(True)
    y = np.arange(len(d))
    ax.hlines(y, d.auc_kaggle, d.auc_ic, color=INK["axis"], lw=1.2, zorder=2)
    ax.scatter(d.auc_ic, y, s=46, color=SERIES["ic"], edgecolor=INK["surface"], linewidths=1.2, zorder=3,
               label="COVID-QU-Ex Test (iç)")
    ax.scatter(d.auc_kaggle, y, s=46, color=SERIES["dis"], edgecolor=INK["surface"], linewidths=1.2, zorder=3,
               label="Kaggle, kopyasız (dış)")
    for i, r in d.iterrows():  # dış değer, turuncu noktanın dış yanına (iç noktanın etiketi gibi okunmasın)
        right = r.auc_kaggle >= r.auc_ic
        ax.annotate(f"{r.auc_kaggle:.3f}", (r.auc_kaggle, i), xytext=(7 if right else -7, 0),
                    textcoords="offset points", ha="left" if right else "right", va="center", fontsize=7,
                    color=INK["secondary"])
    short = {"tam görüntü (Π_ROI doğruluğu)": "tam görüntü", "eş örnekli küçültme": "eş örnekli", "FoveaHE": "FoveaHE",
             METHOD: "şifreli özet"}
    ax.set_yticks(y, [f"{short.get(r.yontem, r.yontem)} {r.temsil if r.temsil != '-' else ''} · {r.model}".replace("  ", " ")
                      for _, r in d.iterrows()])
    left = min(d.auc_kaggle.min(), d.auc_ic.min()) - 0.03
    ax.set_xticks(np.arange(np.ceil(left / 0.025) * 0.025, 1.0 + 1e-9, 0.025))
    ax.set_xlim(left, 1.02)
    ax.set_xlabel("teşhis AUC (makro, üç sınıf)", fontsize=8, color=INK["secondary"])
    ax.set_title("COVID-QU-Ex'te eğitilen şifreli modeller harici derlemede", loc="left", fontsize=9,
                 color=INK["primary"], pad=18)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=7, frameon=False,
              labelcolor=INK["secondary"], borderaxespad=0.1, handletextpad=0.3, columnspacing=1.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=INK["surface"])
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--quick", action="store_true", help="sınıf başına 40 görüntü, tohum 0")
    args = ap.parse_args()
    tag = "_hizli" if args.quick else ""
    seeds = [0] if args.quick else args.seeds
    labels = load_dataset("covidqu").labels
    assert labels == ["COVID-19", "Non-COVID", "Normal"], labels
    full, kg = kaggle_rows(labels, args.quick)
    y = kg.y.to_numpy()
    log(f"[genelleme] Kaggle kopyasız: {len(kg)} görüntü {kg.label.value_counts().to_dict()}; tohumlar {seeds}")
    specs = {c: FoveaSpec.parse(c) for c in CONFIGS}
    layers, geom = kaggle_layers(kg, sorted({k for s in specs.values() for k in s.layer_keys}), args.device, tag)
    emb = kaggle_embeddings(full, kg, args.device, tag)
    inner = read_table(config.TABLES / "cozum_modeller.csv")
    inner = inner[inner.veri == DISPLAY["covidqu"]]
    rival = read_table(config.TABLES / "cozum_rakipler.csv")
    rival = rival[rival.veri == DISPLAY["covidqu"]]
    n_boot = 100 if args.quick else 500
    rows = []
    for seed in seeds:
        jobs = [(method_of(c), c, m, config.CHECKPOINTS / f"fovea_covidqu_{m}_{c}_s{seed}_test.npz")
                for c in CONFIGS for m in MODELS]
        jobs += [(METHOD, "-", m, config.CHECKPOINTS / f"ozet_covidqu_{m}_s{seed}_test.npz") for m in ("D", "D2")]
        for method, cfg, kind, path in jobs:
            if not path.exists():
                log(f"[genelleme] atlandı (ağırlık yok): {path.name}")
                continue
            w = load_weights(path)
            if cfg == "-":
                logits, n_values = forward_numpy(w, emb.astype(np.float64)), 512
                ref = rival[(rival.model == kind) & (rival.tohum == seed)]
            else:
                logits, n_values = fovea_logits(w, layers, geom, specs[cfg]), None
                ref = inner[(inner.model == kind) & (inner.temsil == cfg) & (inner.tohum == seed)]
                n_values = int(ref.sifreli_deger.iloc[0]) if len(ref) else np.nan
            row = {"yontem": method, "temsil": cfg, "model": kind, "tohum": seed, "sifreli_deger": n_values,
                   "auc_ic": float(ref.auc.iloc[0]) if len(ref) else np.nan,
                   **score_row(y, softmax(logits), labels, n_boot)}
            row["fark"] = row["auc_ic"] - row["auc_kaggle"]
            rows.append(row)
            log(f"[genelleme] {method[:28]:28s} {cfg:8s} {kind:2s} tohum={seed} iç AUC={row['auc_ic']:.4f} "
                f"Kaggle AUC={row['auc_kaggle']:.4f} (%95 GA {row['ci95_alt']:.4f}-{row['ci95_ust']:.4f}) "
                f"fark={row['fark']:+.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(config.TABLES / f"cozum_genelleme{tag}.csv", index=False)
    summary = summarize(df)
    write_markdown_table(summary, config.TABLES / f"cozum_genelleme{tag}.md", floatfmt="{:.4f}")
    plot(summary, config.FIGURES / f"cozum_genelleme{tag}.png")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
