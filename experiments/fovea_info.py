"""Adım 1 (çözüm, karar noktası): odaklı temsil teşhis bilgisini koruyor mu?

Her yapılandırmada istemci temsili (`foveahe.representation`: odak F, yakın çevre P ve k, genel bakış G) 224×224
odaklı görüntüye yeniden oluşturulur ve Saldırı B'nin ResNet-18 hattıyla eğitilir (gizli bölge yok). ResNet-18,
temsilin taşıdığı teşhis bilgisinin şifresiz üst sınırıdır. Satırlar:
- tam: aynı hattan tam görüntü (G = 224), referans
- tam_pencere: tam görüntü + odak penceresi göstergesi kanalı; ROI bilgisinin kendi katkısını ayıran ek referans
- sabit: bilgisiz girdi; akıl sağlığı kontrolü (beyinde kat ortalaması AUC tam 0.5)
- U{G}: tüm görüntünün eş örnekli küçültülmesi ("küçült ve tamamen şifrele": DCT-CryptoNets, privateST çizgisi)
- F64: yalnız odak (bağlamın katkısı)
- F{F}[_P{P}k{k}]_G{G}: odaklı temsil ızgarası (COZUM_PLANI §13.4)
Ölçüt (COZUM_PLANI §6): aynı tohumlardaki tam ile fark ≤ 0.02 AUC. Beyinde ana AUC havuzlanmış kat dışı tahminlerden
(Saldırı B ile aynı); kat ortalaması ayrıca raporlanır. Havuzlanmış AUC, katların sabit tahminleri farklı olduğundan
bilgisiz girdide bile 0.5'ten biraz sapabilir. Her koşudan sonra satır CSV'ye yazılır; iş yarıda kalırsa aynı komut
kaldığı yerden sürer.

Çalıştırma: .venv\\Scripts\\python -m experiments.fovea_info [--dataset brain covidqu] [--configs ...] [--seeds 0]
            [--quick] [--build-only] [--plot-only]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

import config
from attacks.context_cnn import train_and_predict
from common.evaluation import auc_score, bootstrap_ci
from common.report import write_markdown_table
from foveahe.data import DISPLAY, Dataset, FoveaCache, build_layers, load_dataset, load_layers
from foveahe.representation import SLOTS, FoveaSpec, render

PREDS = config.RESULTS / "preds"
PREDS.mkdir(parents=True, exist_ok=True)
LOG = config.LOGS / "fovea_info.log"
MAX_LOSS = 0.02
QUICK_CONFIGS = ["tam", "sabit", "F64_P32k2_G32", "U64"]
PAIRS = [("Normal", "Non-COVID"), ("Normal", "COVID-19"), ("Non-COVID", "COVID-19")]
# Şekiller: referans paletin ilk üç kategorik yuvası (tüm çiftlerde doğrulandı) ve metin/eksen mürekkepleri
SERIES = {"odak": "#2a78d6", "es": "#eb6834", "yalniz_odak": "#1baf7a"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
       "surface": "#fcfcfb"}


def log(msg: str):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def default_grid() -> list[FoveaSpec]:
    """Odaklı ızgara (F∈{64,32}, G∈{32,16}, P∈{0,32,16}, k∈{2,3}), eş örnekli küçültmeler (U128 iki ciphertext),
    yalnız odak. Referanslar ve karar için kilit satırlar önce koşar."""
    # U{G}_pencere: eş örnekli küçültme + ROI penceresi kanalı; odaklamanın ROI bilgisinden bağımsız katkısını ayırır
    first = [FoveaSpec.parse(n) for n in ("tam", "tam_pencere", "sabit", "F64_G32", "U64", "F64_P32k2_G32", "U90",
                                          "U64_pencere", "U90_pencere", "F32_G16", "U32", "F64")]
    grid = []
    for f in (64, 32):
        for g in (32, 16):
            grid.append(FoveaSpec(focus=f, glob=g))
            grid += [FoveaSpec(focus=f, periphery=p, k=k, glob=g) for p in (32, 16) for k in (2, 3)]
    grid += [FoveaSpec(glob=g) for g in (128, 90, 64, 48, 32)]
    return first + [s for s in grid if s not in first]


def fold_mean_auc(ds: Dataset, probs: np.ndarray) -> float:
    """Beyin: tahmini olan her katta ayrı AUC, sonra ortalama."""
    vals = []
    for k in np.unique(ds.folds):
        te = np.flatnonzero((ds.folds == k) & ~np.isnan(probs[:, 0]))
        if len(te):
            vals.append(auc_score(ds.y[te], probs[te]))
    return float(np.mean(vals))


def train_eval(ds: Dataset, cache: FoveaCache, spec: FoveaSpec, seed: int, epochs: int, quick: bool):
    t0 = time.perf_counter()
    n_cls = len(ds.labels)
    fold_auc = np.nan
    if ds.name == "brain":
        folds = np.unique(ds.folds)
        probs = np.full((len(ds.y), n_cls), np.nan)
        for k in folds[:1] if quick else folds:
            tr, te = np.flatnonzero(ds.folds != k), np.flatnonzero(ds.folds == k)
            log(f"[{ds.name}] {spec.name} tohum={seed} kat={k} (eğitim {len(tr)}, test {len(te)})")
            p_fold, _ = train_and_predict(cache, tr, te, "tam", n_cls, epochs=epochs, seed=seed, log=log)
            probs[te] = p_fold
        te = np.flatnonzero(~np.isnan(probs[:, 0]))
        p, groups = probs[te], ds.groups[te]
        fold_auc = fold_mean_auc(ds, probs)
    else:
        tr, te = ds.train_idx, ds.test_idx
        if quick:
            rng = np.random.default_rng(0)
            tr, te = rng.choice(tr, 3000, replace=False), rng.choice(te, 1000, replace=False)
        log(f"[{ds.name}] {spec.name} tohum={seed} (eğitim {len(tr)}, test {len(te)})")
        probs, _ = train_and_predict(cache, tr, te, "tam", n_cls, epochs=epochs, seed=seed, log=log)
        p, groups = probs, None
    yt = ds.y[te]
    auc = auc_score(yt, p)
    lo, hi = bootstrap_ci(yt, p, groups=groups, n_boot=500)
    row = {"veri": ds.display, "yapilandirma": spec.name, "F": spec.focus, "P": spec.periphery,
           "k": spec.k if spec.periphery else np.nan, "G": spec.glob, "sifreli_deger": spec.n_values,
           "ciphertext": spec.n_ciphertexts, "tohum": seed, "epoch": epochs, "auc": auc, "ci95_alt": lo,
           "ci95_ust": hi, "n": len(te), "sure_s": time.perf_counter() - t0, "auc_kat_ort": fold_auc}
    if ds.name == "covidqu":
        for a, b in PAIRS:
            ia, ib = ds.labels.index(a), ds.labels.index(b)
            sel = np.isin(yt, [ia, ib])
            score = p[sel, ib] / (p[sel, ia] + p[sel, ib] + 1e-12)
            row[f"auc_{a}_vs_{b}"] = auc_score((yt[sel] == ib).astype(int), score)
    return row, probs


def fill_fold_auc(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Kat ortalaması AUC'si eksik beyin satırlarını kayıtlı tahminlerden tamamlar."""
    if "auc_kat_ort" not in df:
        df["auc_kat_ort"] = np.nan
    need = (df.veri == DISPLAY["brain"]) & df.auc_kat_ort.isna()
    if need.any():
        ds = load_dataset("brain")
        for i in df.index[need]:
            path = PREDS / f"fovea_brain_{df.at[i, 'yapilandirma']}_s{int(df.at[i, 'tohum'])}{tag}.npy"
            if path.exists():
                df.at[i, "auc_kat_ort"] = fold_mean_auc(ds, np.load(path))
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Yapılandırma başına tohum ortalaması; tam ile fark aynı tohumlardaki tam satırlarına göre (pozitif = kayıp)."""
    tam = df[df.yapilandirma == "tam"].set_index(["veri", "tohum"])["auc"]
    rows = []
    for (veri, name), d in df.groupby(["veri", "yapilandirma"], sort=False):
        ref = [tam[(veri, t)] for t in d.tohum if (veri, t) in tam.index]
        loss = float(np.mean(ref) - d.auc.mean()) if ref else np.nan
        spec = FoveaSpec.parse(name)
        single = len(d) == 1
        fold = d.auc_kat_ort if "auc_kat_ort" in d else pd.Series(dtype=float)
        rows.append({"veri": veri, "yapilandirma": name, "sifreli_deger": spec.n_values,
                     "ciphertext": spec.n_ciphertexts, "tohum_sayisi": int(d.tohum.nunique()),
                     "auc_ort": float(d.auc.mean()), "auc_std": np.nan if single else float(d.auc.std(ddof=1)),
                     "ci95_alt": float(d.ci95_alt.iloc[0]) if single else np.nan,
                     "ci95_ust": float(d.ci95_ust.iloc[0]) if single else np.nan,
                     "auc_kat_ort": float(fold.mean()) if fold.notna().any() else np.nan, "tam_fark": loss,
                     "olcut": "-" if name in ("tam", "sabit") or np.isnan(loss)
                     else ("evet" if loss <= MAX_LOSS else "hayır")})
    return pd.DataFrame(rows).sort_values(["veri", "sifreli_deger"], kind="stable").reset_index(drop=True)


def save_row(row: dict, out_csv, tag: str):
    if out_csv.exists():
        df = pd.read_csv(out_csv)
        same = (df.veri == row["veri"]) & (df.yapilandirma == row["yapilandirma"]) & (df.tohum == row["tohum"])
        df = pd.concat([df[~same], pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df = fill_fold_auc(df, tag)
    df.to_csv(out_csv, index=False)
    write_markdown_table(summarize(df), out_csv.with_suffix(".md"), floatfmt="{:.4f}")


def _style(ax):
    ax.set_facecolor(INK["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK["axis"])
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK["muted"], labelcolor=INK["secondary"], labelsize=8, width=0.6)


def plot_examples(ds: Dataset):
    """Her sınıftan bir görüntü: tam görüntü, katmanlar ve yeniden oluşturulmuş görüntüler."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    fovea, uniform = FoveaSpec(focus=64, periphery=32, k=2, glob=32), FoveaSpec(glob=64)
    layers, geom = load_layers(ds, ["g224", *fovea.layer_keys, uniform.glob_key], mmap=True, log=log)
    rng = np.random.default_rng(0)
    pick = np.array([rng.choice(np.flatnonzero(ds.y == c)) for c in range(len(ds.labels))])

    def layer(key):
        return torch.from_numpy(np.asarray(layers[key][pick], dtype=np.float32) / 255).unsqueeze(1)

    g = torch.from_numpy(geom[pick])
    columns = [("tam görüntü (224 px)", layer("g224")), (f"odak {fovea.focus}×{fovea.focus}", layer(fovea.focus_key)),
               (f"yakın çevre {fovea.periphery}×{fovea.periphery}, k={fovea.k:g}", layer(fovea.periphery_key)),
               (f"genel bakış {fovea.glob}×{fovea.glob}", layer(fovea.glob_key)),
               (f"FoveaHE: {fovea.n_values:,} değer".replace(",", "."),
                render({k: layer(k) for k in fovea.layer_keys}, g, fovea)),
               (f"eş örnekli U{uniform.glob}: {uniform.n_values:,} değer".replace(",", "."),
                render({uniform.glob_key: layer(uniform.glob_key)}, g, uniform))]
    fig, axes = plt.subplots(len(pick), len(columns), figsize=(2.05 * len(columns), 2.2 * len(pick)), squeeze=False)
    fig.patch.set_facecolor(INK["surface"])
    for r, i in enumerate(pick):
        for c, (title, images) in enumerate(columns):
            ax = axes[r, c]
            ax.imshow(images[r, 0].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(INK["axis"])
                sp.set_linewidth(0.6)
            if r == 0:
                ax.set_title(title, fontsize=8, color=INK["primary"])
            if c == 0:
                ax.set_ylabel(ds.labels[ds.y[i]], fontsize=8, color=INK["secondary"])
                cx, cy, side = geom[i] * 224
                for scale, width in ((fovea.k, 0.8), (1.0, 1.3)):
                    s = side * scale
                    ax.add_patch(Rectangle((cx - s / 2 - 0.5, cy - s / 2 - 0.5), s, s, fill=False, lw=width,
                                           ec=SERIES["odak"]))
    fig.suptitle(f"Odaklı temsil örnekleri, {ds.display} (ilk sütun: iç kare odak, dış kare yakın çevre penceresi)",
                 fontsize=9, color=INK["primary"])
    fig.tight_layout()
    fig.savefig(config.FIGURES / f"cozum_temsil_ornek_{ds.name}.png", dpi=160, facecolor=INK["surface"])
    plt.close(fig)


def plot_curve(df: pd.DataFrame, tag: str = "", stacked: bool = False):
    """Şifrelenen değer sayısı ↔ teşhis AUC; tam görüntü, ölçüt sınırı ve tam + ROI penceresi referans çizgileri.

    stacked: veri kümeleri alt alta (tez sayfasında okunur yazı için), çıktı adı `_tez` ekli.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    s = summarize(df)
    datasets = [v for v in DISPLAY.values() if v in set(s.veri)]
    shape = (len(datasets), 1) if stacked else (1, len(datasets))
    size = (6.4, 4.4 * len(datasets)) if stacked else (6.4 * len(datasets), 4.6)
    fig, axes = plt.subplots(*shape, figsize=size, squeeze=False)
    fig.patch.set_facecolor(INK["surface"])
    for ax, veri in zip(axes.flat, datasets):
        _style(ax)
        d = s[(s.veri == veri) & (s.yapilandirma != "sabit")].copy()
        specs = d.yapilandirma.map(FoveaSpec.parse)
        fov = d[specs.map(lambda x: x.focus > 0 and x.glob > 0)]
        foc = d[specs.map(lambda x: x.focus > 0 and x.glob == 0)]
        uni = d[specs.map(lambda x: not (x.focus or x.periphery or x.window) and x.glob != 224)]
        uni = uni.sort_values("sifreli_deger")
        ax.grid(axis="y", color=INK["grid"], lw=0.5)
        ax.set_axisbelow(True)
        ax.axvline(SLOTS, color=INK["axis"], lw=0.7)
        # eksenin üstünde: veri etiketleriyle (ör. U90) çakışmasın
        ax.text(SLOTS, 1.01, "1 ciphertext (8.192 slot)", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=7, color=INK["muted"])
        vals = d.auc_ort.to_numpy()
        lo, hi = float(vals.min()), float(vals.max())
        pad = max(5e-4, 0.12 * (hi - lo))
        ax.set_ylim(lo - pad, hi + 2 * pad)
        y_min = ax.get_ylim()[0]
        tam = d[d.yapilandirma == "tam"]
        if len(tam):
            ref = float(tam.auc_ort.iloc[0])
            ax.axhline(ref, color=INK["secondary"], lw=0.8)
            ax.text(0.01, ref, f"tam görüntü (224 px): {ref:.4f}", transform=ax.get_yaxis_transform(), ha="left",
                    va="bottom", fontsize=7, color=INK["secondary"])
            limit = ref - MAX_LOSS
            if limit >= y_min:
                ax.axhline(limit, color=INK["muted"], lw=0.8)
                ax.text(0.01, limit, f"ölçüt sınırı: tam − {MAX_LOSS}", transform=ax.get_yaxis_transform(),
                        ha="left", va="bottom", fontsize=7, color=INK["muted"])
            else:  # sınır tüm noktaların çok altındaysa ekseni sıkıştırmak yerine not düş
                note = "tüm yapılandırmalar geçiyor" if (d.auc_ort >= limit).all() else "geçmeyen yapılandırma var"
                ax.text(0.01, 0.02, f"ölçüt sınırı tam − {MAX_LOSS} = {limit:.4f}, eksenin altında: {note}",
                        transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color=INK["muted"])
        win = d[d.yapilandirma == "tam_pencere"]
        if len(win):
            y = float(win.auc_ort.iloc[0])
            ax.axhline(y, color=INK["muted"], lw=0.8)
            ax.text(0.99, y, f"tam görüntü + ROI penceresi: {y:.4f}", transform=ax.get_yaxis_transform(), ha="right",
                    va="top", fontsize=7, color=INK["muted"])
        ring = dict(edgecolor=INK["surface"], linewidths=0.9, zorder=3)
        if len(uni):
            ax.plot(uni.sifreli_deger, uni.auc_ort, color=SERIES["es"], lw=1.0, zorder=2)
            ax.scatter(uni.sifreli_deger, uni.auc_ort, s=34, color=SERIES["es"], label="eş örnekli küçültme (U)",
                       **ring)
            for _, r in uni.iterrows():  # yakındaki odaklı noktalar çoğunlukla alttaysa etiket üste
                near = fov[(fov.sifreli_deger > r.sifreli_deger / 1.6) & (fov.sifreli_deger < r.sifreli_deger * 1.6)]
                above = len(near) > 0 and (near.auc_ort < r.auc_ort).mean() > 0.5
                ax.annotate(r.yapilandirma, (r.sifreli_deger, r.auc_ort), xytext=(0, 6 if above else -11),
                            textcoords="offset points", ha="center", fontsize=7, color=INK["secondary"])
        uwin = d[specs.map(lambda x: x.window and x.glob != 224)]
        if len(uwin):  # aynı ton, içi boş işaret: renk tek başına kimlik taşımasın
            ax.scatter(uwin.sifreli_deger, uwin.auc_ort, s=34, facecolor=INK["surface"], edgecolor=SERIES["es"],
                       linewidths=1.2, zorder=3, label="eş örnekli + ROI penceresi")
        if len(fov):
            ax.scatter(fov.sifreli_deger, fov.auc_ort, s=34, color=SERIES["odak"], label="odaklı temsil (FoveaHE)",
                       **ring)
            passing = fov[fov.olcut == "evet"]
            notable = {fov.auc_ort.idxmax()}
            if len(passing):
                notable.add(passing.sifreli_deger.idxmin())
            for i in notable:
                r = fov.loc[i]
                ax.annotate(r.yapilandirma, (r.sifreli_deger, r.auc_ort), xytext=(5, 5), textcoords="offset points",
                            fontsize=7, color=INK["primary"])
        if len(foc):
            ax.scatter(foc.sifreli_deger, foc.auc_ort, s=34, color=SERIES["yalniz_odak"], label="yalnız odak (F)",
                       **ring)
            for _, r in foc.iterrows():  # açık renk yüzeyde 3:1'in altında: doğrudan etiket zorunlu
                ax.annotate(r.yapilandirma, (r.sifreli_deger, r.auc_ort), xytext=(5, -10), textcoords="offset points",
                            fontsize=7, color=INK["secondary"])
        for part in (fov, uni, foc):
            err = part[part.auc_std.notna()]
            if len(err):
                ax.errorbar(err.sifreli_deger, err.auc_ort, yerr=err.auc_std, fmt="none", ecolor=INK["muted"],
                            elinewidth=0.7, zorder=1)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator([1024, 2048, 4096, 8192, 16384]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", ".")))
        ax.set_xlim(800, 24000)
        ax.set_xlabel("şifrelenen değer sayısı (log ölçek)", fontsize=8, color=INK["secondary"])
        ax.set_ylabel("teşhis AUC (makro, ResNet-18 üst sınır)", fontsize=8, color=INK["secondary"])
        ax.set_title(veri, loc="left", fontsize=9, color=INK["primary"])
        ax.legend(loc="lower right", fontsize=7, frameon=False, labelcolor=INK["secondary"])
    fig.tight_layout()
    fig.savefig(config.FIGURES / f"cozum_bilgi_egrisi{tag}{'_tez' if stacked else ''}.png", dpi=160,
                facecolor=INK["surface"])
    plt.close(fig)


def run_dataset(name: str, specs, seeds, epochs: int, quick: bool, build_only: bool, out_csv, tag: str):
    ds = load_dataset(name)
    build_layers(ds, sorted({k for s in default_grid() + list(specs) for k in s.layer_keys}), log=log)
    plot_examples(ds)
    if build_only:
        return
    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = set(zip(prev.veri, prev.yapilandirma, prev.tohum.astype(int)))
    for spec in specs:
        todo = [s for s in seeds if (ds.display, spec.name, s) not in done]
        if not todo:
            log(f"[{name}] {spec.name}: tohumlar {list(seeds)} zaten var, atlandı")
            continue
        layers, geom = load_layers(ds, spec.layer_keys, log=log)
        cache = FoveaCache(layers, geom, ds.y, spec)
        for seed in todo:
            ep = 1 if quick or spec.name == "sabit" else epochs
            row, probs = train_eval(ds, cache, spec, seed, ep, quick)
            np.save(PREDS / f"fovea_{name}_{spec.name}_s{seed}{tag}.npy", probs)
            save_row(row, out_csv, tag)
            log(f"[{name}{tag}] {spec.name:15s} tohum={seed} değer={spec.n_values} AUC={row['auc']:.4f} "
                f"(%95 GA {row['ci95_alt']:.4f}-{row['ci95_ust']:.4f}) kat ort={row['auc_kat_ort']:.4f} "
                f"{row['sure_s']:.0f} s")
        del cache, layers
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="*", default=["brain", "covidqu"])
    ap.add_argument("--configs", nargs="*", default=None, help="yapılandırma adları (varsayılan: tam ızgara)")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--epochs-brain", type=int, default=12)
    ap.add_argument("--epochs-cxr", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="küçük alt küme, 1 epoch, 4 yapılandırma")
    ap.add_argument("--build-only", action="store_true", help="yalnızca katman önbelleği ve örnek şekilleri")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--tez", action="store_true", help="tez sayfası için ek alt alta şekil (_tez.png)")
    args = ap.parse_args()
    tag = "_hizli" if args.quick else ""
    out_csv = config.TABLES / f"cozum_bilgi{tag}.csv"
    if not args.plot_only:
        assert torch.cuda.is_available(), "CUDA bulunamadı"
        names = args.configs or (QUICK_CONFIGS if args.quick else [s.name for s in default_grid()])
        specs = [FoveaSpec.parse(n) for n in names]
        for name in args.dataset:
            epochs = args.epochs_brain if name == "brain" else args.epochs_cxr
            run_dataset(name, specs, args.seeds, epochs, args.quick, args.build_only, out_csv, tag)
    if out_csv.exists():
        df = fill_fold_auc(pd.read_csv(out_csv), tag)
        summary = summarize(df)
        write_markdown_table(summary, out_csv.with_suffix(".md"), floatfmt="{:.4f}")
        plot_curve(df, tag)
        if args.tez:
            plot_curve(df, tag, stacked=True)
        print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
