"""Özgünlük bonusu: bütçe duyarlı odak. Sabit şifreli değer bütçesinde çözünürlük odak ile genel bakış arasında nasıl
paylaştırılmalı?

Bütçe B ∈ {1024, 2048, 4096, 8192} değer (8.192 = bir ciphertext) ve odak payı s ∈ {0, 0.25, 0.5, 0.75, 1}:
- s = 0: eş örnekli küçültme, G = ⌊√B⌋ (geometri yok)
- 0 < s < 1: F = ⌊√(s·B)⌋, G = ⌊√(B − F² − 3)⌋ (3 geometri değeri bütçeye dahil)
- s = 1: yalnız odak, F = ⌊√(B − 3)⌋
Yakın çevre katmanı kullanılmaz (Adım 1'de beyinde katkısızdı; CXR'de akciğer penceresi neredeyse tüm görüntü olduğundan
genel bakıştan kabadır). Yapılandırmalar Adım 2 hattıyla, şifreli çalışabilen modellerle değerlendirilir
(`fovea_models.run_dataset`: aynı bölmeler ve ayar protokolü); ağırlık ve tahmin dosyaları `_butce` ekiyle ayrılır.
Çıktılar: results/tables/cozum_butce.csv|md, results/figures/cozum_butce.png (bütçe başına AUC ↔ odak payı).

Çalıştırma: .venv\\Scripts\\python -m experiments.fovea_budget [--dataset brain covidqu] [--models C D]
            [--budgets 1024 2048 4096 8192] [--seeds 0] [--plot-only] [--quick]
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

import config
from common.report import write_markdown_table
from experiments.fovea_models import read_table, run_dataset, summarize
from foveahe.data import DISPLAY
from foveahe.representation import GEOM_VALUES, FoveaSpec

SHARES = (0.0, 0.25, 0.5, 0.75, 1.0)
BUDGETS = (1024, 2048, 4096, 8192)
# Referans paletin mavi (sıralı) rampası, sıralı kullanımda açık yüzeye göre 2:1 üstü basamaklar (250, 400, 550, 700)
RAMP = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
       "surface": "#fcfcfb"}


def allocation(budget: int, share: float) -> FoveaSpec:
    if share <= 0:
        return FoveaSpec(glob=int(math.isqrt(budget)))
    if share >= 1:
        return FoveaSpec(focus=int(math.isqrt(budget - GEOM_VALUES)))
    focus = int(math.isqrt(int(share * budget)))
    return FoveaSpec(focus=focus, glob=int(math.isqrt(budget - focus * focus - GEOM_VALUES)))


def grid(budgets) -> pd.DataFrame:
    rows = []
    for b in budgets:
        for s in SHARES:
            spec = allocation(b, s)
            assert spec.n_values <= b, (b, s, spec)
            rows.append({"butce": b, "odak_payi_hedef": s, "yapilandirma": spec.name, "F": spec.focus, "G": spec.glob,
                         "sifreli_deger": spec.n_values, "odak_payi": spec.focus ** 2 / spec.n_values})
    return pd.DataFrame(rows)


def attach(df: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    """Adım 2 özet satırlarını (model × temsil) bütçe planıyla birleştirir; aynı temsil birden çok bütçede olabilir."""
    s = summarize(df)
    return plan.merge(s[["veri", "model", "temsil", "auc_ort", "auc_std", "tohum_sayisi"]],
                      left_on="yapilandirma", right_on="temsil", how="inner").drop(columns="temsil")


def plot(table: pd.DataFrame, out_path, ncols: int | None = None, panel=(4.6, 3.9)):
    """ncols verilirse paneller satırlara bölünür (tez sayfasında okunur yazı için 2 × 2)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common.tez_bicim import VERI_TEZ, kaydet, sayi

    tez = ncols is not None  # 2 × 2 tez şekli: Türkçe veri adları, ayraçsız bütçe yazımı, PNG + PDF
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    panels = [(v, m) for v in DISPLAY.values() for m in sorted(table.model.unique()) if
              ((table.veri == v) & (table.model == m)).any()]
    if not panels:
        return
    ncols = ncols or len(panels)
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows), squeeze=False)
    fig.patch.set_facecolor(INK["surface"])
    budgets = sorted(table.butce.unique())
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax, (veri, model) in zip(axes.flat, panels):
        ax.set_facecolor(INK["surface"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK["axis"])
            ax.spines[side].set_linewidth(0.6)
        ax.tick_params(colors=INK["muted"], labelcolor=INK["secondary"], labelsize=8, width=0.6)
        ax.grid(axis="y", color=INK["grid"], lw=0.5)
        ax.set_axisbelow(True)
        d = table[(table.veri == veri) & (table.model == model)]
        for b, color in zip(budgets, RAMP[-len(budgets):]):
            g = d[d.butce == b].sort_values("odak_payi")
            if g.empty:
                continue
            fmt = sayi(b) if tez else f"{b:,}".replace(",", ".")
            ax.plot(g.odak_payi, g.auc_ort, "-o", color=color, lw=1.2, ms=5, mec=INK["surface"], mew=0.8,
                    label=fmt + " değer")
        if len(d):  # bütçe başına etiket üst üste biniyordu: yalnız panelin en iyisi (bütçe başına değerler tabloda)
            best = d.loc[d.auc_ort.idxmax()]
            lo, hi = d.auc_ort.min(), d.auc_ort.max()
            ax.set_ylim(lo - 0.06 * (hi - lo), hi + 0.16 * (hi - lo))
            best_b = sayi(best.butce) if tez else f"{best.butce:,}".replace(",", ".")
            ax.annotate(f"en iyi {best.auc_ort:.3f} ({best_b} değer)",
                        (best.odak_payi, best.auc_ort), xytext=(0, 7), textcoords="offset points", ha="center",
                        fontsize=7, color=INK["secondary"])
        ax.set_xlabel("odak payı (odak değerleri / toplam)", fontsize=8, color=INK["secondary"])
        ax.set_ylabel(f"teşhis AUC (Model {model})", fontsize=8, color=INK["secondary"])
        ax.set_title(f"{VERI_TEZ.get(veri, veri) if tez else veri}, Model {model}", loc="left", fontsize=9,
                     color=INK["primary"])
        ax.set_xlim(-0.05, 1.05)
        ax.legend(title="bütçe", title_fontsize=7, fontsize=7, frameon=False, labelcolor=INK["secondary"],
                  loc="lower center")
    fig.tight_layout()
    if tez:
        kaydet(fig, out_path, dpi=200, facecolor=INK["surface"])
    else:
        fig.savefig(out_path, dpi=160, facecolor=INK["surface"])
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="*", default=["brain", "covidqu"])
    ap.add_argument("--models", nargs="*", default=["C", "D"])
    ap.add_argument("--budgets", nargs="*", type=int, default=list(BUDGETS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tek bütçe (2048), hızlı eğitim")
    ap.add_argument("--tez", action="store_true", help="tez sayfası için ek 2 × 2 şekil (_tez.png)")
    args = ap.parse_args()
    tag = "_butce_hizli" if args.quick else "_butce"
    budgets = [2048] if args.quick else args.budgets
    plan = grid(budgets)
    print(plan.to_string(index=False))
    out_csv = config.TABLES / f"cozum{tag}.csv"
    if not args.plot_only:
        configs = list(dict.fromkeys(plan.yapilandirma))
        for name in args.dataset:
            run_dataset(name, args.models, configs, args.seeds, args.device, args.quick, out_csv, tag)
    if out_csv.exists():
        table = attach(read_table(out_csv), plan)
        table.to_csv(config.TABLES / f"cozum{tag}_ozet.csv", index=False)
        best = table.loc[table.groupby(["veri", "model", "butce"]).auc_ort.idxmax()]
        write_markdown_table(best[["veri", "model", "butce", "yapilandirma", "odak_payi", "auc_ort"]],
                             config.TABLES / f"cozum{tag}_en_iyi.md", floatfmt="{:.4f}")
        plot(table, config.FIGURES / f"cozum{tag}.png")
        if args.tez:
            plot(table, config.FIGURES / f"cozum{tag}_tez.png", ncols=2, panel=(3.3, 3.0))
        print(best.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
