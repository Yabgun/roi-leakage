"""Adım 6 (çözüm): büyük sonuç şekli ve gizlilik şartlı hız tablosu.

Tüm adımların tablolarını tek yerde birleştirir (veri kümesi başına bir şekil):
- **Teşhis AUC:** şifreli çalışabilen modelin doğruluğu (`cozum_modeller.csv`; şifreli özet için `cozum_rakipler.csv`).
  Tam şifreleme ve Π_ROI satırlarında bu, aynı modelin tam görüntüdeki doğruluğudur: seçici şifreleme model çıktısını
  değiştirmez.
- **Saldırgan AUC:** tamamen şifreli yöntemlerde (tam şifreleme, FoveaHE, şifreli özet) Adım 4 denetiminin en yüksek
  satırı (`cozum_sizinti.csv`); Π_ROI varsayılanında Saldırı B açık bağlam (`saldiri_B_baglam.csv`); gizlilik şartlı
  Π_ROI'de tanım gereği 0.8; Bi-CryptoNets tarzı bozuk bağlamda `saldiri_B_gurultu.csv`'nin en iyi (en düşük) satırı.
- **Süre ve hız kazancı:** `cozum_maliyet.csv` (aynı koşuda ölçülen tam şifreleme referansına göre). Π_ROI varsayılanı
  ve gizlilik şartlı Π_ROI için `savunma.csv`'deki oranlar aynı koşudaki tam şifreleme süresine uygulanır.
Gizlilik şartlı hız kazancı: saldırgan AUC ≤ 0.8 koşulunu sağlayan yapılandırmanın hız kazancı (Π_ROI'de bu koşul
neredeyse tüm görüntüyü şifrelemeyi gerektirir, yani ~1×).

Beyin MR'da Π_ROI saldırgan satırları varsayılan olarak görünür piksel normalizasyonuyla koşulan tablolardan
(`*_gnorm.csv`, inceleme S1) alınır; `--norm kesit` ilk sürümün tablolarını kullanır.

Çıktılar: results/figures/cozum_pareto_{veri}.png|pdf, results/tables/cozum_ozet.csv|md.
Çalıştırma: .venv\\Scripts\\python -m experiments.fovea_pareto [--dataset brain covidqu] [--configs F32_G16 F64_G32]
            [--norm gorunur|kesit]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import config
from common.report import write_markdown_table
from experiments.fovea_models import ORIGINAL, read_table, summarize as summarize_models
from foveahe.data import DISPLAY

PRIVACY_LIMIT = 0.8          # gizlilik şartı: saldırgan AUC ≤ 0.8 (savunma deneyiyle aynı eşik)
LEAKFREE_LIMIT = 0.55        # Adım 4 ölçütü: şans düzeyi sınırı
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
       "surface": "#fcfcfb"}
RAMP = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")  # sıralı mavi (saldırgan AUC), doğrulanmış basamaklar


def table(name: str):
    path = config.TABLES / name
    return read_table(path) if path.exists() else None


def model_auc(models: pd.DataFrame, veri: str, temsil: str):
    """(en iyi model sınıfı, AUC, std, tohum sayısı) — yoksa None."""
    d = models[(models.veri == veri) & (models.temsil == temsil)]
    if d.empty:
        return None
    best = d.loc[d.auc_ort.idxmax()]
    return best.model, float(best.auc_ort), float(best.auc_std) if pd.notna(best.auc_std) else np.nan, \
        int(best.tohum_sayisi)


def cost_row(costs: pd.DataFrame, veri: str, yontem: str, temsil: str, model: str):
    d = costs[(costs.veri == veri) & (costs.yontem == yontem) & (costs.temsil == temsil) & (costs.model == model)]
    return None if d.empty else d.iloc[0]


def audited_attacker(sizinti: pd.DataFrame, veri: str) -> float:
    """Tamamen şifreli yöntemler: Adım 4'te FoveaHE satırlarının en yükseği (şans düzeyi kanıtı)."""
    d = sizinti[(sizinti.veri == veri) & (sizinti.yontem == "FoveaHE")]
    return float(d.auc_kat_ort.max()) if len(d) else np.nan


def piroi_attacker(baglam: pd.DataFrame, veri: str) -> float:
    d = baglam[(baglam.veri == veri) & (baglam.gorus == "baglam")]
    return float(d.auc.mean()) if len(d) else np.nan


def piroi_speedups(savunma: pd.DataFrame, name: str, size: int) -> tuple[float, float]:
    """(varsayılan ROI hız kazancı, gizlilik şartlı hız kazancı): savunma deneyindeki analitik oranlar."""
    col = f"hiz_kazanci_{size}"
    d = savunma[savunma.veri == name]
    if d.empty or col not in d:
        return np.nan, np.nan
    default = d[d.politika == "goruntu_roi"]
    private = d[(d.auc <= PRIVACY_LIMIT) & d[col].notna()]
    return (float(default[col].iloc[0]) if len(default) else np.nan,
            float(private[col].max()) if len(private) else 1.0)


def build_rows(name: str, configs, models, costs, sizinti, baglam, rakipler, savunma, gurultu) -> list[dict]:
    veri, size = DISPLAY[name], ORIGINAL[name]
    full_name = f"U{size}"
    rows = []
    leakfree = audited_attacker(sizinti, veri)
    full = model_auc(models, veri, full_name)
    full_cost = {m: cost_row(costs, veri, "tam şifreleme", f"{size} px", m) for m in ("D", "C")}

    def add(yontem, model, auc, std, seeds, sure, hiz, saldirgan, deger, sizinti_turu, not_=""):
        rows.append({"veri": veri, "yontem": yontem, "model": model, "sifreli_deger": deger, "teshis_auc": auc,
                     "teshis_auc_std": std, "tohum_sayisi": seeds, "saldirgan_auc": saldirgan, "sure_s": sure,
                     "hiz_kazanci": hiz, "sizinti_turu": sizinti_turu, "not": not_})

    if full:
        model, auc, std, seeds = full
        c = full_cost.get(model if model == "C" else "D")
        sure = float(c.toplam_s) if c is not None else np.nan
        add("tam şifreleme", model, auc, std, seeds, sure, 1.0, leakfree, size * size, "yok",
            "referans: tüm görüntü şifreli")
        default_sp, private_sp = piroi_speedups(savunma, name, size)
        add("Π_ROI (varsayılan ROI)", model, auc, std, seeds, sure / default_sp if default_sp else np.nan, default_sp,
            piroi_attacker(baglam, veri), size * size, "açık pikseller + ROI meta verisi",
            "hız savunma deneyindeki oran; doğruluk tam görüntüyle aynı: seçici şifreleme çıktıyı değiştirmez")
        measured = costs[(costs.veri == veri) & costs.yontem.str.startswith("Π_ROI")]
        if len(measured):  # CXR'de gizlilik şartlı Π_ROI doğrudan ölçüldü (%98 şifreli)
            m = measured.iloc[0]
            p_sure, private_sp = float(m.toplam_s), sure / float(m.toplam_s)
            p_not = f"ölçüldü ({m.yontem})"
        else:
            p_sure = sure / private_sp if private_sp else np.nan
            p_not = "savunma deneyinden (beyinde şart tüm görüntüyü şifrelemeyi gerektiriyor)"
        add(f"Π_ROI (gizlilik şartlı, saldırgan ≤ {PRIVACY_LIMIT})", model, auc, std, seeds, p_sure, private_sp,
            PRIVACY_LIMIT, size * size, "sınırlı açık piksel",
            f"{p_not}: gizlilik şartı hızı tam şifreleme düzeyine geri çekiyor")

    for cfg in configs:  # FoveaHE yapılandırmaları: her biri için en iyi model sınıfı
        best = model_auc(models, veri, cfg)
        if not best:
            continue
        model, auc, std, seeds = best
        c = cost_row(costs, veri, f"FoveaHE-{model}", cfg, model)
        add(f"FoveaHE {cfg}", model, auc, std, seeds, float(c.toplam_s) if c is not None else np.nan,
            float(c.hiz_kazanci_tam) if c is not None else np.nan, leakfree,
            int(c.sifreli_deger) if c is not None else np.nan, "yok", "odaklı temsilin tamamı şifreli")

    if rakipler is not None:
        d = rakipler[rakipler.veri == veri]
        if len(d):
            g = d.groupby("model", sort=False).agg(
                auc_ort=("auc", "mean"), auc_std=("auc", "std"), tohum=("tohum", "nunique"),
                sure=("toplam_s", "mean"), deger=("sifreli_deger", "first"),
                istemci=("istemci_on_isleme_s", "mean")).reset_index()
            best = g.loc[g.auc_ort.idxmax()]
            hiz = float(full_cost["D"].toplam_s) / float(best.sure) if full_cost.get("D") is not None else np.nan
            add("şifreli özet (ImageNet ResNet-18)", best.model, float(best.auc_ort), float(best.auc_std),
                int(best.tohum), float(best.sure), hiz, leakfree, int(best.deger), "yok",
                f"istemcide 11.2 M parametreli ağ ({best.istemci * 1000:.0f} ms); hız D ailesi referansına göre")

    if gurultu is not None and full:  # Bi-CryptoNets tarzı: hız Π_ROI varsayılanı, saldırgan en iyi bozulma düzeyi
        d = gurultu[(gurultu.veri == veri) & (gurultu.gorus == "baglam")]
        if len(d):
            best = d.loc[d.auc.idxmin()]
            model, auc, std, seeds = full
            default_sp, _ = piroi_speedups(savunma, name, size)
            add(f"bozuk bağlam ({best.bozulma} {best.duzey:g})", model, auc, std, seeds,
                float(full_cost["D"].toplam_s) / default_sp if default_sp else np.nan, default_sp, float(best.auc),
                size * size, "gürültülü/bulanık açık bağlam",
                "doğruluk Π_ROI ile aynı varsayıldı; saldırgan en iyi bozulma düzeyinden")
    return rows


def plot(df: pd.DataFrame, name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    d = df[df.hiz_kazanci.notna() & df.teshis_auc.notna()].copy()
    if d.empty:
        return
    cmap = LinearSegmentedColormap.from_list("saldirgan", RAMP)
    norm = Normalize(vmin=0.5, vmax=1.0)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    fig.patch.set_facecolor(INK["surface"])
    ax.set_facecolor(INK["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK["axis"])
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK["muted"], labelcolor=INK["secondary"], labelsize=8, width=0.6)
    ax.grid(axis="both", color=INK["grid"], lw=0.5)
    ax.set_axisbelow(True)
    def short(name: str) -> str:
        if name.startswith("Π_ROI"):
            return r"$\Pi_{\mathrm{ROI}}$ (gizlilik şartlı)" if "gizlilik" in name else r"$\Pi_{\mathrm{ROI}}$ (varsayılan)"
        return name.split(" (")[0]

    lo, hi = d.teshis_auc.min(), d.teshis_auc.max()
    span = max(hi - lo, 1e-3)
    # Π_ROI türevleri aynı modeli kullanır, doğrulukları birebir aynıdır: noktalar tam çakışıyor. Renkleri (saldırgan
    # AUC) farklı olduğu için ikisi de görünmeli; yalnızca çizimde küçük dikey kaydırma uygulanır, etiketler üst/alt.
    clusters = {}
    for i, (_, r) in enumerate(d.iterrows()):
        clusters.setdefault((round(np.log2(r.hiz_kazanci), 1), round(r.teshis_auc, 4)), []).append(i)
    dy = np.zeros(len(d))
    for idx in clusters.values():
        for k, i in enumerate(idx):
            dy[i] = (k - (len(idx) - 1) / 2) * 0.05 * span
    sc = ax.scatter(d.hiz_kazanci, d.teshis_auc + dy, c=d.saldirgan_auc, cmap=cmap, norm=norm, s=110,
                    edgecolor=INK["surface"], linewidths=1.2, zorder=3)
    dodged = any(len(v) > 1 for v in clusters.values())
    ax.set_xscale("log", base=2)
    ticks = [1, 2, 4, 8, 16, 32]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}×"))
    ax.set_xlim(0.7, max(48, d.hiz_kazanci.max() * 1.6))
    pad = max(0.01, 0.22 * span)
    ax.set_ylim(lo - pad, hi + 1.4 * pad)
    ax.set_xlabel("hız kazancı: aynı model sınıfının tam şifreleme süresine göre (log ölçek)", fontsize=8,
                  color=INK["secondary"])
    ax.set_ylabel("teşhis AUC (şifreli çalışabilen model)", fontsize=8, color=INK["secondary"])
    bar = fig.colorbar(sc, ax=ax, pad=0.02)
    bar.set_label("saldırgan AUC (sunucunun gördüğünden teşhis)", fontsize=8, color=INK["secondary"])
    bar.ax.tick_params(colors=INK["muted"], labelcolor=INK["secondary"], labelsize=7, width=0.6)
    bar.ax.axhline(LEAKFREE_LIMIT, color=INK["surface"], lw=1.2)
    bar.ax.text(1.6, LEAKFREE_LIMIT, f" şans düzeyi sınırı {LEAKFREE_LIMIT}", va="center", fontsize=7,
                color=INK["secondary"], transform=bar.ax.get_yaxis_transform())
    fig.tight_layout(rect=(0, 0.035 if dodged else 0, 1, 1))
    if dodged:
        fig.text(0.01, 0.012, "Not: aynı doğrulukta çakışan noktalara yalnızca çizimde küçük dikey kaydırma uygulanmıştır.",
                 ha="left", fontsize=7.5, color=INK["secondary"])
    # Etiketler yerleşim kesinleşince: her nokta için sırayla 8 konum denenir; hiçbir işarete, önceki etikete ya da eksen
    # dışına taşmayan ilk konum seçilir (5 tohumda yakınlaşan noktalar sabit "üste yaz" kuralıyla çakışıyordu).
    from matplotlib.transforms import Bbox
    renderer = fig.canvas.get_renderer()
    xy = np.column_stack([d.hiz_kazanci.to_numpy(), (d.teshis_auc + dy).to_numpy()])
    radius = np.sqrt(110) / 2 * fig.dpi / 72 + 2
    taken = [Bbox.from_extents(px - radius, py - radius, px + radius, py + radius)
             for px, py in ax.transData.transform(xy)]
    frame = ax.get_window_extent(renderer)
    spots = [((0, 11), "center", "bottom"), ((0, -11), "center", "top"), ((10, 0), "left", "center"),
             ((-10, 0), "right", "center"), ((8, 8), "left", "bottom"), ((-8, 8), "right", "bottom"),
             ((8, -8), "left", "top"), ((-8, -8), "right", "top")]
    for i, (_, r) in enumerate(d.iterrows()):
        order = spots if dy[i] >= 0 else [spots[1], spots[0]] + spots[2:]
        chosen = None
        for offset, ha, va in order:
            ann = ax.annotate(short(r.yontem), xy[i], xytext=offset, textcoords="offset points", ha=ha, va=va,
                              fontsize=7, color=INK["primary"])
            box = ann.get_window_extent(renderer)
            inside = frame.x0 <= box.x0 and box.x1 <= frame.x1 and frame.y0 <= box.y0 and box.y1 <= frame.y1
            if inside and not any(box.overlaps(t) for t in taken):
                chosen = box
                break
            ann.remove()
        if chosen is None:  # hiçbir konum boş değilse ilk tercih
            offset, ha, va = order[0]
            ann = ax.annotate(short(r.yontem), xy[i], xytext=offset, textcoords="offset points", ha=ha, va=va,
                              fontsize=7, color=INK["primary"])
            chosen = ann.get_window_extent(renderer)
        taken.append(chosen)
    for ext in ("png", "pdf"):
        fig.savefig(config.FIGURES / f"cozum_pareto_{name}.{ext}", dpi=200, facecolor=INK["surface"])
    plt.close(fig)


def replace_brain(df, new, brain: str, what: str):
    """Beyin satırlarını görünür piksel normalizasyonlu tablodan alır (S1); tablo yoksa uyarıp ilk sürümü korur."""
    if df is None:
        return new
    if new is None or not (new.veri == brain).any():
        print(f"UYARI: {what} için _gnorm tablosu yok; beyin satırları ilk sürümden (kesit normalizasyonu)")
        return df
    return pd.concat([df[df.veri != brain], new[new.veri == brain]], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="*", default=["brain", "covidqu"])
    ap.add_argument("--configs", nargs="*", default=["F32_G16", "F64_G32"])
    ap.add_argument("--norm", choices=["gorunur", "kesit"], default="gorunur")
    args = ap.parse_args()
    models_raw, costs = table("cozum_modeller.csv"), table("cozum_maliyet.csv")
    if models_raw is None or costs is None:
        raise SystemExit("cozum_modeller.csv ve cozum_maliyet.csv gerekli")
    models = summarize_models(models_raw)
    sizinti, baglam = table("cozum_sizinti.csv"), table("saldiri_B_baglam.csv")
    rakipler, savunma = table("cozum_rakipler.csv"), table("savunma.csv")
    gurultu = table("saldiri_B_gurultu.csv")
    if args.norm == "gorunur":
        baglam = replace_brain(baglam, table("saldiri_B_baglam_gnorm.csv"), DISPLAY["brain"], "Saldırı B")
        savunma = replace_brain(savunma, table("savunma_gnorm.csv"), "brain", "savunma")
        gurultu = replace_brain(gurultu, table("saldiri_B_gurultu_gnorm.csv"), DISPLAY["brain"], "bozuk bağlam")
    rows = []
    for name in args.dataset:
        rows += build_rows(name, args.configs, models, costs, sizinti, baglam, rakipler, savunma, gurultu)
    df = pd.DataFrame(rows)
    df.to_csv(config.TABLES / "cozum_ozet.csv", index=False)
    write_markdown_table(df.drop(columns=["not"]), config.TABLES / "cozum_ozet.md", floatfmt="{:.4f}")
    for name in args.dataset:
        plot(df[df.veri == DISPLAY[name]], name)
    print(df.drop(columns=["not"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
