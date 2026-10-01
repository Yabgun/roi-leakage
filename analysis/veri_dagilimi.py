"""Veri kümelerinin ve deney bölmelerinin sınıf dağılımı (danışman maddesi 2).

Bütün sayılar manifestolardan sayılır; elle sayı yazılmaz. Makalenin "Veri kümeleri" bölümündeki sayılarla
(results/beklenen/makale_degerleri.json, `metin_veri`) birebir karşılaştırılır; tutmazsa betik hata verir.

Çıktılar:
- results/tables/veri_dagilimi_beyin.csv|md      beyin MR: kat × sınıf (kesit) ve kat başına hasta
- results/tables/veri_dagilimi_covidqu.csv|md    COVID-QU-Ex: resmi bölme × sınıf (sayı ve yüzde)
- results/tables/veri_dagilimi_kaggle.csv|md     Kaggle CXR: bölme × sınıf, COVID-QU-Ex kopyaları ve kalan dış test kümesi
- results/tables/veri_dagilimi_deneyler.csv|md   makaledeki her deneyin eğitim / doğrulama / test sayıları
- results/tables/veri_dagilimi_makale_eslesme.csv|md
- results/figures/veri_dagilimi.png|pdf          sınıf dağılımı (beyin katları, COVID-QU-Ex bölmeleri)
- results/figures/veri_kaggle_kopya.png|pdf      Kaggle kopya ayıklama: sınıf başına kopyası olan ve kalan
- results/figures/roi_alani_sinif.png|pdf        sınıf başına ROI alanı (Saldırı A'nın kullandığı meta verinin özü)

Çalıştırma: .venv\\Scripts\\python -m analysis.veri_dagilimi
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from common.report import write_markdown_table  # noqa: E402
from common.tez_bicim import SINIF_TEZ, kaydet, sayi  # noqa: E402

BEKLENEN = config.RESULTS / "beklenen" / "makale_degerleri.json"
# Referans paletin ilk üç kategorik yuvası (tüm çiftlerde renk körlüğü sınamasından geçer); sırası sabittir
RENK = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
       "surface": "#fcfcfb"}
BEYIN_SINIF = ["meningioma", "glioma", "pituitary"]
COVID_SINIF = ["COVID-19", "Non-COVID", "Normal"]
KAGGLE_SINIF = ["COVID19", "NORMAL", "PNEUMONIA"]
KAGGLE_AD = {"COVID19": "COVID-19", "NORMAL": "Normal", "PNEUMONIA": "Pnömoni"}


def yukle():
    beyin = pd.read_csv(config.DATA_PROC / "brain" / "meta.csv")
    cq = pd.read_csv(config.DATA_PROC / "covidqu_manifest.csv")
    cq = cq[cq.lung_mask_path.notna() & (cq.lung_mask_path != "")].reset_index(drop=True)
    kg = pd.read_csv(config.DATA_PROC / "kaggle_cxr_manifest.csv")
    kopya = pd.read_csv(config.DATA_PROC / "cxr_capraz_tekrarlar.csv")
    kg["covidqu_kopyasi"] = kg.index.isin(kopya.kaggle_idx.unique())
    return beyin, cq, kg


def tablo_beyin(beyin):
    t = pd.crosstab(beyin.fold, beyin.label_name)[BEYIN_SINIF]
    t.columns = [SINIF_TEZ[c] for c in BEYIN_SINIF]
    t["Toplam kesit"] = t.sum(axis=1)
    t["Hasta"] = beyin.groupby("fold").pid.nunique()
    t.loc["Toplam"] = list(t.drop(columns="Hasta").sum()) + [beyin.pid.nunique()]
    t.index = [f"Kat {i}" if i != "Toplam" else i for i in t.index]
    return t.reset_index(names="Kat")


def tablo_covidqu(cq):
    t = pd.crosstab(cq.split, cq.label)[COVID_SINIF].loc[["Train", "Val", "Test"]]
    t.loc["Toplam"] = t.sum()
    out = pd.DataFrame(index=t.index)
    for c in COVID_SINIF:
        out[SINIF_TEZ[c]] = [f"{sayi(n)} (%{100 * n / tot:.1f})" for n, tot in zip(t[c], t.sum(axis=1))]
    out["Toplam"] = [sayi(n) for n in t.sum(axis=1)]
    out.index = ["Eğitim (Train)", "Doğrulama (Val)", "Test", "Toplam"]
    return t, out.reset_index(names="Bölme")


def tablo_kaggle(kg):
    t = pd.crosstab(kg.split, kg.label)[KAGGLE_SINIF].loc[["train", "test"]]
    t.loc["toplam"] = t.sum()
    t.loc["COVID-QU-Ex'te kopyası olan"] = kg[kg.covidqu_kopyasi].label.value_counts().reindex(KAGGLE_SINIF).fillna(0)
    t.loc["Kopyasız (dış test kümesi)"] = kg[~kg.covidqu_kopyasi].label.value_counts().reindex(KAGGLE_SINIF).fillna(0)
    t = t.astype(int)
    t.columns = [KAGGLE_AD[c] for c in KAGGLE_SINIF]
    t["Toplam"] = t.sum(axis=1)
    t.index = ["Eğitim (train)", "Test (test)", "Toplam", "COVID-QU-Ex'te kopyası olan", "Kopyasız (dış test kümesi)"]
    return t.reset_index(names="Küme")


def tablo_deneyler(beyin, cq, kg):
    katlar = np.sort(beyin.fold.unique())
    boy = beyin.fold.value_counts()
    egitim_4 = [len(beyin) - boy[k] for k in katlar]
    tur = [(k, katlar[(i + 1) % len(katlar)]) for i, k in enumerate(katlar)]
    egitim_3 = [len(beyin) - boy[k] - boy[v] for k, v in tur]
    n = {s: int((cq.split == s).sum()) for s in ("Train", "Val", "Test")}
    aralik = lambda a: f"{sayi(min(a))}–{sayi(max(a))}"  # noqa: E731
    np_cq = cq.label.isin(["Normal", "Non-COVID"])  # gerçekçilik testi: normal ve pnömoni (ikili)
    np_kg = kg.label.isin(["NORMAL", "PNEUMONIA"])
    satir = [
        ("Saldırı A ve B (Tablo II, III)", "Beyin MR", "Hasta bazlı 5 kat; her turda 1 kat test, 4 kat eğitim",
         aralik(egitim_4), "–", aralik(boy.values), "Her kesit bir kez test edilir; ölçüler 5 katın birleşimi"),
        ("Saldırı A ve B (Tablo II, III)", "COVID-QU-Ex", "Resmi bölme; eğitim = Train + Val",
         sayi(n["Train"] + n["Val"]), "–", sayi(n["Test"]), "Test bölmesi eğitimde hiç kullanılmaz"),
        ("Saldırı A, girdi boyutu (Tablo II)", "Kaggle CXR", "Kümenin kendi train/test bölmesi; normal ve pnömoni",
         sayi(int(((kg.split == "train") & kg.label.isin(["NORMAL", "PNEUMONIA"])).sum())), "–",
         sayi(int(((kg.split == "test") & kg.label.isin(["NORMAL", "PNEUMONIA"])).sum())), "Yalnız görüntü boyutu"),
        ("Şifreli modeller (Tablo V, VIII)", "Beyin MR",
         "Hasta bazlı 5 kat; her turda 1 kat test, sonraki kat doğrulama, 3 kat eğitim", aralik(egitim_3),
         aralik(boy.values), aralik(boy.values), "Doğrulama kaybıyla erken durdurma ve ağırlık azaltma seçimi"),
        ("Şifreli modeller (Tablo V, VIII)", "COVID-QU-Ex", "Resmi Train / Val / Test", sayi(n["Train"]),
         sayi(n["Val"]), sayi(n["Test"]), "Doğrulama kaybıyla erken durdurma ve ağırlık azaltma seçimi"),
        ("U-Net akciğer segmentasyonu", "COVID-QU-Ex", "Resmi Train / Val", sayi(n["Train"]), sayi(n["Val"]), "–",
         "Kaggle görüntülerine akciğer maskesi üretmek için"),
        ("Gerçekçilik testi, yön A (metin)", "COVID-QU-Ex → Kaggle CXR",
         "Eğitim: COVID-QU-Ex Train + Val, normal ve COVID dışı pnömoni; test: kopyasız Kaggle, normal ve pnömoni",
         sayi(int((cq.split.isin(["Train", "Val"]) & np_cq).sum())), "–", sayi(int((~kg.covidqu_kopyasi & np_kg).sum())),
         "Saldırgan hastanenin verisini hiç görmez"),
        ("Gerçekçilik testi, yön B (metin)", "Kaggle CXR → COVID-QU-Ex",
         "Eğitim: kopyasız Kaggle train, normal ve pnömoni; test: COVID-QU-Ex Test, normal ve COVID dışı pnömoni",
         sayi(int(((kg.split == "train") & ~kg.covidqu_kopyasi & np_kg).sum())), "–",
         sayi(int(((cq.split == "Test") & np_cq).sum())), "Küçük ve başka kaynaklı eğitim kümesi"),
        ("Genelleme (metin)", "COVID-QU-Ex → Kaggle CXR", "Tablo V modelleri yeniden eğitilmeden kopyasız Kaggle'da",
         "–", "–", sayi(int((~kg.covidqu_kopyasi).sum())), "3 sınıf; akciğer maskesi U-Net'ten"),
    ]
    return pd.DataFrame(satir, columns=["Deney", "Veri", "Bölme düzeni", "Eğitim", "Doğrulama", "Test", "Not"])


def makale_eslesme(beyin, cq, kg) -> pd.DataFrame:
    mv = json.loads(BEKLENEN.read_text(encoding="utf-8"))["metin_veri"]
    hesap = {"covidqu_toplam": len(cq), "covidqu_covid19": int((cq.label == "COVID-19").sum()),
             "covidqu_non_covid": int((cq.label == "Non-COVID").sum()), "covidqu_normal": int((cq.label == "Normal").sum()),
             "covidqu_train": int((cq.split == "Train").sum()), "covidqu_val": int((cq.split == "Val").sum()),
             "covidqu_test": int((cq.split == "Test").sum()), "beyin_hasta": int(beyin.pid.nunique()),
             "beyin_kesit": len(beyin), "beyin_meningiom": int((beyin.label_name == "meningioma").sum()),
             "beyin_gliom": int((beyin.label_name == "glioma").sum()),
             "beyin_hipofiz": int((beyin.label_name == "pituitary").sum()), "kaggle_toplam": len(kg),
             "kaggle_kopyasi_olan": int(kg.covidqu_kopyasi.sum()), "kaggle_kopyasiz_dis_test": int((~kg.covidqu_kopyasi).sum())}
    rows = [{"kaynak": f"Makale metni (satır {mv['tex_satiri']} civarı): {k}", "makale": mv[k], "hesaplanan": v,
             "tuttu": mv[k] == v} for k, v in hesap.items()]
    cok_katli = int((beyin.groupby("pid").fold.nunique() > 1).sum())
    rows.append({"kaynak": "Makale metni: hiçbir hasta birden fazla katta değil", "makale": 0, "hesaplanan": cok_katli,
                 "tuttu": cok_katli == 0})
    return pd.DataFrame(rows)


def eksen_sade(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK["axis"])
    ax.tick_params(colors=INK["secondary"], labelsize=8.5, length=0)
    ax.yaxis.grid(True, color=INK["grid"], lw=0.6)
    ax.set_axisbelow(True)


def gruplu_cubuk(ax, gruplar, degerler, siniflar, baslik, toplam_yaz=True):
    """degerler: (grup, sınıf). Sınıflar sabit renk sırasıyla; grup toplamı grubun üstüne yazılır."""
    n_g, n_s = degerler.shape
    w = 0.26
    x = np.arange(n_g)
    for j in range(n_s):
        ax.bar(x + (j - (n_s - 1) / 2) * w, degerler[:, j], width=w - 0.03, color=RENK[j], label=siniflar[j], zorder=2)
    if toplam_yaz:
        for i in range(n_g):
            ax.text(x[i], degerler[i].max() * 1.03, f"toplam {sayi(degerler[i].sum())}", ha="center", va="bottom",
                    fontsize=8, color=INK["secondary"])
    ax.set_xticks(x, gruplar)
    ax.set_title(baslik, fontsize=10, color=INK["primary"], loc="left")
    ax.set_ylabel("Görüntü sayısı", fontsize=9, color=INK["secondary"])
    ax.set_ylim(0, degerler.max() * 1.32)  # üstte gösterge ve toplam yazıları için boşluk
    eksen_sade(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", ncols=3, labelcolor=INK["primary"])


def sekil_dagilim(beyin, cq_sayi):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), facecolor=INK["surface"], width_ratios=[5, 3.2])
    tb = pd.crosstab(beyin.fold, beyin.label_name)[BEYIN_SINIF]
    hasta = beyin.groupby("fold").pid.nunique()
    gruplu_cubuk(axes[0], [f"Kat {k}\n({hasta[k]} hasta)" for k in tb.index], tb.to_numpy(),
                 [SINIF_TEZ[c] for c in BEYIN_SINIF],
                 f"Beyin MR: hasta bazlı 5 kat ({sayi(len(beyin))} kesit, {beyin.pid.nunique()} hasta)")
    tc = cq_sayi.loc[["Train", "Val", "Test"], COVID_SINIF]
    gruplu_cubuk(axes[1], ["Eğitim\n(Train)", "Doğrulama\n(Val)", "Test"], tc.to_numpy(),
                 [SINIF_TEZ[c] for c in COVID_SINIF], f"COVID-QU-Ex: resmi bölmeler ({sayi(tc.to_numpy().sum())} görüntü)")
    for ax in axes:
        ax.set_facecolor(INK["surface"])
    fig.tight_layout()
    kaydet(fig, config.FIGURES / "veri_dagilimi", facecolor=INK["surface"])
    plt.close(fig)


def sekil_kaggle(kg):
    fig, ax = plt.subplots(figsize=(7.5, 3.0), facecolor=INK["surface"])
    y = np.arange(len(KAGGLE_SINIF))[::-1]
    kopya = kg[kg.covidqu_kopyasi].label.value_counts().reindex(KAGGLE_SINIF).fillna(0).to_numpy()
    kalan = kg[~kg.covidqu_kopyasi].label.value_counts().reindex(KAGGLE_SINIF).fillna(0).to_numpy()
    ax.barh(y, kalan, color=RENK[0], height=0.55, label="Kopyasız: dış test kümesinde kullanılan", zorder=2)
    ax.barh(y, kopya, left=kalan + kalan.max() * 0.004, color=INK["axis"], height=0.55,
            label="COVID-QU-Ex'te kopyası olan: dışarıda bırakılan", zorder=2)
    for yi, a, b in zip(y, kalan, kopya):
        ax.text(a / 2, yi, sayi(a), ha="center", va="center", fontsize=8.5, color="#ffffff")
        ax.text(a + b + kalan.max() * 0.02, yi, f"{sayi(a + b)} toplam", ha="left", va="center", fontsize=8.5,
                color=INK["secondary"])
    ax.set_yticks(y, [KAGGLE_AD[c] for c in KAGGLE_SINIF])
    ax.set_xlabel("Görüntü sayısı", fontsize=9, color=INK["secondary"])
    ax.set_xlim(0, (kalan + kopya).max() * 1.18)
    ax.set_title(f"Kaggle CXR: {sayi(len(kg))} görüntünün {sayi(int(kopya.sum()))}'inin COVID-QU-Ex'te kopyası var; "
                 f"dış test {sayi(int(kalan.sum()))}", fontsize=9.5, color=INK["primary"], loc="left")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK["axis"])
    ax.tick_params(colors=INK["secondary"], labelsize=8.5, length=0)
    ax.xaxis.grid(True, color=INK["grid"], lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK["primary"])  # en kısa çubuğun hizası boş
    ax.set_facecolor(INK["surface"])
    fig.tight_layout()
    kaydet(fig, config.FIGURES / "veri_kaggle_kopya", facecolor=INK["surface"])
    plt.close(fig)


def sekil_roi_alani(beyin, cq):
    oz = config.DATA_PROC / "saldiri_A_oznitelikler_covidqu.csv"
    if not oz.exists():
        raise SystemExit("Önce `python -m analysis.saldiri_a_tahmin` çalıştırılmalı (COVID-QU-Ex ROI öznitelikleri).")
    f = pd.read_csv(oz)
    if not (f.file.to_numpy() == cq.file.to_numpy()).all():
        raise RuntimeError("öznitelik önbelleği manifestoyla aynı sırada değil")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=INK["surface"])
    paneller = [(axes[0], [100 * beyin.tumor_frac[beyin.label_name == c] for c in BEYIN_SINIF],
                 [SINIF_TEZ[c] for c in BEYIN_SINIF], "Beyin MR: tümör maskesinin alanı", True),
                (axes[1], [100 * f.area_frac[cq.label == c] for c in COVID_SINIF],
                 [SINIF_TEZ[c] for c in COVID_SINIF], "COVID-QU-Ex: akciğer maskesinin alanı", False)]
    for ax, veri, adlar, baslik, logy in paneller:
        bp = ax.boxplot(veri, widths=0.5, patch_artist=True, showfliers=True,
                        medianprops=dict(color=INK["primary"], lw=1.4),
                        flierprops=dict(marker="o", ms=2, mfc=INK["muted"], mec="none", alpha=0.5),
                        whiskerprops=dict(color=INK["secondary"], lw=0.9), capprops=dict(color=INK["secondary"], lw=0.9))
        for patch, renk in zip(bp["boxes"], RENK):
            patch.set_facecolor(renk)
            patch.set_alpha(0.85)
            patch.set_edgecolor(INK["surface"])
        ax.set_xticks(range(1, len(adlar) + 1), adlar)
        if logy:
            ax.set_yscale("log")
            ax.set_yticks([0.1, 1, 10], ["%0.1", "%1", "%10"])
            ax.tick_params(which="minor", length=0)
        else:
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"%{v:.0f}"))
        ax.set_ylabel("Görüntü alanına oranı", fontsize=9, color=INK["secondary"])
        ax.set_title(baslik, fontsize=10, color=INK["primary"], loc="left")
        eksen_sade(ax)
        ax.set_facecolor(INK["surface"])
    fig.suptitle("Sınıf başına ROI alanı: sunucunun Π_ROI'de açıkça gördüğü meta verinin bir parçası (Saldırı A)",
                 fontsize=10, color=INK["secondary"])
    fig.tight_layout()
    kaydet(fig, config.FIGURES / "roi_alani_sinif", facecolor=INK["surface"])
    plt.close(fig)


def main():
    beyin, cq, kg = yukle()
    kontrol = makale_eslesme(beyin, cq, kg)
    kontrol.to_csv(config.TABLES / "veri_dagilimi_makale_eslesme.csv", index=False)
    write_markdown_table(kontrol.assign(tuttu=kontrol.tuttu.map({True: "evet", False: "HAYIR"})),
                         config.TABLES / "veri_dagilimi_makale_eslesme.md")
    print(f"Makale eşleşmesi (veri sayıları): {int(kontrol.tuttu.sum())}/{len(kontrol)}")
    if not kontrol.tuttu.all():
        print(kontrol[~kontrol.tuttu].to_string(index=False))
        raise SystemExit("Makaledeki veri sayılarıyla tutmayan değer var.")
    tb = tablo_beyin(beyin)
    cq_sayi, tc = tablo_covidqu(cq)
    tk = tablo_kaggle(kg)
    td = tablo_deneyler(beyin, cq, kg)
    for df, ad in ((tb, "beyin"), (tc, "covidqu"), (tk, "kaggle"), (td, "deneyler")):
        df.to_csv(config.TABLES / f"veri_dagilimi_{ad}.csv", index=False)
        write_markdown_table(df, config.TABLES / f"veri_dagilimi_{ad}.md")
    sekil_dagilim(beyin, cq_sayi)
    sekil_kaggle(kg)
    sekil_roi_alani(beyin, cq)
    for df in (tb, tc, tk, td):
        print(df.to_string(index=False), "\n")


if __name__ == "__main__":
    main()
