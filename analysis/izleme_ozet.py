"""İzleme koşularının özeti: eğitim oldu mu, ezberleme var mı (danışman maddesi 4).

`boruhatti.egit --protokol izleme` koşularında model, test kümesine hiç dokunmadan, eğitimden ayrı bir doğrulama
kümesiyle her epoch'ta ölçülmüştür. Bu betik koşuların eğrilerini (`modeller/*_izleme_*/<bölüm>/egitim_egrisi_*.csv`)
toplar. Ham eğrileri `results/tables/izleme/` altına kopyalar ve şunları üretir:
- `results/tables/izleme_ozet.csv|md`: son ve en iyi epoch'ta eğitim/doğrulama kaybı ve doğruluğu, aradaki fark,
  doğrulama kaybının en düşük olduğu epoch.
- `results/figures/izleme_saldirgan.png|pdf` ve `izleme_foveahe.png|pdf`: eğitim ve doğrulama eğrileri.

Şifreli modellerde (D, D2, C) her ağırlık azaltma adayı ayrı eğitilir. Burada yalnız seçilen adayın eğrisi çizilir;
eğitim, doğrulama kaybı 10 epoch iyileşmeyince durur ve en iyi epoch'un ağırlıkları kullanılır.

Çalıştırma: .venv\\Scripts\\python -m analysis.izleme_ozet
"""
from __future__ import annotations

import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

import config  # noqa: E402
from boruhatti.ortak import MODELLER, json_oku  # noqa: E402
from common.report import write_markdown_table  # noqa: E402
from common.tez_bicim import kaydet  # noqa: E402

RENK = {"egitim": "#2a78d6", "dogrulama": "#eb6834", "ink": "#52514e", "grid": "#e1e0d9", "surface": "#fcfcfb"}
VERI_AD = {"beyin": "Beyin MR", "covidqu": "COVID-QU-Ex"}


def kosular() -> list[dict]:
    out = []
    for klasor in sorted(MODELLER.glob("*_izleme_s0")):
        ayar, sonuc = json_oku(klasor / "ayar.json"), json_oku(klasor / "sonuc.json")
        for bolum_klasor in sorted(p for p in klasor.iterdir() if p.is_dir()):
            ep = pd.read_csv(bolum_klasor / "egitim_egrisi_epoch.csv")
            onek, secilen_epoch = "", None
            if "secim" in sonuc and bolum_klasor.name in sonuc["secim"]:
                s = sonuc["secim"][bolum_klasor.name]
                onek = f"wd{s['agirlik_azaltma']:g}_lr{s['ogrenme_orani']:g}/"
                secilen_epoch = int(s["en_iyi_epoch"])  # erken durdurmanın döndürdüğü ağırlıkların epoch'u
                ep = ep[ep["onek"] == onek]
            hedef = config.TABLES / "izleme" / klasor.name
            hedef.mkdir(parents=True, exist_ok=True)
            for f in bolum_klasor.glob("egitim_egrisi_*.csv"):
                shutil.copy2(f, hedef / f.name)
            if ayar["model"] == "saldirgan":
                ad = f"Bağlam saldırganı (ResNet-18), {'Π_ROI görüşü' if ayar['gorus'] == 'baglam' else ayar['gorus']}"
                girdi = ad
            else:
                ad = f"{ayar['model']}, {ayar['temsil']}"
                girdi = ad
            b = ayar["bolmeler"][bolum_klasor.name]
            out.append({"klasor": klasor.name, "veri": VERI_AD[ayar["veri"]], "model": ad, "girdi": girdi,
                        "aile": "saldirgan" if ayar["model"] == "saldirgan" else "foveahe", "egitim_n": b["egitim"],
                        "dogrulama_n": b["dogrulama"], "egri": ep.reset_index(drop=True), "secim": onek.rstrip("/"),
                        "secilen_epoch": secilen_epoch})
    return out


def ozet_satiri(k: dict) -> dict:
    e = k["egri"]
    son = e.iloc[-1]
    en_iyi = e.loc[e["dogrulama_kaybi"].idxmin()]
    if k["aile"] == "foveahe":
        kullanilan = e[e["epoch"] == k["secilen_epoch"]].iloc[0]
        ne = f"epoch {k['secilen_epoch']} (erken durdurma; doğrulama kaybı 1e-4'ten fazla iyileşmeyince 10 epoch sonra durur)"
    else:
        kullanilan, ne = son, "son epoch (sabit epoch)"
    return {"veri": k["veri"], "model": k["model"], "secilen_ayar": k["secim"] or "–", "egitim_n": k["egitim_n"],
            "dogrulama_n": k["dogrulama_n"], "epoch_sayisi": int(e["epoch"].max()),
            "dogrulama_kaybi_en_dusuk_epoch": int(en_iyi["epoch"]), "kullanilan_agirlik": ne,
            "egitim_dogrulugu": float(kullanilan["egitim_dogrulugu"]), "dogrulama_dogrulugu": float(kullanilan["dogrulama_dogrulugu"]),
            "dogruluk_farki": float(kullanilan["egitim_dogrulugu"] - kullanilan["dogrulama_dogrulugu"]),
            "egitim_kaybi": float(kullanilan["egitim_kaybi"]), "dogrulama_kaybi": float(kullanilan["dogrulama_kaybi"]),
            "dogrulama_kaybi_en_dusuk": float(e["dogrulama_kaybi"].min()),
            "son_epoch_dogrulama_kaybi_artisi": float(son["dogrulama_kaybi"] - e["dogrulama_kaybi"].min()),
            "dogrulama_f1_makro": float(kullanilan["dogrulama_f1_makro"]), "dogrulama_auc": float(kullanilan["dogrulama_auc"])}


def sekil(kosu_listesi: list[dict], dosya: str, baslik: str) -> None:
    n = len(kosu_listesi)
    fig, axes = plt.subplots(n, 2, figsize=(9.5, 2.6 * n), facecolor=RENK["surface"], squeeze=False)
    for r, k in enumerate(kosu_listesi):
        e = k["egri"]
        for c, (sutun, ad) in enumerate((("kaybi", "Kayıp"), ("dogrulugu", "Doğruluk"))):
            ax = axes[r, c]
            ax.plot(e["epoch"], e[f"egitim_{sutun}"], "-o", ms=3, lw=1.6, color=RENK["egitim"], label="eğitim")
            ax.plot(e["epoch"], e[f"dogrulama_{sutun}"], "-o", ms=3, lw=1.6, color=RENK["dogrulama"], label="doğrulama")
            if k["aile"] == "foveahe":
                ax.axvline(k["secilen_epoch"], color=RENK["ink"], lw=0.8, ls=":")
            ax.set_title(f"{k['veri']} · {k['model']} · {ad}", fontsize=8.5, color="#0b0b0b", loc="left")
            ax.grid(True, color=RENK["grid"], lw=0.6)
            ax.set_facecolor(RENK["surface"])
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(labelsize=7.5, colors=RENK["ink"])
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel("epoch", fontsize=7.5, color=RENK["ink"])
        axes[r, 0].legend(frameon=False, fontsize=7.5)
    h = fig.get_figheight()  # üst boşluk inç cinsinden sabit: satır sayısı değişince başlık kaymaz
    fig.suptitle(baslik, fontsize=10, color="#0b0b0b", y=1 - 0.12 / h, va="top")
    fig.tight_layout()
    fig.subplots_adjust(top=1 - 0.6 / h)
    kaydet(fig, config.FIGURES / dosya, facecolor=RENK["surface"])
    plt.close(fig)


def main():
    k = kosular()
    if not k:
        raise SystemExit("İzleme koşusu bulunamadı: önce `python -m boruhatti.egit ... --protokol izleme`.")
    ozet = pd.DataFrame([ozet_satiri(x) for x in k])
    ozet.to_csv(config.TABLES / "izleme_ozet.csv", index=False)
    write_markdown_table(ozet, config.TABLES / "izleme_ozet.md", floatfmt="{:.4f}")
    sekil([x for x in k if x["aile"] == "saldirgan"], "izleme_saldirgan",
          "Bağlam saldırganı: eğitim ve doğrulama eğrileri (izleme koşusu; test kümesi kullanılmadı)")
    sekil([x for x in k if x["aile"] == "foveahe"], "izleme_foveahe",
          "Şifreli modeller: seçilen ayarın eğitim ve doğrulama eğrileri (noktalı çizgi: erken durdurmanın seçtiği epoch)")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(ozet.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
