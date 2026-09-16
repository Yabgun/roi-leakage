"""Tüm deney çıktılarını tek bir özet dosyasında toplar: results/OZET.md

Çalıştırma: .venv\\Scripts\\python -m experiments.summarize
"""
from __future__ import annotations

import json

import pandas as pd

import config


def _csv(name):
    p = config.TABLES / name
    return pd.read_csv(p) if p.exists() else None


def _json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _md(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    def fmt(v):
        return floatfmt.format(v) if isinstance(v, float) and not pd.isna(v) else ("" if pd.isna(v) else str(v))
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)) + "\n"


def main():
    out = ["# Deney Özeti (otomatik)\n"]

    veri = _json(config.DATA_PROC / "veri_ozeti.json")
    if veri:
        out.append("## Veri\n")
        out.append(f"- Beyin MR: {veri.get('beyin_kesit')} kesit, {veri.get('beyin_hasta')} hasta, "
                   f"sınıflar {veri.get('beyin_sinif')}, birden fazla katta görünen hasta: {veri.get('beyin_cok_katli_hasta')}\n")
        out.append(f"- COVID-QU-Ex: {veri.get('covidqu')}\n")
        out.append(f"- Kaggle CXR: {veri.get('kaggle_cxr')}\n")
        if "kopyasi_olan_kaggle" in veri:
            out.append(f"- COVID-QU-Ex'te kopyası olan Kaggle görüntüsü: {veri['kopyasi_olan_kaggle']}/{veri['kaggle_toplam']}\n")

    ok, cost = _csv("piroi_dogruluk.csv"), _csv("piroi_maliyet.csv")
    if ok is not None or cost is not None:
        out.append("\n## Π_ROI yeniden üretimi\n")
        if ok is not None:
            out.append(_md(ok, "{:.3g}"))
        if cost is not None:
            cols = ["size", "rho", "ciphertext_sayisi", "sifreli_slot", "sifreleme_s", "sunucu_s", "toplam_s",
                    "hiz_kazanci_toplam", "yukleme_MB"]
            out.append("\n" + _md(cost[[c for c in cols if c in cost]]))
            out.append("\n![](figures/piroi_hiz_kazanci.png)\n")

    a = _csv("saldiri_A_meta_veri.csv")
    if a is not None:
        out.append("\n## Saldırı A: yalnızca ROI meta verisi\n")
        out.append(_md(a[["veri", "hedef", "roi", "oznitelik_grubu", "auc_ort", "ci95_alt", "ci95_ust", "n"]]))
        per = _json(config.TABLES / "saldiri_A_beyin_sinif_auc.json")
        if per:
            out.append(f"\nBeyin, sınıf bazında (bire karşı hepsi) AUC: {per}\n")
        yon = _json(config.TABLES / "saldiri_A_beyin_yon_kontrolu_heuristik.json")
        if yon:
            out.append(f"\nKesit yönü kontrolü: yalnızca yön AUC={yon['yalniz_yon_auc']:.3f}, yalnızca meta veri "
                       f"AUC={yon['yalniz_meta_auc']:.3f}, ikisi birlikte AUC={yon['yon_arti_meta_auc']:.3f}; "
                       f"yön içinde: " + ", ".join(f"{k} {v['meta_veri_auc']:.3f} (n={v['n']})"
                                                   for k, v in yon["yon_icinde"].items()) +
                       ". Görsel doğrulama (12'şer rastgele örnek): sagital 12/12, aksiyel ~9-10/12, koronal ~8/12.\n")

    b = _csv("saldiri_B_baglam.csv")
    if b is not None:
        out.append("\n## Saldırı B: bağlam (CNN)\n")
        cols = ["veri", "gorus", "tohum", "gizli_alan_ort", "auc", "ci95_alt", "ci95_ust"] + \
               [c for c in b.columns if c.startswith("auc_") and "vs" in c]
        out.append(_md(b[cols]))

    seg = _json(config.TABLES / "akciger_segmentasyon.json")
    t = _csv("saldiri_B_transfer.csv")
    if seg or t is not None:
        out.append("\n## Gerçekçilik testi\n")
        if seg:
            out.append(f"- Akciğer segmentasyonu: {seg}\n")
        if t is not None:
            out.append(_md(t[["yon", "gorus", "tohum", "auc", "ci95_alt", "ci95_ust", "n_egitim", "n_test"]]))

    cam, abl = _csv("kok_neden_gradcam.csv"), _csv("kok_neden_onisleme.csv")
    if cam is not None or abl is not None:
        out.append("\n## Kök neden\n")
        if cam is not None:
            out.append("Grad-CAM dikkat kütlesinin bölgelere dağılımı:\n\n" + _md(cam))
        if abl is not None:
            out.append("\nÖnişleme ablasyonu:\n\n" + _md(abl))
        out.append("\n![](figures/kok_neden_gradcam_covidqu.png)\n\n![](figures/kok_neden_gradcam_brain.png)\n")

    d = _csv("savunma.csv")
    if d is not None:
        out.append("\n## Savunma ve gerçek bedel\n")
        cols = ["veri", "politika", "adim", "gizli_alan", "auc"] + [c for c in d.columns if c.startswith("hiz_kazanci")]
        out.append(_md(d[cols]))
        s = _json(config.TABLES / "savunma_ozet.json")
        if s:
            out.append("\n```json\n" + json.dumps(s, ensure_ascii=False, indent=2) + "\n```\n")
        out.append("\n![](figures/savunma_covidqu.png)\n\n![](figures/savunma_brain.png)\n")

    # Çözüm adımlarının tabloları: sütunları yeniden türetmek yerine adım betiklerinin yazdığı markdown gömülür.
    cozum = [("Adım 1 — bilgi kaybı: hangi temsil yetiyor?", ["cozum_bilgi.md"],
              ["cozum_bilgi_egrisi.png", "cozum_temsil_ornek_brain.png", "cozum_temsil_ornek_covidqu.png"]),
             ("Adım 2 — şifreli çalışabilen modeller (D, D2, C)", ["cozum_modeller.md"], []),
             ("Adım 3 — şifreli doğruluk eşleşmesi ve maliyet", ["cozum_dogruluk_eslesme.md", "cozum_maliyet.md"], []),
             ("Adım 4 — sızıntı denetimi: sunucunun gördüğü", ["cozum_sizinti.md"], []),
             ("Adım 5a — rakip: şifreli özet (HETAL tarzı)", ["cozum_rakipler.md"], []),
             ("Adım 5b — rakip: bozuk açık bağlam (Bi-CryptoNets tarzı)", ["saldiri_B_gurultu.md"], []),
             ("Adım 6 — birleşik özet: hız × doğruluk × sızıntı", ["cozum_ozet.md"],
              ["cozum_pareto_brain.png", "cozum_pareto_covidqu.png"]),
             ("Bonus — bütçe duyarlı odak: sabit şifreli değer bütçesinde odak payı", ["cozum_butce_en_iyi.md"],
              ["cozum_butce.png"]),
             ("Bonus — harici derlemede genelleme (Kaggle CXR, kopyalar çıkarılmış)", ["cozum_genelleme.md"],
              ["cozum_genelleme.png"])]
    cozum = [(t, [m for m in f if (config.TABLES / m).exists()], [g for g in p if (config.FIGURES / g).exists()])
             for t, f, p in cozum]
    if any(f or p for _, f, p in cozum):
        out.append("\n## Çözüm: Odaklı Tam Şifreleme (FoveaHE)\n")
        out.append("\nSeçicilik şifrelemede değil çözünürlükte: istemci ROI merkezli, sabit boyutlu, çok çözünürlüklü "
                   "bir temsil çıkarır ve tamamını şifreler; sunucuya açık piksel ya da değişken meta veri gitmez.\n")
        for title, files, figs in cozum:
            if not files and not figs:
                continue
            out.append(f"\n### {title}\n\n")
            out += [(config.TABLES / m).read_text(encoding="utf-8") + "\n" for m in files]
            out += [f"\n![](figures/{g})\n" for g in figs]

    (config.RESULTS / "OZET.md").write_text("".join(out), encoding="utf-8")
    print("".join(out))


if __name__ == "__main__":
    main()
