"""Ezber denetimi: makaledeki sonuçlar ezbere mi dayanıyor? (danışman maddesi 4)

Ezber, modelin eğitim görüntülerini etiketleriyle birlikte akılda tutmasıdır. Eğitim başarısını yükseltir ama yeni
görüntüye taşınmaz. Makaledeki bütün sayılar eğitimde ve model seçiminde kullanılmayan test görüntülerinden ölçülür.
Bu betik bunun üstüne doğrudan bir denetim yapar: etiket karıştırma (permütasyon) testi.

Modelin eğitimde gördüğü bütün etiketler (eğitim ve varsa doğrulama) kendi aralarında rastgele karıştırılır.
Görüntüler, model, ayarlar ve bölmeler makaledekiyle aynıdır. Böyle eğitilen model etiketleri ancak ezberleyebilir;
görüntü ile gerçek teşhis arasında öğrenebileceği bir ilişki yoktur. Test görüntüleri gerçek etiketleriyle ölçülür.
Bu, farklı karıştırmalarla birçok kez tekrarlanır ve "yalnız ezberle ulaşılabilen" test AUC'lerinin dağılımı çıkar.
Gerçek etiketlerle eğitilen modelin (makaledeki model) test AUC'si bu dağılımın bütün değerlerinden yüksekse sonuç
ezberle açıklanamaz.

Karıştırılmış etiketli tek bir koşunun AUC'si 0.5'ten belirgin biçimde sapabilir (iki yönde de). Rastgele öğrenilen
ağırlık yönü, görüntülerdeki sınıfla ilişkili güçlü yapıya (ör. tümörün büyüklüğü ve konumu) rastgele denk gelebilir.
Bu yüzden karşılaştırma tek koşuyla değil, tekrarların dağılımıyla yapılır.

Denetlenen modeller (makalenin ana modelleri, eğitim tohumu 0, makaledeki ilk bölme):
- Bağlam saldırganı (ResNet-18), Π_ROI görüşü: beyin MR kat 1 (12 epoch), COVID-QU-Ex resmi Test (5 epoch).
- FoveaHE F32_G16: beyin MR Model D (kat 1), COVID-QU-Ex Model D2 (resmi Test).
Gerçek etiketli sonuç, makaledeki kayıtlı tahminlerden (`results/tahminler/`) aynı test görüntülerinde hesaplanır.

Çıktılar: `results/tables/ezber_denetimi.csv|md` (özet), `ezber_denetimi_kosular.csv` (her karıştırma).
Süre: RTX 2070'te ~19 dk (saldırgan 5 + 3 tekrar, FoveaHE 20 + 20 tekrar).
Çalıştırma: python -m boruhatti.ezber_denetimi
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import config
from boruhatti.ortak import aygit, is_parcacigi
from common.evaluation import auc_score
from common.report import write_markdown_table

TOHUM = 0  # eğitim tohumu (makaledeki koşuyla aynı); karıştırmalar 1000 + k tohumuyla
VERI_AD = {"beyin": "Beyin MR", "covidqu": "COVID-QU-Ex"}
TEKRAR = {("saldirgan", "beyin"): 5, ("saldirgan", "covidqu"): 3, ("foveahe", "beyin"): 20, ("foveahe", "covidqu"): 20}
MAKALE_TAHMINI = {("saldirgan", "beyin"): "beyin_baglam_saldirgani_piroi_s0.csv",
                  ("saldirgan", "covidqu"): "covidqu_baglam_saldirgani_piroi_s0.csv",
                  ("foveahe", "beyin"): "beyin_model_D_foveahe_F32_G16_s0.csv",
                  ("foveahe", "covidqu"): "covidqu_model_D2_foveahe_F32_G16_s0.csv"}


def karistir(y: np.ndarray, gorulen: np.ndarray, tohum: int) -> np.ndarray:
    """Modelin gördüğü etiketleri kendi aralarında karıştırır; test etiketlerine dokunmaz."""
    yk = y.copy()
    yk[gorulen] = np.random.default_rng(tohum).permutation(y[gorulen])
    return yk


def makaledeki_auc(tur: str, veri: str, n_test: int) -> float:
    """Makaledeki kayıtlı tahminlerden aynı test bölmesinin AUC'si (beyin MR kat 1, COVID-QU-Ex resmi Test)."""
    df = pd.read_csv(config.RESULTS / "tahminler" / MAKALE_TAHMINI[(tur, veri)], encoding="utf-8-sig")
    df = df[df["kat"] == 1] if veri == "beyin" else df[df["resmi_bolme"] == "Test"]
    if len(df) != n_test:
        raise SystemExit(f"{MAKALE_TAHMINI[(tur, veri)]}: test görüntüsü sayısı {len(df)}, beklenen {n_test}")
    return auc_score(df["gercek_etiket_no"].to_numpy(), df[[c for c in df if c.startswith("olasilik_")]].to_numpy())


def saldirgan_kosucu(veri: str, dev: str):
    """Veriyi bir kez yükler; her çağrıda verilen karıştırmayla eğitip test olasılıklarını döndürür."""
    from attacks.context_cnn import train_and_predict
    from boruhatti.egit import EPOCH, SALDIRGAN_LR, SALDIRGAN_YIGIN, saldirgan_bolmeleri, saldirgan_verisi
    cache, y, bilgi, etiketler = saldirgan_verisi(veri, "gorunur" if veri == "beyin" else "dagitildigi_gibi", dev)
    _, tr, _, te = saldirgan_bolmeleri(veri, "tez", bilgi)[0]

    def kos(tohum: int) -> dict:
        yk = karistir(y, tr, tohum)
        cache.y = torch.from_numpy(yk.astype(np.int64))  # eğitim yığınları etiketi buradan alır
        egitim = []
        p, _ = train_and_predict(cache, tr, te, "baglam", len(etiketler), epochs=EPOCH[veri], batch_size=SALDIRGAN_YIGIN,
                                 lr=SALDIRGAN_LR, seed=TOHUM, log=lambda *a: None,
                                 izle=lambda olay, b: egitim.append(b["egitim_dogrulugu"]) if olay == "epoch" else None)
        return {"p": p, "yk": yk, "egitim_dogrulugu": egitim[-1], "ayar": f"{EPOCH[veri]} epoch"}

    return "Bağlam saldırganı (ResNet-18), Π_ROI görüşü", y, tr, te, kos


def foveahe_kosucu(veri: str, dev: str):
    from experiments.fovea_models import MAX_EPOCHS, PATIENCE, VectorData, fit_select, predict, splits
    from foveahe.data import load_dataset
    from foveahe.representation import FoveaSpec
    ds = load_dataset({"beyin": "brain", "covidqu": "covidqu"}[veri])
    spec = FoveaSpec.parse("F32_G16")
    data = VectorData(ds, spec, dev)
    _, tr, va, te = splits(ds, quick=False)[0]
    tur, n = ("D" if veri == "beyin" else "D2"), len(ds.labels)

    def kos(tohum: int) -> dict:
        yk = karistir(ds.y, np.concatenate([tr, va]), tohum)
        model, ep, wd, _ = fit_select(tur, spec, data, yk, tr, va, n, TOHUM, dev, MAX_EPOCHS, PATIENCE)
        p_tr = predict(model, data, tr, dev, TOHUM, n)
        return {"p": predict(model, data, te, dev, TOHUM, n), "yk": yk,
                "egitim_dogrulugu": float((p_tr.argmax(1) == yk[tr]).mean()), "ayar": f"wd={wd:g}, epoch {ep}"}

    return f"FoveaHE F32_G16, Model {tur}", ds.y, tr, te, kos


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--modeller", nargs="*", default=["saldirgan", "foveahe"], choices=["saldirgan", "foveahe"])
    ap.add_argument("--veri", nargs="*", default=["beyin", "covidqu"], choices=list(VERI_AD))
    ap.add_argument("--tekrar", type=int, default=None, help="her model için karıştırma sayısı (varsayılan: TEKRAR)")
    ap.add_argument("--aygit", default=None)
    args = ap.parse_args()
    is_parcacigi(4)
    dev = aygit(args.aygit)
    ozet, kosular = [], []
    for tur in args.modeller:
        for veri in args.veri:
            t0 = time.perf_counter()
            ad, y, tr, te, kos = (saldirgan_kosucu if tur == "saldirgan" else foveahe_kosucu)(veri, dev)
            gercek = makaledeki_auc(tur, veri, len(te))
            k_sayisi = args.tekrar or TEKRAR[(tur, veri)]
            print(f"[ezber denetimi] {VERI_AD[veri]}, {ad}: gerçek etiketle test AUC {gercek:.4f}; "
                  f"{k_sayisi} karıştırma", flush=True)
            aucs, egitim = [], []
            for k in range(k_sayisi):
                r = kos(1000 + k)
                auc = auc_score(y[te], r["p"])
                sinif = [roc_auc_score(y[te] == c, r["p"][:, c]) for c in range(r["p"].shape[1])]
                aucs.append(auc)
                egitim.append(r["egitim_dogrulugu"])
                kosular.append({"veri": VERI_AD[veri], "model": ad, "karistirma": k, "karistirma_tohumu": 1000 + k,
                                "karisik_egitim_dogrulugu": r["egitim_dogrulugu"], "test_auc": auc,
                                **{f"sinif{c}_auc": v for c, v in enumerate(sinif)}, "ayar": r["ayar"]})
                print(f"  karıştırma {k}: eğitim doğruluğu (karışık etikete göre) {r['egitim_dogrulugu']:.3f}, "
                      f"test AUC {auc:.4f} [{r['ayar']}]", flush=True)
            a = np.array(aucs)
            std = float(a.std(ddof=1)) if len(a) > 1 else float("nan")
            ozet.append({"veri": VERI_AD[veri], "model": ad, "test_bolmesi": "kat 1" if veri == "beyin" else "resmi Test",
                         "egitim_n": len(tr), "test_n": len(te), "gercek_etiketle_test_auc": gercek,
                         "karistirma_sayisi": len(a), "karisik_egitim_dogrulugu_ort": float(np.mean(egitim)),
                         "cogunluk_sinifi_orani": float(np.bincount(y[tr]).max() / len(tr)),
                         "karisik_test_auc_ort": float(a.mean()), "karisik_test_auc_std": std,
                         "karisik_test_auc_en_dusuk": float(a.min()), "karisik_test_auc_en_yuksek": float(a.max()),
                         "z": (gercek - a.mean()) / std if std > 0 else float("nan"),
                         "p": (1 + int((a >= gercek).sum())) / (len(a) + 1),
                         "sonuc": ("gerçek etiketli sonuç bütün karıştırılmış koşuların üstünde" if gercek > a.max()
                                   else "AYIRT EDİLEMEDİ"),
                         "sure_s": round(time.perf_counter() - t0)})
            print(f"  → karışık ort. {a.mean():.4f} (en düşük {a.min():.4f}, en yüksek {a.max():.4f}); "
                  f"{ozet[-1]['sonuc']}", flush=True)
            del kos
            torch.cuda.empty_cache() if dev == "cuda" else None
    pd.DataFrame(kosular).to_csv(config.TABLES / "ezber_denetimi_kosular.csv", index=False)
    df = pd.DataFrame(ozet)
    df.to_csv(config.TABLES / "ezber_denetimi.csv", index=False)
    write_markdown_table(df, config.TABLES / "ezber_denetimi.md", floatfmt="{:.4f}")
    print(f"\nYazıldı: {config.TABLES / 'ezber_denetimi.csv'}, {config.TABLES / 'ezber_denetimi_kosular.csv'}")


if __name__ == "__main__":
    main()
