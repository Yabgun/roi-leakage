"""Makaledeki sınıflandırıcılar için kesinlik, duyarlılık, F1 ve karışıklık matrisi (danışman maddesi 3).

Yeniden eğitim yapılmaz. Makaledeki tabloları üreten koşular her görüntü için sınıf olasılıklarını kaydetmiştir
(`results/preds`). Tahmin edilen sınıf, olasılığı en yüksek sınıftır. Bu tahminlerden şunlar hesaplanır: doğruluk,
dengeli doğruluk, kesinlik, duyarlılık ve F1 (sınıf başına, makro ve ağırlıklı ortalama) ile karışıklık matrisi.
Aynı dosyalardan AUC de yeniden hesaplanır ve makalede basılı değerle karşılaştırılır
(`results/beklenen/makale_degerleri.json`). Tek bir değer bile tutmazsa betik hata verir ve tablo yazmaz.

Kapsam, makalenin sınıflandırma sonucu veren tabloları ve metnidir:
- Tablo II: Saldırı A, meta veri saldırganı. Tahminleri `analysis.saldiri_a_tahmin` üretir.
- Tablo III: Saldırı B, bağlam saldırganı (ResNet-18). Beyin MR'da görünür piksel normalizasyonu kullanılır.
- Tablo V: şifreli çalışabilen modeller (D, D2, C).
- Tablo VIII: teşhis modelleri, şifreli özet ve bozuk bağlam saldırganı.
- Bulgular metni: ResNet-18 ile bilgi düzeyi ve kesit normalizasyonu karşılaştırması.

Tablo IV (savunma), VI (maliyet) ve VII (sızıntı denetimi) sınıflandırma başarısı ölçmez. Bunlar eşik, maliyet ve
şans düzeyi sorularıdır.

Beyin MR'da ölçüler, makaledeki gibi 5 katın kat dışı tahminleri birleştirilerek hesaplanır. Ortalama ± standart sapma
5 tohum üzerindendir; makaledeki gibi örneklem standart sapması kullanılır.

Çıktılar:
- `results/tables/`: `metrikler_ozet.csv|md`, `metrikler_sinif.csv|md`, `karisiklik_matrisleri.csv`,
  `makale_eslesme.csv|md`
- `results/figures/`: `karisiklik_{beyin,covidqu}.png|pdf`

Çalıştırma: .venv\\Scripts\\python -m analysis.metrikler
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,  # noqa: E402
                             precision_recall_fscore_support)

import config  # noqa: E402
from common.evaluation import auc_score, bootstrap_ci  # noqa: E402
from common.report import write_markdown_table  # noqa: E402
from common.tez_bicim import SINIF_TEZ, kaydet  # noqa: E402
from foveahe.data import load_dataset  # noqa: E402

PREDS = config.RESULTS / "preds"
BEKLENEN = config.RESULTS / "beklenen" / "makale_degerleri.json"
T5 = [0, 1, 2, 3, 4]
T0 = [0]
VERI_AD = {"brain": "Beyin MR", "covidqu": "COVID-QU-Ex", "kaggle": "Kaggle CXR"}
GORUS = {"tam": "Tam (şifresiz)", "yalniz_roi": "Yalnız ROI", "baglam": "Bağlam (Π_ROI görüşü)",
         "baglam_genis10": "Bağlam + 10 piksel", "baglam_genis20": "Bağlam + 20 piksel",
         "baglam_genis40": "Bağlam + 40 piksel", "kutu": "Sınırlayıcı kutu"}
GRUP = {"girdi_boyutu": "Girdi boyutu", "konum": "Konum", "buyukluk": "Büyüklük", "sekil": "Şekil", "tumu": "Tümü"}
TEMSIL = {"U512": "Tam görüntü (512×512)", "U256": "Tam görüntü (256×256)", "F32_G16": "FoveaHE F32_G16",
          "F64_G32": "FoveaHE F64_G32", "U64": "Eş örnekli U64", "U64_pencere": "U64 + geometri"}
SALDIRGAN_B = "Bağlam saldırganı (ResNet-18)"
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "surface": "#fcfcfb"}
SIRALI = LinearSegmentedColormap.from_list(  # tek tonlu mavi (referans paletin sıralı basamakları)
    "sirali_mavi", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


@dataclass
class Kayit:
    anahtar: str
    tablo: str
    veri: str
    hedef: str
    model: str
    girdi: str
    desen: str  # {s}: tohum
    tohumlar: list = field(default_factory=lambda: T5)
    siniflar: list | None = None  # .npz dosyaları sınıf adlarını kendisi taşır


def kayitlar() -> list[Kayit]:
    k = []
    # Tablo II: Saldırı A
    for g in ("girdi_boyutu", "konum", "buyukluk", "sekil", "tumu"):
        k.append(Kayit(f"A_beyin_{g}", "Tablo II", "brain", "Tümör tipi (3 sınıf)", "Meta veri saldırganı (HistGB)",
                       GRUP[g], f"saldiri_A_beyin_tumor_tipi_{g}_s{{s}}.npz"))
    for g in ("konum", "buyukluk", "sekil", "tumu"):
        k.append(Kayit(f"A_covidqu_{g}", "Tablo II", "covidqu", "Teşhis (3 sınıf)", "Meta veri saldırganı (HistGB)",
                       GRUP[g], f"saldiri_A_covidqu_teshis_{g}_s{{s}}.npz"))
    k.append(Kayit("A_covidqu_np_tumu", "Tablo II", "covidqu", "Normal / COVID dışı pnömoni",
                   "Meta veri saldırganı (HistGB)", "Tümü", "saldiri_A_covidqu_normal_pnomoni_tumu_s{s}.npz"))
    k.append(Kayit("A_kaggle_np_boyut", "Tablo II", "kaggle", "Normal / pnömoni", "Meta veri saldırganı (HistGB)",
                   "Girdi boyutu", "saldiri_A_kaggle_normal_pnomoni_girdi_boyutu_s{s}.npz"))
    # Tablo III: Saldırı B (beyin MR görünür piksel normalizasyonu)
    for v in GORUS:
        tohum = T5 if v in ("tam", "baglam", "baglam_genis40") else T0
        k.append(Kayit(f"B_beyin_{v}", "Tablo III", "brain", "Tümör tipi (3 sınıf)", SALDIRGAN_B, GORUS[v],
                       f"brain_{v}_s{{s}}_gnorm.npy", tohum))
        k.append(Kayit(f"B_covidqu_{v}", "Tablo III", "covidqu", "Teşhis (3 sınıf)", SALDIRGAN_B, GORUS[v],
                       f"covidqu_{v}_s{{s}}.npy", tohum))
    # Metin: kesit tabanlı normalizasyon (ek denetim)
    for v, tohum in (("tam", T5), ("baglam", T5), ("yalniz_roi", T0)):
        k.append(Kayit(f"B_beyin_{v}_kesit", "Metin (kesit normalizasyonu)", "brain", "Tümör tipi (3 sınıf)",
                       SALDIRGAN_B, GORUS[v] + ", kesit normalizasyonu", f"brain_{v}_s{{s}}.npy", tohum))
    # Tablo V: şifreli çalışabilen modeller
    for veri, tam in (("brain", "U512"), ("covidqu", "U256")):
        for model in ("D", "D2", "C"):
            for cfg in (tam, "F32_G16", "F64_G32", "U64", "U64_pencere"):
                k.append(Kayit(f"M_{veri}_{model}_{cfg}", "Tablo V", veri,
                               "Tümör tipi (3 sınıf)" if veri == "brain" else "Teşhis (3 sınıf)",
                               f"Model {model}", TEMSIL[cfg], f"fovea_model_{veri}_{model}_{cfg}_s{{s}}.npy"))
    # Tablo VIII: şifreli özet ve bozuk bağlam saldırganı
    for veri, model in (("brain", "D"), ("covidqu", "D2")):
        k.append(Kayit(f"Ozet_{veri}_{model}", "Tablo VIII", veri,
                       "Tümör tipi (3 sınıf)" if veri == "brain" else "Teşhis (3 sınıf)",
                       f"Şifreli özet + Model {model}", "ResNet-18 özeti (512 değer)",
                       f"ozet_{veri}_{model}_s{{s}}.npy"))
    k.append(Kayit("Bozuk_beyin", "Tablo VIII", "brain", "Tümör tipi (3 sınıf)", SALDIRGAN_B,
                   "Bağlam + Gauss gürültüsü (σ = 0.4)", "gurultu_brain_gauss0.4_baglam_s{s}_gnorm.npy", T0))
    k.append(Kayit("Bozuk_covidqu", "Tablo VIII", "covidqu", "Teşhis (3 sınıf)", SALDIRGAN_B,
                   "Bağlam + Gauss gürültüsü (σ = 0.4)", "gurultu_covidqu_gauss0.4_baglam_s{s}.npy", T0))
    # Metin: ResNet-18 ile bilgi düzeyi (FoveaHE Adım 1)
    for veri, cfgs in (("brain", ("tam", "F32_G16", "F64_G32", "tam_pencere")), ("covidqu", ("tam", "F32_G16", "F64_G32"))):
        for cfg in cfgs:
            ad = {"tam": "Tam (224×224)", "tam_pencere": "Tam + pencere kanalı"}.get(cfg, f"FoveaHE {cfg}")
            k.append(Kayit(f"R_{veri}_{cfg}", "Metin (bilgi düzeyi)", veri,
                           "Tümör tipi (3 sınıf)" if veri == "brain" else "Teşhis (3 sınıf)",
                           "ResNet-18 (bilgi üst sınırı)", ad, f"fovea_{veri}_{cfg}_s{{s}}.npy"))
    return k


def hizala(probs: np.ndarray, ds):
    """Kayıtlı olasılık dizisini (tam uzunluk ya da yalnız test) etiketlerle eşleştirir."""
    if probs.shape[0] == len(ds.y):
        idx = np.flatnonzero(~np.isnan(probs[:, 0]))
        return idx, probs[idx]
    if ds.test_idx is not None and probs.shape[0] == len(ds.test_idx):
        return ds.test_idx, probs
    raise ValueError(f"tahmin dizisi boyutu tanınmadı: {probs.shape}")


def oku(k: Kayit, s: int, ds_map: dict):
    path = PREDS / k.desen.format(s=s)
    if not path.exists():
        raise FileNotFoundError(f"kayıtlı tahmin yok: {path.name} ({k.tablo}, {k.model}, {k.girdi})")
    if path.suffix == ".npz":
        with np.load(path) as z:
            return z["proba"], z["y"], z["idx"], [str(c) for c in z["siniflar"]]
    ds = ds_map[k.veri]
    idx, p = hizala(np.load(path), ds)
    return p, ds.y[idx], idx, list(ds.labels)


def olcumler(y: np.ndarray, p: np.ndarray) -> dict:
    n_cls = p.shape[1]
    yp = p.argmax(1)
    pr, rc, f1, destek = precision_recall_fscore_support(y, yp, labels=range(n_cls), zero_division=0)
    return {"auc": auc_score(y, p), "dogruluk": accuracy_score(y, yp), "dengeli_dogruluk": balanced_accuracy_score(y, yp),
            "kesinlik_makro": pr.mean(), "duyarlilik_makro": rc.mean(), "f1_makro": f1.mean(),
            "f1_agirlikli": float(np.average(f1, weights=destek)), "n": len(y),
            "sinif": {"kesinlik": pr, "duyarlilik": rc, "f1": f1, "destek": destek},
            "karisiklik": confusion_matrix(y, yp, labels=range(n_cls))}


def sinif_adi(ad: str) -> str:
    return SINIF_TEZ.get(ad, {"NORMAL": "Normal", "PNEUMONIA": "Pnömoni"}.get(ad, ad))


def hesapla(kayit_listesi, ds_map):
    sonuc = {}
    for k in kayit_listesi:
        tohum_sonuclari, siniflar = [], None
        for s in k.tohumlar:
            p, y, idx, siniflar = oku(k, s, ds_map)
            tohum_sonuclari.append(olcumler(y, p) | {"tohum": s, "y": y, "p": p})
        sonuc[k.anahtar] = {"kayit": k, "tohumlar": tohum_sonuclari, "siniflar": [sinif_adi(c) for c in siniflar]}
    return sonuc


def ort_std(values):
    a = np.asarray(values, dtype=float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else np.nan)


def bicim(m, s, d):
    return f"{m:.{d}f}" if np.isnan(s) else f"{m:.{d}f} ± {s:.{d}f}"


def ozet_tablolari(sonuc):
    ozet, sinif, karisik = [], [], []
    for a, r in sonuc.items():
        k, ts = r["kayit"], r["tohumlar"]
        satir = {"anahtar": a, "tablo": k.tablo, "veri": VERI_AD[k.veri], "hedef": k.hedef, "model": k.model,
                 "girdi": k.girdi, "tohum_sayisi": len(ts), "n_test": ts[0]["n"]}
        for m in ("auc", "dogruluk", "dengeli_dogruluk", "kesinlik_makro", "duyarlilik_makro", "f1_makro", "f1_agirlikli"):
            satir[m], satir[m + "_std"] = ort_std([t[m] for t in ts])
        ozet.append(satir)
        for ci, ad in enumerate(r["siniflar"]):
            sinif.append({"anahtar": a, "tablo": k.tablo, "veri": VERI_AD[k.veri], "model": k.model, "girdi": k.girdi,
                          "sinif": ad, "kesinlik": np.mean([t["sinif"]["kesinlik"][ci] for t in ts]),
                          "duyarlilik": np.mean([t["sinif"]["duyarlilik"][ci] for t in ts]),
                          "f1": np.mean([t["sinif"]["f1"][ci] for t in ts]),
                          "destek": int(ts[0]["sinif"]["destek"][ci])})
        for t in ts:
            cm = t["karisiklik"]
            for i, gi in enumerate(r["siniflar"]):
                for j, tj in enumerate(r["siniflar"]):
                    karisik.append({"anahtar": a, "tohum": t["tohum"], "gercek": gi, "tahmin": tj, "sayi": int(cm[i, j])})
    return pd.DataFrame(ozet), pd.DataFrame(sinif), pd.DataFrame(karisik)


def tutar(hesap: float, makale: float, ondalik: int) -> bool:
    return abs(hesap - makale) <= 0.5 * 10 ** -ondalik + 1e-9


def makale_eslesme(sonuc, ds_map) -> pd.DataFrame:
    bek = json.loads(BEKLENEN.read_text(encoding="utf-8"))
    kontrol = []

    def ekle(kaynak, makale, hesap, ondalik):
        kaynak = (kaynak.replace("\\PiROI{}", "Π_ROI").replace("\\_", "_").replace("($\\sigma = 0.4$)", "(σ = 0.4)")
                  .replace("$", ""))  # makalenin LaTeX yazımı düz metne
        kontrol.append({"kaynak": kaynak, "makale": makale, "hesaplanan": round(hesap, 6), "ondalik": ondalik,
                        "tuttu": tutar(hesap, makale, ondalik)})

    def ort(anahtar, m="auc"):
        return ort_std([t[m] for t in sonuc[anahtar]["tohumlar"]])

    # Tablo II
    a_map = {("Beyin MR, tümör tipi", "Girdi boyutu"): "A_beyin_girdi_boyutu", ("Beyin MR, tümör tipi", "Konum"): "A_beyin_konum",
             ("Beyin MR, tümör tipi", "Büyüklük"): "A_beyin_buyukluk", ("Beyin MR, tümör tipi", "Şekil"): "A_beyin_sekil",
             ("Beyin MR, tümör tipi", "Tümü"): "A_beyin_tumu", ("COVID-QU-Ex, teşhis", "Konum"): "A_covidqu_konum",
             ("COVID-QU-Ex, teşhis", "Büyüklük"): "A_covidqu_buyukluk", ("COVID-QU-Ex, teşhis", "Şekil"): "A_covidqu_sekil",
             ("COVID-QU-Ex, teşhis", "Tümü"): "A_covidqu_tumu", ("COVID-QU-Ex, normal/pnömoni", "Tümü"): "A_covidqu_np_tumu",
             ("Kaggle, normal/pnömoni", "Girdi boyutu"): "A_kaggle_np_boyut"}
    gruplar = ds_map["brain"].groups
    for r in bek["tablo_II_saldiri_A"]["satirlar"]:
        a = a_map[(r["veri_hedef"], r["oznitelikler"])]
        etiket = f"Tablo II, {r['veri_hedef']}, {r['oznitelikler']}"
        ekle(etiket + ", AUC", r["auc"], ort(a)[0], 3)
        if r["ga95"]:
            son = sonuc[a]["tohumlar"][-1]
            lo, hi = bootstrap_ci(son["y"], son["p"], groups=gruplar if a.startswith("A_beyin") else None, n_boot=500)
            ekle(etiket + ", %95 GA alt", r["ga95"][0], lo, 3)
            ekle(etiket + ", %95 GA üst", r["ga95"][1], hi, 3)
    # Tablo III
    gorus_ters = {v: k for k, v in GORUS.items()}
    gorus_ters["Bağlam (\\PiROI{})"] = "baglam"
    for r in bek["tablo_III_saldiri_B"]["satirlar"]:
        v = gorus_ters[r["gorus"]]
        for veri, kisa in (("brain", "beyin"), ("covidqu", "covidqu")):
            m, s = ort(f"B_{'beyin' if veri == 'brain' else 'covidqu'}_{v}")
            ekle(f"Tablo III, {r['gorus']}, {VERI_AD[veri]}, AUC", r[kisa]["auc"], m, 4)
            if r[kisa]["std"] is not None:
                ekle(f"Tablo III, {r['gorus']}, {VERI_AD[veri]}, std", r[kisa]["std"], s, 4)
    # Tablo V
    for r in bek["tablo_V_modeller"]["satirlar"]:
        veri = "brain" if r["veri"] == "Beyin" else "covidqu"
        tam = "U512" if veri == "brain" else "U256"
        for kol, cfg in (("Tam", tam), ("F32_G16", "F32_G16"), ("F64_G32", "F64_G32"), ("U64", "U64"), ("U64+geo", "U64_pencere")):
            ekle(f"Tablo V, {r['veri']}, Model {r['model']}, {kol}", r[kol], ort(f"M_{veri}_{r['model']}_{cfg}")[0], 3)
    # Tablo VIII
    for r in bek["tablo_VIII_karsilastirma"]["satirlar"]:
        y = r["yontem"]
        for veri, kisa, model in (("brain", "beyin", "D"), ("covidqu", "covidqu", "D2")):
            tam = "U512" if veri == "brain" else "U256"
            if y.startswith("FoveaHE F32"):
                a = f"M_{veri}_{model}_F32_G16"
            elif y.startswith("FoveaHE F64"):
                a = f"M_{veri}_{model}_F64_G32"
            elif y.startswith("Şifreli özet"):
                a = f"Ozet_{veri}_{model}"
            else:  # tam şifreleme ve Π_ROI satırları: seçici şifreleme çıktıyı değiştirmez, teşhis modeli tam görüntüdür
                a = f"M_{veri}_{model}_{tam}"
            ekle(f"Tablo VIII, {y}, {VERI_AD[veri]}, teşhis AUC", r[kisa]["auc"], ort(a)[0], 3)
            if "varsayılan" in y:
                ekle(f"Tablo VIII, {y}, {VERI_AD[veri]}, saldırgan AUC", r[kisa]["saldirgan"],
                     ort(f"B_{kisa}_baglam")[0], 2)
            if y.startswith("Bozuk bağlam"):
                ekle(f"Tablo VIII, {y}, {VERI_AD[veri]}, saldırgan AUC", r[kisa]["saldirgan"],
                     ort(f"Bozuk_{kisa}")[0], 2)
    # Metin değerleri
    mt = bek["metin_bozuk_baglam"]
    ekle("Metin, bozuk bağlam σ = 0.4, Beyin MR, saldırgan AUC", mt["beyin"], ort("Bozuk_beyin")[0], 3)
    ekle("Metin, bozuk bağlam σ = 0.4, COVID-QU-Ex, saldırgan AUC", mt["covidqu"], ort("Bozuk_covidqu")[0], 3)
    m1 = bek["metin_adim1_resnet"]
    for cfg in ("tam", "F32_G16", "F64_G32", "tam_pencere"):
        ekle(f"Metin, ResNet-18 bilgi düzeyi, Beyin MR, {cfg}", m1[f"beyin_{cfg}"], ort(f"R_brain_{cfg}")[0], 4)
    kayip = max(ort("R_covidqu_tam")[0] - ort(f"R_covidqu_{c}")[0] for c in ("F32_G16", "F64_G32"))
    ekle("Metin, ResNet-18 bilgi düzeyi, COVID-QU-Ex, en büyük kayıp", m1["covidqu_en_buyuk_kayip"], kayip, 4)
    mk = bek["metin_saldiri_B_kesit"]
    fark = max(abs(ort(f"B_beyin_{v}")[0] - ort(f"B_beyin_{v}_kesit")[0]) for v in ("tam", "baglam"))
    ekle("Metin, kesit normalizasyonu, tam/bağlam en büyük fark", mk["tam_baglam_en_buyuk_fark"], fark, 3)
    ekle("Metin, kesit normalizasyonu, yalnız ROI", mk["yalniz_roi_kesit"], ort("B_beyin_yalniz_roi_kesit")[0], 3)
    ms = bek["metin_saldiri_A_beyin_sinif"]
    t0 = sonuc["A_beyin_tumu"]["tohumlar"][0]
    for ad, ci in (("meningiom", 0), ("gliom", 1), ("hipofiz", 2)):
        ekle(f"Metin, Saldırı A sınıf bazında AUC, {ad}", ms[ad], auc_score((t0["y"] == ci).astype(int), t0["p"][:, ci]), 3)
    return pd.DataFrame(kontrol)


def karisiklik_sekli(sonuc, veri: str, paneller: list[tuple[str, str]], dosya: str) -> None:
    n = len(paneller)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4.1 * rows), facecolor=INK["surface"])
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, (anahtar, baslik) in zip(axes.ravel(), paneller):
        r = sonuc[anahtar]
        t = r["tohumlar"][0]
        cm = t["karisiklik"].astype(float)
        oran = cm / cm.sum(1, keepdims=True)
        ax.imshow(oran, cmap=SIRALI, vmin=0, vmax=1)
        ad = [a.rsplit(" ", 1)[0] + "\n" + a.rsplit(" ", 1)[1] if len(a) > 12 and " " in a else a
              for a in r["siniflar"]]  # uzun sınıf adı iki satıra (başlıklar şekil dışına taşmasın)
        for i in range(len(ad)):
            for j in range(len(ad)):
                renk = "#ffffff" if oran[i, j] > 0.55 else INK["primary"]
                ax.text(j, i, f"{int(cm[i, j])}\n(%{100 * oran[i, j]:.0f})", ha="center", va="center", fontsize=8.5,
                        color=renk)
        ax.set_xticks(range(len(ad)), ad, fontsize=8, color=INK["secondary"])
        ax.set_yticks(range(len(ad)), ad, fontsize=8, color=INK["secondary"])
        ax.set_xlabel("Tahmin edilen", fontsize=8.5, color=INK["secondary"])
        ax.set_ylabel("Gerçek", fontsize=8.5, color=INK["secondary"])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(f"{baslik}\nAUC {t['auc']:.3f} · doğruluk {t['dogruluk']:.3f} · makro F1 {t['f1_makro']:.3f}",
                     fontsize=9, color=INK["primary"])
    fig.suptitle(f"{VERI_AD[veri]}: karışıklık matrisleri (tohum 0; renk satır yüzdesi)", fontsize=11,
                 color=INK["primary"])
    fig.tight_layout()
    kaydet(fig, config.FIGURES / dosya, facecolor=INK["surface"])
    plt.close(fig)


def md_ozet(ozet: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({"Tablo": ozet.tablo, "Veri": ozet.veri, "Hedef": ozet.hedef, "Model": ozet.model,
                       "Girdi": ozet.girdi, "Tohum": ozet.tohum_sayisi, "Test n": ozet.n_test})
    for m, ad, d in (("auc", "AUC", 4), ("dogruluk", "Doğruluk", 3), ("dengeli_dogruluk", "Dengeli doğruluk", 3),
                     ("kesinlik_makro", "Kesinlik (makro)", 3), ("duyarlilik_makro", "Duyarlılık (makro)", 3),
                     ("f1_makro", "F1 (makro)", 3), ("f1_agirlikli", "F1 (ağırlıklı)", 3)):
        df[ad] = [bicim(a, b, d) for a, b in zip(ozet[m], ozet[m + "_std"])]
    return df


def main():
    ds_map = {"brain": load_dataset("brain"), "covidqu": load_dataset("covidqu")}
    sonuc = hesapla(kayitlar(), ds_map)
    kontrol = makale_eslesme(sonuc, ds_map)
    kontrol.to_csv(config.TABLES / "makale_eslesme.csv", index=False)
    md = pd.DataFrame({"Makaledeki yer": kontrol.kaynak,
                       "Makalede basılı": [f"{m:.{d}f}" for m, d in zip(kontrol.makale, kontrol.ondalik)],
                       "Yeniden hesaplanan": [f"{h:.{d + 2}f}" for h, d in zip(kontrol.hesaplanan, kontrol.ondalik)],
                       "Aynı basamağa yuvarlanınca": [f"{h:.{d}f}" for h, d in zip(kontrol.hesaplanan, kontrol.ondalik)],
                       "Tuttu": kontrol.tuttu.map({True: "evet", False: "HAYIR"})})
    write_markdown_table(md, config.TABLES / "makale_eslesme.md")
    tutmayan = kontrol[~kontrol.tuttu]
    print(f"Makale eşleşmesi: {int(kontrol.tuttu.sum())}/{len(kontrol)} değer tuttu.")
    if len(tutmayan):
        print(tutmayan.to_string(index=False))
        raise SystemExit("Makaleyle tutmayan değer var; tablolar yazılmadı.")
    ozet, sinif, karisik = ozet_tablolari(sonuc)
    ozet.to_csv(config.TABLES / "metrikler_ozet.csv", index=False)
    write_markdown_table(md_ozet(ozet), config.TABLES / "metrikler_ozet.md")
    sinif.to_csv(config.TABLES / "metrikler_sinif.csv", index=False)
    write_markdown_table(sinif.drop(columns="anahtar").rename(columns={
        "tablo": "Tablo", "veri": "Veri", "model": "Model", "girdi": "Girdi", "sinif": "Sınıf", "kesinlik": "Kesinlik",
        "duyarlilik": "Duyarlılık", "f1": "F1", "destek": "Destek (n)"}), config.TABLES / "metrikler_sinif.md",
        floatfmt="{:.3f}")
    karisik.to_csv(config.TABLES / "karisiklik_matrisleri.csv", index=False)
    for veri, model in (("brain", "D"), ("covidqu", "D2")):
        kisa = "beyin" if veri == "brain" else "covidqu"
        tam = "U512" if veri == "brain" else "U256"
        karisiklik_sekli(sonuc, veri, [
            (f"A_{kisa}_tumu", "Meta veri saldırganı (Tablo II, tümü)"),
            (f"B_{kisa}_tam", "Bağlam saldırganı, tam görüş (Tablo III)"),
            (f"B_{kisa}_baglam", "Bağlam saldırganı, Π_ROI görüşü (Tablo III)"),
            (f"M_{veri}_{model}_{tam}", f"Model {model}, tam görüntü = Π_ROI (Tablo V)"),
            (f"M_{veri}_{model}_F32_G16", f"Model {model}, FoveaHE F32_G16 (Tablo V)"),
            (f"Ozet_{veri}_{model}", f"Şifreli özet + Model {model} (Tablo VIII)")], f"karisiklik_{kisa}")
    print(md_ozet(ozet)[["Tablo", "Veri", "Model", "Girdi", "AUC", "Doğruluk", "F1 (makro)"]].to_string(index=False))


if __name__ == "__main__":
    main()
