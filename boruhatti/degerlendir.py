"""Kayıtlı modellerle test: "model ile sonuç" (danışman maddeleri 1, 3, 7).

Dört iş yapar (birlikte ya da ayrı):
1. `--saldirgan modeller/<ad> ...`: `boruhatti.egit` ile kaydedilen saldırgan modellerini (beyin MR'da 5 katın her biri
   için bir model) kendi test bölmelerinde çalıştırır. Ölçüler, karışıklık matrisi ve görüntü başına tahminler yazılır.
   Makaledeki aynı tohumlu koşu ve 5 tohum ortalaması ile karşılaştırılır.
2. `--makale-modelleri`: makaledeki şifreli modellerin (Tablo V ve Tablo VIII'deki şifreli özet) kayıtlı CKKS'e hazır
   ağırlıklarıyla test bölmelerini şifresiz çalıştırır. 5 tohumun AUC ortalaması makaledeki değerle karşılaştırılır.
   Ağırlıklar `modeller/makale/` (Hugging Face'ten indirilen) ya da `results/checkpoints/` altında aranır.
3. `--sifreli N`: aynı ağırlıklarla N test görüntüsünde gerçek CKKS şifreli çıkarım yapar (TenSEAL, N = 16384).
   Şifreli ve şifresiz logitler karşılaştırılır (makale: en büyük fark 1.6e-4, tahmin uyumu %100).
4. `--foveahe modeller/<ad> ...`: `boruhatti.egit` ile yeniden eğitilen şifreli modeli, makaledeki aynı tohum ve
   bölmenin kayıtlı modeliyle aynı test görüntülerinde karşılaştırır (AUC farkı, tahmin uyumu, ağırlık farkı).

Çıktılar: `results/boruhatti/degerlendirme_*.csv|md|json`.

Örnek: python -m boruhatti.degerlendir --makale-modelleri --sifreli 10 --saldirgan modeller/saldirgan_beyin_baglam_tez_s0
"""
from __future__ import annotations

import os

for _degisken in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_degisken, "4")  # işlemciyi kilitlememek için (numpy içe aktarılmadan önce)

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd
import torch

import config
from analysis.metrikler import olcumler
from boruhatti.ortak import CIKTI, MODELLER, aygit, is_parcacigi, json_oku, json_yaz
from common.report import write_markdown_table
from common.tez_bicim import SINIF_TEZ

TABLO_V = [("brain", "U512"), ("covidqu", "U256")]
TEMSILLER = ["F32_G16", "F64_G32", "U64", "U64_pencere"]
TOHUMLAR = [0, 1, 2, 3, 4]


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def agirlik_yolu(ad: str) -> Path:
    for kok in (MODELLER / "makale", config.CHECKPOINTS):
        if (kok / ad).exists():
            return kok / ad
    raise FileNotFoundError(f"{ad} bulunamadı: modeller/makale/ altına indirin (README: modelleri indirme)")


def ileri(w: dict, x: np.ndarray) -> np.ndarray:
    from foveahe.he_cnn import forward_numpy_c
    from foveahe.he_models import forward_numpy
    return forward_numpy_c(w, x) if str(w["kind"]) == "C" else forward_numpy(w, x)


def bolum_adlari(ds) -> list[tuple[str, np.ndarray]]:
    """Şifreli modellerin test bölmeleri (experiments.fovea_models.splits ile aynı adlar)."""
    if ds.name == "brain":
        return [(f"k{k}", np.flatnonzero(ds.folds == k)) for k in np.unique(ds.folds)]
    return [("test", np.flatnonzero(ds.split == "Test"))]


# ---------------------------------------------------------------- makaledeki şifreli modeller
def makale_modelleri() -> pd.DataFrame:
    from experiments.fovea_models import VectorData
    from foveahe.data import STORE, load_dataset
    from foveahe.he_models import load_weights
    from foveahe.representation import FoveaSpec
    ref = pd.read_csv(config.TABLES / "cozum_modeller.csv")
    satirlar = []
    for veri, tam in TABLO_V:
        ds = load_dataset(veri)
        bolumler = bolum_adlari(ds)
        for cfg in [tam] + TEMSILLER:
            spec = FoveaSpec.parse(cfg)
            data = VectorData(ds, spec, "cpu")
            t0 = time.perf_counter()
            logit = {(m, s): np.full((len(ds.y), len(ds.labels)), np.nan) for m in ("D", "D2", "C") for s in TOHUMLAR}
            for bolum, idx in bolumler:
                agirliklar = {(m, s): load_weights(agirlik_yolu(f"fovea_{veri}_{m}_{cfg}_s{s}_{bolum}.npz"))
                              for m in ("D", "D2", "C") for s in TOHUMLAR}
                for i in range(0, len(idx), 128):
                    parca = idx[i:i + 128]
                    x = data.get(parca).double().numpy()
                    for anahtar, w in agirliklar.items():
                        logit[anahtar][parca] = ileri(w, x)
            for m in ("D", "D2", "C"):
                aucs = []
                for s in TOHUMLAR:
                    p = softmax(logit[(m, s)][np.flatnonzero(~np.isnan(logit[(m, s)][:, 0]))])
                    yt = ds.y[~np.isnan(logit[(m, s)][:, 0])]
                    o = olcumler(yt, p)
                    aucs.append(o["auc"])
                    r = ref[(ref.veri == ds.display) & (ref.model == m) & (ref.temsil == cfg) & (ref.tohum == s)]
                    satirlar.append({"veri": ds.display, "model": f"Model {m}", "temsil": cfg, "tohum": s, "auc": o["auc"],
                                     "dogruluk": o["dogruluk"], "f1_makro": o["f1_makro"],
                                     "makaledeki_kosu_auc": float(r.auc.iloc[0]) if len(r) else np.nan})
                print(f"  [{veri}] Model {m:2s} {cfg:12s} 5 tohum AUC ort. {np.mean(aucs):.4f} ({time.perf_counter() - t0:.0f} s)",
                      flush=True)
        # Tablo VIII: şifreli özet (ResNet-18 özeti + D / D2)
        emb = np.load(STORE / veri / "emb_r18.npy").astype(np.float64)
        m = "D" if veri == "brain" else "D2"
        ref_o = pd.read_csv(config.TABLES / "cozum_rakipler.csv")
        for s in TOHUMLAR:
            p = np.full((len(ds.y), len(ds.labels)), np.nan)
            for bolum, idx in bolumler:
                p[idx] = softmax(ileri(load_weights(agirlik_yolu(f"ozet_{veri}_{m}_s{s}_{bolum}.npz")), emb[idx]))
            ok = ~np.isnan(p[:, 0])
            o = olcumler(ds.y[ok], p[ok])
            r = ref_o[(ref_o.veri == ds.display) & (ref_o.model == m) & (ref_o.tohum == s)]
            satirlar.append({"veri": ds.display, "model": f"Şifreli özet + Model {m}", "temsil": "ResNet-18 özeti (512)",
                             "tohum": s, "auc": o["auc"], "dogruluk": o["dogruluk"], "f1_makro": o["f1_makro"],
                             "makaledeki_kosu_auc": float(r.auc.iloc[0]) if len(r) else np.nan})
    df = pd.DataFrame(satirlar)
    df["fark"] = df.auc - df.makaledeki_kosu_auc
    return df


def makale_karsilastir(df: pd.DataFrame) -> pd.DataFrame:
    """5 tohum ortalamasını makaledeki basılı değerle (Tablo V, VIII; 3 basamak) karşılaştırır."""
    bek = json_oku(config.RESULTS / "beklenen" / "makale_degerleri.json")
    kol = {"U512": "Tam", "U256": "Tam", "F32_G16": "F32_G16", "F64_G32": "F64_G32", "U64": "U64", "U64_pencere": "U64+geo"}
    ozet = []
    for (veri, model, temsil), g in df.groupby(["veri", "model", "temsil"], sort=False):
        kisa = "Beyin" if veri.startswith("beyin") else "COVID"
        makale = np.nan
        if model.startswith("Model"):
            r = [x for x in bek["tablo_V_modeller"]["satirlar"] if x["veri"] == kisa and f"Model {x['model']}" == model]
            makale = r[0][kol[temsil]] if r else np.nan
        else:
            r = [x for x in bek["tablo_VIII_karsilastirma"]["satirlar"] if x["yontem"].startswith("Şifreli özet")]
            makale = r[0]["beyin" if kisa == "Beyin" else "covidqu"]["auc"]
        ort = float(g.auc.mean())
        ozet.append({"veri": veri, "model": model, "temsil": temsil, "auc_5_tohum": ort, "auc_std": float(g.auc.std(ddof=1)),
                     "f1_makro_5_tohum": float(g.f1_makro.mean()), "makalede": makale,
                     "en_buyuk_tohum_farki": float(g.fark.abs().max()),
                     "tuttu": bool(abs(ort - makale) <= 0.0005 + 1e-9)})
    return pd.DataFrame(ozet)


# ---------------------------------------------------------------- gerçek CKKS şifreli çıkarım
def sifreli_sinama(n: int) -> list[dict]:
    from experiments.fovea_models import VectorData
    from foveahe.data import load_dataset
    from foveahe.he_cnn import CNNInference
    from foveahe.he_infer import FoveaHEInference
    from foveahe.he_models import load_weights
    from foveahe.representation import FoveaSpec
    from he.piroi import make_context
    ctx = make_context()
    out = []
    for veri, m, cfg in (("brain", "D", "F32_G16"), ("brain", "C", "F32_G16"), ("covidqu", "D2", "F32_G16")):
        ds = load_dataset(veri)
        bolum, idx = bolum_adlari(ds)[0]
        w = load_weights(agirlik_yolu(f"fovea_{veri}_{m}_{cfg}_s0_{bolum}.npz"))
        rng = np.random.default_rng(0)
        secim = np.concatenate([rng.choice(idx[ds.y[idx] == c], max(1, n // len(ds.labels)), replace=False)
                                for c in range(len(ds.labels))])
        x = VectorData(ds, FoveaSpec.parse(cfg), "cpu").get(secim).double().numpy()
        cikarim = CNNInference(ctx, w) if m == "C" else FoveaHEInference(ctx, w)
        t0 = time.perf_counter()
        sifreli = np.array([cikarim.run(xi, measure_bytes=False)[0] for xi in x])
        sure = (time.perf_counter() - t0) / len(x)
        acik = ileri(w, x)
        out.append({"veri": ds.display, "model": f"Model {m}", "temsil": cfg, "goruntu": len(x),
                    "en_buyuk_logit_farki": float(np.abs(sifreli - acik).max()),
                    "tahmin_uyumu": float((sifreli.argmax(1) == acik.argmax(1)).mean()),
                    "goruntu_basina_s": sure})
        print(f"  [şifreli] {ds.display} Model {m} {cfg}: {len(x)} görüntü, en büyük logit farkı "
              f"{out[-1]['en_buyuk_logit_farki']:.1e}, tahmin uyumu %{100 * out[-1]['tahmin_uyumu']:.0f}, "
              f"görüntü başına {sure:.2f} s", flush=True)
    return out


# ---------------------------------------------------------------- kaydedilen saldırgan modelleri
def saldirgan(klasor: Path, dev: str) -> dict:
    from attacks.context_cnn import make_model
    from boruhatti.egit import makaledeki_saldirgan_auc, saldirgan_bolmeleri, saldirgan_olasilik, saldirgan_verisi
    ayar = json_oku(klasor / "ayar.json")
    cache, y, bilgi, etiketler = saldirgan_verisi(ayar["veri"], ayar["normalizasyon"], dev)
    siniflar = [SINIF_TEZ[e] for e in etiketler]
    olasilik = np.full((len(y), len(siniflar)), np.nan)
    for bolum, tr, va, te in saldirgan_bolmeleri(ayar["veri"], ayar["protokol"], bilgi):
        model = make_model(len(siniflar))
        model.load_state_dict(torch.load(klasor / f"{bolum}.pt", map_location=dev, weights_only=True))
        model.to(dev)
        hedef = te if te is not None else va
        olasilik[hedef] = saldirgan_olasilik(model, cache, hedef, ayar["gorus"], len(siniflar), np.random.default_rng(0))
    ok = np.flatnonzero(~np.isnan(olasilik[:, 0]))
    m = olcumler(y[ok], olasilik[ok])
    sonuc = {"model": klasor.name, "veri": ayar["veri"], "gorus": ayar["gorus"], "protokol": ayar["protokol"], "n": len(ok),
             **{k: float(m[k]) for k in ("auc", "dogruluk", "dengeli_dogruluk", "kesinlik_makro", "duyarlilik_makro",
                                         "f1_makro", "f1_agirlikli")}, "karisiklik": m["karisiklik"].tolist()}
    makale, tablo = makaledeki_saldirgan_auc(ayar["veri"], ayar["gorus"], ayar["normalizasyon"], ayar["tohum"])
    if makale is not None:
        df = pd.read_csv(config.TABLES / tablo)
        df = df[(df.gorus == ayar["gorus"]) & df.veri.str.startswith("beyin" if ayar["veri"] == "beyin" else "akciğer")]
        sonuc.update({"makaledeki_ayni_tohum_auc": makale, "fark": sonuc["auc"] - makale,
                      "makaledeki_5_tohum_auc": float(df.auc.mean()),
                      "makaledeki_5_tohum_std": float(df.auc.std(ddof=1)) if len(df) > 1 else np.nan})
    pd.DataFrame({"satir": ok, "gercek": [siniflar[i] for i in y[ok]],
                  "tahmin": [siniflar[i] for i in olasilik[ok].argmax(1)]}).to_csv(
        CIKTI / f"tahminler_{klasor.name}.csv", index=False, encoding="utf-8-sig")
    print(f"  [saldırgan] {klasor.name}: AUC={sonuc['auc']:.4f} doğruluk={sonuc['dogruluk']:.4f} makro F1="
          f"{sonuc['f1_makro']:.4f}" + (f" | makaledeki aynı tohum {makale:.4f} (fark {sonuc['fark']:+.4f}), "
                                          f"5 tohum {sonuc['makaledeki_5_tohum_auc']:.4f}" if makale is not None else ""))
    return sonuc


# ---------------------------------------------------------------- yeniden eğitilen şifreli modeller
def foveahe_yeniden(klasor: Path) -> dict:
    """`boruhatti.egit --model foveahe_*` modelini makaledeki aynı tohum ve bölmenin kayıtlı modeliyle karşılaştırır.

    İki model aynı test görüntülerinde şifresiz çalıştırılır; AUC farkı, tahmin uyumu ve en büyük ağırlık farkı yazılır.
    """
    from experiments.fovea_models import VectorData
    from foveahe.data import load_dataset
    from foveahe.he_models import load_weights
    from foveahe.representation import FoveaSpec
    ayar = json_oku(klasor / "ayar.json")
    model, veri, temsil, tohum = ayar["model"].split()[-1], {"beyin": "brain", "covidqu": "covidqu"}[ayar["veri"]], \
        ayar["temsil"], ayar["tohum"]
    ds = load_dataset(veri)
    data = VectorData(ds, FoveaSpec.parse(temsil), "cpu")
    test = dict(bolum_adlari(ds))
    idx_hepsi, z_yeni, z_makale, agirlik_farki = [], [], {s: [] for s in TOHUMLAR}, 0.0
    for bolum in ayar["bolmeler"]:
        w_yeni = load_weights(klasor / f"{bolum}.npz")
        x = data.get(test[bolum]).double().numpy()
        idx_hepsi.append(test[bolum])
        z_yeni.append(ileri(w_yeni, x))
        for s in TOHUMLAR:
            w_makale = load_weights(agirlik_yolu(f"fovea_{veri}_{model}_{temsil}_s{s}_{bolum}.npz"))
            z_makale[s].append(ileri(w_makale, x))
            if s == tohum:  # sonradan eklenen üst veri anahtarları (ör. n_geom) karşılaştırılmaz
                for k in (set(w_yeni) & set(w_makale)) - {"kind"}:
                    agirlik_farki = max(agirlik_farki, float(np.max(np.abs(np.asarray(w_yeni[k], np.float64)
                                                                            - np.asarray(w_makale[k], np.float64)))))
    idx = np.concatenate(idx_hepsi)
    p_yeni = softmax(np.concatenate(z_yeni))
    p_makale = {s: softmax(np.concatenate(z)) for s, z in z_makale.items()}
    m_yeni = olcumler(ds.y[idx], p_yeni)
    auc_tohum = {s: float(olcumler(ds.y[idx], p)["auc"]) for s, p in p_makale.items()}
    sonuc = {"model": klasor.name, "veri": ayar["veri"], "temsil": temsil, "tohum": tohum,
             "bolmeler": list(ayar["bolmeler"]), "n": len(idx), "auc": float(m_yeni["auc"]),
             "f1_makro": float(m_yeni["f1_makro"]), "makaledeki_model_auc": auc_tohum[tohum],
             "fark": float(m_yeni["auc"] - auc_tohum[tohum]),
             "tahmin_uyumu": float((p_yeni.argmax(1) == p_makale[tohum].argmax(1)).mean()),
             "en_buyuk_agirlik_farki": agirlik_farki, "makaledeki_5_tohum_auc": auc_tohum,
             "makaledeki_5_tohum_auc_aralik": [min(auc_tohum.values()), max(auc_tohum.values())]}
    print(f"  [FoveaHE] {klasor.name}: AUC={sonuc['auc']:.4f} | makaledeki aynı tohum ve bölmenin modeli "
          f"{sonuc['makaledeki_model_auc']:.4f} (fark {sonuc['fark']:+.4f}), tahmin uyumu %{100 * sonuc['tahmin_uyumu']:.1f}, "
          f"en büyük ağırlık farkı {agirlik_farki:.1e}; makaledeki 5 tohum aynı bölmede "
          f"{sonuc['makaledeki_5_tohum_auc_aralik'][0]:.4f}–{sonuc['makaledeki_5_tohum_auc_aralik'][1]:.4f}")
    return sonuc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saldirgan", nargs="*", default=[], help="modeller/<ad> klasörleri")
    ap.add_argument("--foveahe", nargs="*", default=[], help="yeniden eğitilen şifreli modellerin modeller/<ad> klasörleri")
    ap.add_argument("--makale-modelleri", action="store_true")
    ap.add_argument("--sifreli", type=int, default=0, help="CKKS ile şifreli çıkarım yapılacak görüntü sayısı")
    ap.add_argument("--aygit", default=None)
    args = ap.parse_args()
    is_parcacigi(4)
    CIKTI.mkdir(parents=True, exist_ok=True)
    if args.makale_modelleri:
        print("[makale modelleri] Tablo V ve Tablo VIII (şifreli özet) ağırlıklarıyla şifresiz test:")
        df = makale_modelleri()
        df.to_csv(CIKTI / "degerlendirme_makale_modelleri_tohum.csv", index=False)
        oz = makale_karsilastir(df)
        oz.to_csv(CIKTI / "degerlendirme_makale_modelleri.csv", index=False)
        write_markdown_table(oz.assign(tuttu=oz.tuttu.map({True: "evet", False: "HAYIR"})),
                             CIKTI / "degerlendirme_makale_modelleri.md")
        print(f"  makaledeki 5 tohum ortalamasıyla eşleşen: {int(oz.tuttu.sum())}/{len(oz)}; en büyük tohum farkı "
              f"{df.fark.abs().max():.2e}")
    if args.sifreli:
        print(f"[şifreli çıkarım] {args.sifreli} görüntü:")
        json_yaz(CIKTI / "degerlendirme_sifreli.json", sifreli_sinama(args.sifreli))
    if args.saldirgan:
        dev = aygit(args.aygit)
        sonuclar = [saldirgan(Path(k), dev) for k in args.saldirgan]
        json_yaz(CIKTI / "degerlendirme_saldirgan.json", sonuclar)
    if args.foveahe:
        print("[FoveaHE] yeniden eğitilen şifreli modeller, makaledeki aynı tohum ve bölmenin modeliyle:")
        json_yaz(CIKTI / "degerlendirme_foveahe.json", [foveahe_yeniden(Path(k)) for k in args.foveahe])


if __name__ == "__main__":
    main()
