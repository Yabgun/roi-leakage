"""Model eğitimi: makaledeki ana modeller, model dosyası ve eğitim eğrileri (danışman maddeleri 1, 4, 7).

Modeller:
- `saldirgan`: bağlam saldırganı, ImageNet ön eğitimli ResNet-18 (makale Tablo III). Eğitim `attacks.context_cnn`
  ile birebir aynıdır: AdamW, OneCycle (en çok 3e-4), yığın 64, beyin MR 12 / COVID-QU-Ex 5 epoch; beyin MR'da görünür
  piksel normalizasyonu.
- `foveahe_D`, `foveahe_D2`, `foveahe_C`: şifreli çalışabilen modeller (Tablo V). Eğitim `experiments.fovea_models`
  ile birebir aynıdır: AdamW, doğrulama kaybıyla erken durdurma ve ağırlık azaltma seçimi.

Protokoller (`--protokol`):
- `tez`: makaledeki sayıları üreten bölmeler.
  - Saldırgan: beyin MR'da hasta bazlı 5 kat, her katın modeli ayrı kaydedilir. COVID-QU-Ex'te Train + Val ile
    eğitim, Test ile değerlendirme.
  - Şifreli modeller: beyinde test katı k, doğrulama katı k+1; COVID-QU-Ex'te Train / Val / Test.
- `izleme`: ezberleme denetimi için eğitimden ayrı bir doğrulama kümesi; test kümesi eğitimde ve model seçiminde
  kullanılmaz.
  - Saldırgan: beyinde kat 2 doğrulama, kat 3–5 eğitim (kat 1 hiç kullanılmaz); COVID-QU-Ex'te Train ile eğitim, Val
    doğrulama (Test kullanılmaz). Makaledeki saldırgan eğitimi doğrulama kümesi kullanmadığı için bu ayrı bir koşudur.
  - Şifreli modeller: tez protokolünün ilk bölmesi (beyinde test katı 1, doğrulama katı 2; COVID-QU-Ex'te Train / Val /
    Test). Makaledeki eğitim de doğrulama kümesiyle erken durdurduğu için bu koşu makaledeki ilk bölmenin aynısıdır.
    Eğitimden sonra test bölmesinde ölçülür; `degerlendir --foveahe` makaledeki modelle karşılaştırır.
  - Her epoch'ta eğitim ve doğrulama kaybı, doğruluğu, makro F1 ve AUC kaydedilir; her adımda öğrenme oranı.

Kayıt (`modeller/<ad>/`):
- ağırlıklar: saldırganda `.pt`, şifreli modellerde `.pt` ve CKKS'e hazır `.npz`
- `ayar.json`, `sonuc.json` (test ölçüleri; tez protokolünde makaledeki değerle fark), `tahminler.csv`
- `<bölüm>/egitim_egrisi_*.csv|png`

wandb'de giriş yapılmışsa aynı eğriler wandb'ye de gider (`WANDB_ENTITY`, `WANDB_PROJECT`); kapatmak için
`--wandb-kapali` ya da `WANDB_MODE=disabled`.

Örnekler:
  python -m boruhatti.egit --model saldirgan --veri beyin --gorus baglam --protokol tez
  python -m boruhatti.egit --model saldirgan --veri covidqu --gorus baglam --protokol izleme
  python -m boruhatti.egit --model foveahe_D --veri beyin --temsil F32_G16 --protokol izleme
"""
from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402
from analysis.metrikler import olcumler  # noqa: E402
from attacks.context_cnn import EVAL_BATCH, CpuCache, VisibleNormCache, pad_batch, train_and_predict  # noqa: E402
from boruhatti.ortak import MODELLER, Izleyici, aygit, git_surumu, is_parcacigi, json_yaz  # noqa: E402
from common.tez_bicim import SINIF_TEZ, kaydet  # noqa: E402

VERI = {"beyin": "brain", "covidqu": "covidqu"}
EPOCH = {"beyin": 12, "covidqu": 5}
SALDIRGAN_LR, SALDIRGAN_YIGIN = 3e-4, 64
RENK = {"egitim": "#2a78d6", "dogrulama": "#eb6834", "ink": "#52514e", "grid": "#e1e0d9", "surface": "#fcfcfb"}


# ---------------------------------------------------------------- ortak
def capraz_entropi(y: np.ndarray, p: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))))


def ozet_olcu(m: dict) -> dict:
    return {k: float(m[k]) for k in ("auc", "dogruluk", "dengeli_dogruluk", "kesinlik_makro", "duyarlilik_makro",
                                     "f1_makro", "f1_agirlikli")}


def egri_ciz(klasor, baslik: str) -> None:
    """egitim_egrisi_epoch.csv (+ adim) → egitim_egrisi.png: kayıp, doğruluk ve öğrenme oranı."""
    e = klasor / "egitim_egrisi_epoch.csv"
    if not e.exists():
        return
    ep = pd.read_csv(e)
    adim = pd.read_csv(klasor / "egitim_egrisi_adim.csv") if (klasor / "egitim_egrisi_adim.csv").exists() else None
    oneks = list(dict.fromkeys(ep["onek"].fillna(""))) if "onek" in ep else [""]
    panel = 3 if adim is not None else 2
    fig, axes = plt.subplots(1, panel, figsize=(4.2 * panel, 3.4), facecolor=RENK["surface"])
    for onek in oneks:
        d = ep[ep["onek"].fillna("") == onek] if "onek" in ep else ep
        cizgi = "-" if onek == oneks[-1] or len(oneks) == 1 else ":"
        etiket = f" ({onek.rstrip('/')})" if len(oneks) > 1 else ""
        for ax, sutun, ad in ((axes[0], "kaybi", "kayıp"), (axes[1], "dogrulugu", "doğruluk")):
            for kume, renk in (("egitim", RENK["egitim"]), ("dogrulama", RENK["dogrulama"])):
                s = f"{kume}_{sutun}"
                if s in d and d[s].notna().any():
                    ax.plot(d["epoch"], d[s], cizgi, color=renk, lw=1.6, marker="o", ms=3,
                            label=("eğitim" if kume == "egitim" else "doğrulama") + etiket)
            ax.set_title(ad.capitalize(), fontsize=10, color=RENK["ink"], loc="left")
            ax.set_xlabel("epoch", fontsize=9, color=RENK["ink"])
    if adim is not None and "ogrenme_orani" in adim:
        axes[2].plot(adim["adim"], adim["ogrenme_orani"], color=RENK["ink"], lw=1.2)
        axes[2].set_title("Öğrenme oranı", fontsize=10, color=RENK["ink"], loc="left")
        axes[2].set_xlabel("adım", fontsize=9, color=RENK["ink"])
    for ax in axes:
        ax.grid(True, color=RENK["grid"], lw=0.6)
        ax.set_facecolor(RENK["surface"])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8, colors=RENK["ink"])
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle(baslik, fontsize=10, color="#0b0b0b")
    fig.tight_layout()
    kaydet(fig, klasor / "egitim_egrisi", facecolor=RENK["surface"])
    plt.close(fig)


def tahmin_tablosu(klasor, idx, y, p, siniflar) -> None:
    df = pd.DataFrame({"satir": idx, "gercek": [siniflar[i] for i in y[idx]], "tahmin": [siniflar[i] for i in p.argmax(1)]})
    df["dogru_mu"] = np.where(df.gercek == df.tahmin, "evet", "hayır")
    for j, s in enumerate(siniflar):
        df[f"olasilik_{s}"] = np.round(p[:, j], 6)
    df.to_csv(klasor / "tahminler.csv", index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------- saldırgan (ResNet-18)
def saldirgan_verisi(veri: str, norm: str, dev: str):
    from experiments.attack_context import build_cache
    if veri == "beyin":
        meta = pd.read_csv(config.DATA_PROC / "brain" / "meta.csv")
        base = config.DATA_PROC / "brain"
        imgs, masks = build_cache("brain", [base / p for p in meta.img_path], [base / p for p in meta.mask_path])
        y = meta["label"].to_numpy() - 1
        if norm == "gorunur":
            from experiments.brain_visible_norm import load_raw
            cache = VisibleNormCache(load_raw(), masks, y, device=dev)
        else:
            cache = CpuCache(imgs, masks, y, device=dev)
        return cache, y, {"folds": meta["fold"].to_numpy()}, meta.groupby("label")["label_name"].first().tolist()
    man = pd.read_csv(config.DATA_PROC / "covidqu_manifest.csv")
    man = man[man.lung_mask_path.notna() & (man.lung_mask_path != "")].reset_index(drop=True)
    imgs, masks = build_cache("covidqu", man.img_path, man.lung_mask_path)
    etiketler = ["COVID-19", "Non-COVID", "Normal"]
    y = man.label.map({l: i for i, l in enumerate(etiketler)}).to_numpy()
    return CpuCache(imgs, masks, y, device=dev), y, {"split": man["split"].to_numpy()}, etiketler


def saldirgan_bolmeleri(veri: str, protokol: str, bilgi: dict):
    """(bölüm adı, eğitim, doğrulama ya da None, test ya da None) listesi."""
    if veri == "beyin":
        folds = bilgi["folds"]
        katlar = np.unique(folds)
        if protokol == "tez":
            return [(f"kat{k}", np.flatnonzero(folds != k), None, np.flatnonzero(folds == k)) for k in katlar]
        test_k, dog_k = katlar[0], katlar[1]
        return [(f"izleme_dogrulama_kat{dog_k}", np.flatnonzero(~np.isin(folds, [test_k, dog_k])),
                 np.flatnonzero(folds == dog_k), None)]
    split = bilgi["split"]
    if protokol == "tez":
        return [("test", np.flatnonzero(np.isin(split, ["Train", "Val"])), None, np.flatnonzero(split == "Test"))]
    return [("izleme_dogrulama_val", np.flatnonzero(split == "Train"), np.flatnonzero(split == "Val"), None)]


@torch.no_grad()
def saldirgan_olasilik(model, cache, idx, gorus: str, n_cls: int, rng) -> np.ndarray:
    """train_and_predict'in değerlendirmesiyle aynı: 32 bit, karışık sıra, sabit yığın boyutu."""
    model.eval()
    order = rng.permutation(len(idx))
    probs = np.zeros((len(idx), n_cls))
    for i in range(0, len(idx), EVAL_BATCH):
        part = order[i:i + EVAL_BATCH]
        x, _ = cache.batch(pad_batch(idx[part]), gorus, train=False)
        probs[part] = torch.softmax(model(x).double(), 1)[:len(part)].cpu().numpy()
    return probs


def makaledeki_saldirgan_auc(veri: str, gorus: str, norm: str, tohum: int):
    tablo = "saldiri_B_baglam_gnorm.csv" if (veri == "beyin" and norm == "gorunur") else "saldiri_B_baglam.csv"
    df = pd.read_csv(config.TABLES / tablo)
    df = df[(df.gorus == gorus) & (df.tohum == tohum) & (df.veri.str.startswith("beyin" if veri == "beyin" else "akciğer"))]
    return (float(df.auc.iloc[0]), tablo) if len(df) else (None, tablo)


def saldirgan_egit(args) -> dict:
    dev = aygit(args.aygit)
    norm = args.norm if args.veri == "beyin" else "dagitildigi_gibi"
    cache, y, bilgi, etiketler = saldirgan_verisi(args.veri, norm, dev)
    siniflar = [SINIF_TEZ[e] for e in etiketler]
    n_cls, epochs = len(siniflar), args.epoch or EPOCH[args.veri]
    ad = f"saldirgan_{args.veri}_{args.gorus}_{args.protokol}_s{args.tohum}" + ("_hizli" if args.hizli else "")
    klasor = MODELLER / ad
    bolmeler = saldirgan_bolmeleri(args.veri, args.protokol, bilgi)
    if args.hizli:  # boru hattı sınaması: küçük alt küme
        r = np.random.default_rng(0)
        bolmeler = [(b, r.choice(tr, min(600, len(tr)), replace=False),
                     None if va is None else r.choice(va, min(200, len(va)), replace=False),
                     None if te is None else r.choice(te, min(200, len(te)), replace=False)) for b, tr, va, te in bolmeler[:1]]
    ayar = {"model": "saldirgan", "mimari": "ResNet-18 (ImageNet IMAGENET1K_V1), son katman 3 sınıf", "veri": args.veri,
            "gorus": args.gorus, "normalizasyon": norm, "protokol": args.protokol, "tohum": args.tohum, "epoch": epochs,
            "ogrenme_orani_en_cok": SALDIRGAN_LR, "zamanlayici": "OneCycle (pct_start 0.15)", "yigin": SALDIRGAN_YIGIN,
            "agirlik_azaltma": 1e-4, "girdi": "224×224; 3 kanal [görünür, görünür, gizli bölge göstergesi]",
            "siniflar": siniflar, "aygit": dev, "git": git_surumu(), "hizli": args.hizli,
            "bolmeler": {b: {"egitim": len(tr), "dogrulama": 0 if va is None else len(va), "test": 0 if te is None else len(te)}
                         for b, tr, va, te in bolmeler}}
    json_yaz(klasor / "ayar.json", ayar)
    oof = np.full((len(y), n_cls), np.nan)
    rng_izleme = np.random.default_rng(10_000 + args.tohum)  # eğitimin rastgele sayı dizisine dokunmaz
    for bolum, tr, va, te in bolmeler:
        iz = Izleyici(f"{ad}_{bolum}", ad, {**ayar, "bolum": bolum}, [args.veri, args.gorus, args.protokol, "saldirgan"],
                      klasor / bolum, wandb_ac=not args.wandb_kapali)
        t0 = time.perf_counter()

        def izle(olay, b, va=va):
            if olay == "adim":
                iz.adim(b)
                return
            satir = {"egitim_kaybi": b["egitim_kaybi"], "egitim_dogrulugu": b["egitim_dogrulugu"], "sure_s": b["sure_s"]}
            if va is not None:
                p_va = saldirgan_olasilik(b["model"], cache, va, args.gorus, n_cls, rng_izleme)
                m = olcumler(y[va], p_va)
                satir.update({"dogrulama_kaybi": capraz_entropi(y[va], p_va), "dogrulama_dogrulugu": m["dogruluk"],
                              "dogrulama_f1_makro": m["f1_makro"], "dogrulama_auc": m["auc"]})
            iz.epoch(b["epoch"], satir)
            print(f"      [{bolum}] epoch {b['epoch']}: " + ", ".join(f"{k}={v:.4f}" for k, v in satir.items()), flush=True)

        hedef = te if te is not None else va
        probs, model = train_and_predict(cache, tr, hedef, args.gorus, n_cls, epochs=epochs, batch_size=SALDIRGAN_YIGIN,
                                         lr=SALDIRGAN_LR, seed=args.tohum, log=print, izle=izle)
        torch.save(model.state_dict(), klasor / f"{bolum}.pt")
        m = olcumler(y[hedef], probs)
        on = "test" if te is not None else "son_dogrulama"
        iz.ozet({f"{on}/{k}": v for k, v in ozet_olcu(m).items()} | {"sure_s": time.perf_counter() - t0})
        iz.karisiklik(y[hedef], probs.argmax(1), siniflar)
        iz.bitir()
        egri_ciz(klasor / bolum, f"{ad} / {bolum}")
        if te is not None:
            oof[te] = probs
        print(f"  [{bolum}] {on}: AUC={m['auc']:.4f} doğruluk={m['dogruluk']:.4f} makro F1={m['f1_makro']:.4f}", flush=True)
        del model
        torch.cuda.empty_cache() if dev == "cuda" else None
    sonuc = {"ad": ad, "ayar": ayar}
    done = np.flatnonzero(~np.isnan(oof[:, 0]))
    if len(done):
        m = olcumler(y[done], oof[done])
        sonuc["test"] = ozet_olcu(m) | {"n": len(done), "karisiklik": m["karisiklik"].tolist()}
        makale, tablo = makaledeki_saldirgan_auc(args.veri, args.gorus, norm, args.tohum)
        if makale is not None and not args.hizli:
            sonuc["makaledeki_kosu"] = {"tablo": f"results/tables/{tablo}", "auc": makale, "fark": m["auc"] - makale}
        tahmin_tablosu(klasor, done, y, oof[done], siniflar)
        print(f"[{ad}] test (birleşik): AUC={m['auc']:.4f} doğruluk={m['dogruluk']:.4f} makro F1={m['f1_makro']:.4f}"
              + (f" | makaledeki koşu {makale:.4f}, fark {m['auc'] - makale:+.4f}" if "makaledeki_kosu" in sonuc else ""))
    json_yaz(klasor / "sonuc.json", sonuc)
    return sonuc


# ---------------------------------------------------------------- şifreli modeller (D, D2, C)
def fovea_egit(args) -> dict:
    from experiments.fovea_models import (MAX_EPOCHS, PATIENCE, VectorData, export_any, fit_select, predict, splits)
    from foveahe.data import load_dataset
    from foveahe.he_models import save_weights
    from foveahe.representation import FoveaSpec
    dev = aygit(args.aygit)
    tur = args.model.split("_", 1)[1]
    ds = load_dataset(VERI[args.veri])
    siniflar = [SINIF_TEZ[e] for e in ds.labels]
    spec = FoveaSpec.parse(args.temsil)
    data = VectorData(ds, spec, dev)
    bolmeler = splits(ds, quick=args.hizli)
    if args.protokol == "izleme":
        bolmeler = bolmeler[:1]
    ad = f"foveahe_{tur}_{args.veri}_{args.temsil}_{args.protokol}_s{args.tohum}" + ("_hizli" if args.hizli else "")
    klasor = MODELLER / ad
    max_ep, sabir = (3, 2) if args.hizli else (MAX_EPOCHS, PATIENCE)
    ayar = {"model": f"Model {tur}", "veri": args.veri, "temsil": args.temsil, "sifreli_deger": spec.n_values,
            "protokol": args.protokol, "tohum": args.tohum, "en_cok_epoch": max_ep, "sabir": sabir,
            "secim": "ağırlık azaltma {1e-4, 1e-2, 1} (sınırda genişler), doğrulama kaybıyla erken durdurma",
            "siniflar": siniflar, "aygit": dev, "git": git_surumu(), "hizli": args.hizli,
            "bolmeler": {b: {"egitim": len(tr), "dogrulama": len(va), "test": len(te)} for b, tr, va, te in bolmeler}}
    json_yaz(klasor / "ayar.json", ayar)
    probs = np.full((len(ds.y), len(siniflar)), np.nan)
    secim = {}
    for bolum, tr, va, te in bolmeler:
        iz = Izleyici(f"{ad}_{bolum}", ad, {**ayar, "bolum": bolum}, [args.veri, args.temsil, args.protokol, f"model_{tur}"],
                      klasor / bolum, wandb_ac=not args.wandb_kapali)

        def izle_uret(wd, lr, tr=tr, va=va):
            onek = f"wd{wd:g}_lr{lr:g}/"

            def cb(ep, model, egitim_kaybi, dogrulama_kaybi):
                p_tr = predict(model, data, tr, dev, args.tohum, len(siniflar))
                p_va = predict(model, data, va, dev, args.tohum, len(siniflar))
                m_va = olcumler(ds.y[va], p_va)
                iz.epoch(ep, {"egitim_kaybi": egitim_kaybi, "dogrulama_kaybi": dogrulama_kaybi,
                              "egitim_dogrulugu": float((p_tr.argmax(1) == ds.y[tr]).mean()),
                              "dogrulama_dogrulugu": m_va["dogruluk"], "dogrulama_f1_makro": m_va["f1_makro"],
                              "dogrulama_auc": m_va["auc"]}, onek=onek)
            return cb

        model, ep, wd, lr = fit_select(tur, spec, data, ds.y, tr, va, len(siniflar), args.tohum, dev, max_ep, sabir,
                                       izle=izle_uret)
        secim[bolum] = {"agirlik_azaltma": wd, "ogrenme_orani": lr, "en_iyi_epoch": ep}
        p_te = predict(model, data, te, dev, args.tohum, len(siniflar))
        probs[te] = p_te
        save_weights(export_any(tur, model), klasor / f"{bolum}.npz")
        torch.save(model.state_dict(), klasor / f"{bolum}.pt")
        m = olcumler(ds.y[te], p_te)
        iz.ozet({f"test/{k}": v for k, v in ozet_olcu(m).items()} | {"secilen_wd": wd, "secilen_lr": lr, "secilen_epoch": ep})
        iz.karisiklik(ds.y[te], p_te.argmax(1), siniflar)
        iz.bitir()
        egri_ciz(klasor / bolum, f"{ad} / {bolum} (seçilen: wd={wd:g}, lr={lr:g}, epoch {ep})")
        print(f"  [{bolum}] seçilen wd={wd:g} lr={lr:g} epoch={ep} | test AUC={m['auc']:.4f} makro F1={m['f1_makro']:.4f}")
    done = np.flatnonzero(~np.isnan(probs[:, 0]))
    m = olcumler(ds.y[done], probs[done])
    sonuc = {"ad": ad, "ayar": ayar, "secim": secim,
             "test": ozet_olcu(m) | {"n": len(done), "karisiklik": m["karisiklik"].tolist()}}
    tahmin_tablosu(klasor, done, ds.y, probs[done], siniflar)
    json_yaz(klasor / "sonuc.json", sonuc)
    print(f"[{ad}] test: AUC={m['auc']:.4f} doğruluk={m['dogruluk']:.4f} makro F1={m['f1_makro']:.4f}")
    return sonuc


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=["saldirgan", "foveahe_D", "foveahe_D2", "foveahe_C"])
    ap.add_argument("--veri", required=True, choices=list(VERI))
    ap.add_argument("--protokol", default="tez", choices=["tez", "izleme"])
    ap.add_argument("--gorus", default="baglam", help="saldırgan için: tam, baglam, yalniz_roi, baglam_genis40, kutu ...")
    ap.add_argument("--norm", default="gorunur", choices=["gorunur", "kesit"], help="beyin MR normalizasyonu (makale: gorunur)")
    ap.add_argument("--temsil", default="F32_G16", help="şifreli modeller için: F32_G16, F64_G32, U64, U512, U256 ...")
    ap.add_argument("--tohum", type=int, default=0)
    ap.add_argument("--epoch", type=int, default=None, help="saldırgan için (varsayılan: beyin 12, COVID-QU-Ex 5)")
    ap.add_argument("--aygit", default=None, help="cuda ya da cpu (varsayılan: varsa cuda)")
    ap.add_argument("--is-parcacigi", type=int, default=4, help="işlemci iş parçacığı sınırı")
    ap.add_argument("--wandb-kapali", action="store_true")
    ap.add_argument("--hizli", action="store_true", help="boru hattı sınaması: küçük alt küme, 1 epoch")
    args = ap.parse_args()
    if args.hizli and args.epoch is None:
        args.epoch = 1
    is_parcacigi(args.is_parcacigi)
    (saldirgan_egit if args.model == "saldirgan" else fovea_egit)(args)


if __name__ == "__main__":
    main()
