"""Kopya görüntü taraması: ezberleme denetimi ve yeni (dış) veri kümeleri için (danışman maddesi 4 ve dış test).

Yöntem makaledekiyle aynıdır (`experiments/prepare_data.py`): 64 bitlik fark hash'i (dHash) ile aday çiftler
(Hamming ≤ 10), sonra 32×32 normalize küçük resimlerin korelasyonu ≥ 0.97 olanlar kopya sayılır.

Kipler:
- `--ic`: çalışmanın kendi bölmeleri arasında kopya. COVID-QU-Ex'te Train + Val ile Test; beyin MR'da beş kat
  arasında. Test görüntüsünün eğitimde bir kopyası varsa model onu "ezberlemiş" olabilir; bu tarama bunu ölçer.
- `--klasor YOL`: yeni bir klasördeki görüntüler, çalışmanın üç veri kümesiyle (COVID-QU-Ex, Kaggle CXR, beyin MR)
  karşılaştırılır. Kaggle'daki birçok beyin MR ve göğüs röntgeni derlemesi aynı kaynaklardan toplanmıştır (makalede
  Kaggle CXR'nin 6432 görüntüsünün 4665'i COVID-QU-Ex'te çıktı). "Hiç kullanılmamış" bir kümeyle test etmeden önce
  bu tarama çalıştırılmalıdır.

Çıktılar: `results/tables/kopya_ic_*.csv|md` ya da `results/boruhatti/kopya_<klasör>.csv`.

Örnekler:
  python -m boruhatti.kopya_kontrol --ic
  python -m boruhatti.kopya_kontrol --klasor D:/yeni_beyin_mr --veri beyin
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import config
from boruhatti.ortak import CIKTI
from common.imhash import near_duplicate_pairs
from common.report import write_markdown_table
from experiments.prepare_data import signature

MAKS_HAMMING, MIN_KORELASYON = 10, 0.97  # experiments/prepare_data.py:180 ile aynı
UZANTI = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def imzalar(yollar, is_parcacigi: int = 4):
    """(dHash uint64, 32×32 küçük resim float32 (N, 1024))."""
    with ThreadPoolExecutor(is_parcacigi) as ex:
        s = list(ex.map(signature, [str(p) for p in yollar]))
    return (np.array([int(x[2], 16) for x in s], dtype=np.uint64),
            np.stack([x[3] for x in s]).astype(np.float32).reshape(len(s), -1))


def kopyalar(ha, ta, hb, tb) -> pd.DataFrame:
    """A ve B arasında doğrulanmış kopya çiftleri (i: A, j: B)."""
    satir = []
    for i, j, d in near_duplicate_pairs(ha, hb, max_dist=MAKS_HAMMING):
        kor = float(np.dot(ta[i], tb[j]) / ta.shape[1])
        if kor >= MIN_KORELASYON:
            satir.append({"i": i, "j": j, "hamming": d, "korelasyon": kor})
    return pd.DataFrame(satir, columns=["i", "j", "hamming", "korelasyon"])


def covidqu_imzalari():
    man = pd.read_csv(config.DATA_PROC / "covidqu_manifest.csv")
    man = man[man.lung_mask_path.notna() & (man.lung_mask_path != "")].reset_index(drop=True)
    tum = pd.read_csv(config.DATA_PROC / "covidqu_manifest.csv")  # küçük resimler tüm manifesto sırasında
    th = np.load(config.DATA_PROC / "covidqu_thumbs.npy").astype(np.float32).reshape(len(tum), -1)
    if len(tum) != len(man):
        th = th[tum.lung_mask_path.notna() & (tum.lung_mask_path != "")]
    return man, np.array([int(h, 16) for h in man.dhash], dtype=np.uint64), th


def beyin_imzalari():
    meta = pd.read_csv(config.DATA_PROC / "brain" / "meta.csv")
    h, t = imzalar([config.DATA_PROC / "brain" / p for p in meta.img_path])
    return meta, h, t


def ic_tarama() -> None:
    sonuc = []
    man, h, t = covidqu_imzalari()
    egitim = np.flatnonzero(man.split.isin(["Train", "Val"]).to_numpy())
    test = np.flatnonzero((man.split == "Test").to_numpy())
    k = kopyalar(h[test], t[test], h[egitim], t[egitim])
    if len(k):
        k = k.assign(test_dosya=man.file.to_numpy()[test[k.i]], test_etiket=man.label.to_numpy()[test[k.i]],
                     egitim_dosya=man.file.to_numpy()[egitim[k.j]], egitim_bolme=man.split.to_numpy()[egitim[k.j]],
                     egitim_etiket=man.label.to_numpy()[egitim[k.j]])
    k.to_csv(config.TABLES / "kopya_ic_covidqu.csv", index=False)
    n_test = k.i.nunique() if len(k) else 0
    ayni = int((k.test_etiket == k.egitim_etiket).sum()) if len(k) else 0
    sonuc.append({"veri": "COVID-QU-Ex", "karsilastirma": "Test ↔ Train + Val", "test_goruntu": len(test),
                  "kopyasi_olan_test_goruntusu": n_test, "oran": n_test / len(test), "kopya_cifti": len(k),
                  "ayni_etiketli_cift": ayni})
    meta, hb, tb = beyin_imzalari()
    for kat in np.unique(meta.fold):
        te, tr = np.flatnonzero(meta.fold == kat), np.flatnonzero(meta.fold != kat)
        kb = kopyalar(hb[te], tb[te], hb[tr], tb[tr])
        n = kb.i.nunique() if len(kb) else 0
        ayni_hasta = int((meta.pid.to_numpy()[te[kb.i]] == meta.pid.to_numpy()[tr[kb.j]]).sum()) if len(kb) else 0
        sonuc.append({"veri": "Beyin MR", "karsilastirma": f"Kat {kat} ↔ diğer 4 kat", "test_goruntu": len(te),
                      "kopyasi_olan_test_goruntusu": n, "oran": n / len(te), "kopya_cifti": len(kb),
                      "ayni_etiketli_cift": int((meta.label.to_numpy()[te[kb.i]] == meta.label.to_numpy()[tr[kb.j]]).sum())
                      if len(kb) else 0, "ayni_hasta": ayni_hasta})
    df = pd.DataFrame(sonuc)
    df.to_csv(config.TABLES / "kopya_ic_ozet.csv", index=False)
    write_markdown_table(df, config.TABLES / "kopya_ic_ozet.md")
    print(df.to_string(index=False))


def etki_analizi() -> None:
    """Şüpheli test görüntüleri çıkarılınca makaledeki ana sonuçlar ne kadar değişir (kayıtlı tahminlerle; eğitim yok).

    Şüpheli sayılanlar:
    - COVID-QU-Ex: Train + Val'de kopyası olan Test görüntüleri.
    - Beyin MR: başka katta kopyası olan kesitler ve hasta kimliği yalnız son harfiyle ayrılan (ör. MR024780B /
      MR024780E) hastası başka katta da bulunan kesitler. Bu ikincisi aynı kişinin farklı çekimleri olabilir.
    """
    import re
    from common.evaluation import auc_score
    from foveahe.data import load_dataset
    sonuc = []
    man, h, t = covidqu_imzalari()
    egitim = np.flatnonzero(man.split.isin(["Train", "Val"]).to_numpy())
    test = np.flatnonzero((man.split == "Test").to_numpy())
    k = kopyalar(h[test], t[test], h[egitim], t[egitim])
    supheli_c = set(test[k.i.unique()]) if len(k) else set()
    meta, hb, tb = beyin_imzalari()
    supheli_b = set()
    for kat in np.unique(meta.fold):
        te, tr = np.flatnonzero(meta.fold == kat), np.flatnonzero(meta.fold != kat)
        kb = kopyalar(hb[te], tb[te], hb[tr], tb[tr])
        supheli_b |= set(te[kb.i.unique()]) if len(kb) else set()
    kok = meta.pid.astype(str).map(lambda p: re.sub(r"[A-Za-z]$", "", p))
    cok_katli = kok.groupby(kok).transform(lambda s: meta.fold[s.index].nunique() > 1)
    supheli_b_kok = set(np.flatnonzero(cok_katli.to_numpy()))
    print(f"Beyin MR: hasta kimliği son harf ayrılınca birden fazla katta görünen kök kimlik: "
          f"{kok[cok_katli].nunique()} ({int(cok_katli.sum())} kesit)")
    ds = {"brain": load_dataset("brain"), "covidqu": load_dataset("covidqu")}
    modeller = [("brain", "Bağlam saldırganı, Π_ROI görüşü", "brain_baglam_s{s}_gnorm.npy"),
                ("brain", "Bağlam saldırganı, tam görüş", "brain_tam_s{s}_gnorm.npy"),
                ("brain", "Model D, FoveaHE F32_G16", "fovea_model_brain_D_F32_G16_s{s}.npy"),
                ("brain", "Model D, tam görüntü (Π_ROI)", "fovea_model_brain_D_U512_s{s}.npy"),
                ("covidqu", "Bağlam saldırganı, Π_ROI görüşü", "covidqu_baglam_s{s}.npy"),
                ("covidqu", "Bağlam saldırganı, tam görüş", "covidqu_tam_s{s}.npy"),
                ("covidqu", "Model D2, FoveaHE F32_G16", "fovea_model_covidqu_D2_F32_G16_s{s}.npy"),
                ("covidqu", "Model D2, tam görüntü (Π_ROI)", "fovea_model_covidqu_D2_U256_s{s}.npy")]
    for veri, ad, desen in modeller:
        d = ds[veri]
        cikar = {"Kopyası olan test görüntüleri": supheli_b if veri == "brain" else supheli_c}
        if veri == "brain":
            cikar["Kopyası olan + kök kimliği başka katta olan"] = supheli_b | supheli_b_kok
        satir = {"veri": "Beyin MR" if veri == "brain" else "COVID-QU-Ex", "model": ad}
        tum, kalan = {}, {a: [] for a in cikar}
        tum_aucs = []
        for s in range(5):
            p = np.load(config.RESULTS / "preds" / desen.format(s=s))
            idx = np.flatnonzero(~np.isnan(p[:, 0])) if p.shape[0] == len(d.y) else d.test_idx
            p = p[idx] if p.shape[0] == len(d.y) else p
            tum_aucs.append(auc_score(d.y[idx], p))
            for a, kume in cikar.items():
                tut = ~np.isin(idx, list(kume))
                kalan[a].append(auc_score(d.y[idx][tut], p[tut]))
                tum[a] = int((~tut).sum())
        satir["auc_hepsi"] = float(np.mean(tum_aucs))
        for a in cikar:
            satir[f"cikarilan ({a})"] = tum[a]
            satir[f"auc ({a} çıkarılınca)"] = float(np.mean(kalan[a]))
            satir[f"fark ({a})"] = float(np.mean(kalan[a]) - np.mean(tum_aucs))
        sonuc.append(satir)
    df = pd.DataFrame(sonuc)
    sayilar = [c for c in df if c.startswith("cikarilan")]  # COVID satırlarında boş olduğu için ondalıklı görünmesin
    df = df.assign(**{c: df[c].astype("Int64") for c in sayilar})
    df.to_csv(config.TABLES / "kopya_etki.csv", index=False)
    write_markdown_table(df.assign(**{c: df[c].map(lambda v: "" if pd.isna(v) else str(int(v))) for c in sayilar}),
                         config.TABLES / "kopya_etki.md")
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(df.round(4).to_string(index=False))


def klasor_tarama(klasor: Path, veri: str) -> None:
    yollar = sorted(p for p in klasor.rglob("*") if p.suffix.lower() in UZANTI)
    if not yollar:
        raise SystemExit(f"{klasor} altında görüntü yok")
    print(f"{len(yollar)} görüntünün imzası hesaplanıyor...")
    h, t = imzalar(yollar)
    kumeler = []
    if veri in ("covidqu", "hepsi"):
        man, hc, tc = covidqu_imzalari()
        kumeler.append(("COVID-QU-Ex", hc, tc, man.file.to_numpy(), man.split.to_numpy()))
        kg = pd.read_csv(config.DATA_PROC / "kaggle_cxr_manifest.csv")
        tk = np.load(config.DATA_PROC / "kaggle_cxr_thumbs.npy").astype(np.float32).reshape(len(kg), -1)
        kumeler.append(("Kaggle CXR", np.array([int(x, 16) for x in kg.dhash], dtype=np.uint64), tk,
                        kg.img_path.map(lambda s: Path(s).name).to_numpy(), kg.split.to_numpy()))
    if veri in ("beyin", "hepsi"):
        meta, hb, tb = beyin_imzalari()
        kumeler.append(("Beyin MR (figshare)", hb, tb, meta.img_path.to_numpy(), meta.fold.astype(str).to_numpy()))
    satir = []
    for ad, hk, tk, dosya, bolme in kumeler:
        k = kopyalar(h, t, hk, tk)
        for r in k.itertuples(index=False):
            satir.append({"yeni_goruntu": str(yollar[r.i]), "kume": ad, "kumedeki_dosya": dosya[r.j],
                          "kumedeki_bolme": bolme[r.j], "hamming": r.hamming, "korelasyon": round(r.korelasyon, 4)})
    df = pd.DataFrame(satir)
    CIKTI.mkdir(parents=True, exist_ok=True)
    cikti = CIKTI / f"kopya_{klasor.name}.csv"
    df.to_csv(cikti, index=False, encoding="utf-8-sig")
    n = df.yeni_goruntu.nunique() if len(df) else 0
    print(f"{len(yollar)} yeni görüntünün {n}'inin (%{100 * n / len(yollar):.1f}) çalışmanın veri kümelerinde kopyası var.")
    if n:
        print(df.groupby("kume").yeni_goruntu.nunique().to_string())
        print(f"Ayrıntı: {cikti}. Kopyası olan görüntüleri dış testten çıkarın.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ic", action="store_true")
    ap.add_argument("--klasor", type=Path)
    ap.add_argument("--veri", default="hepsi", choices=["hepsi", "beyin", "covidqu"])
    args = ap.parse_args()
    if not args.ic and not args.klasor:
        ap.error("--ic ya da --klasor verin")
    if args.ic:
        ic_tarama()
        etki_analizi()
    if args.klasor:
        klasor_tarama(args.klasor, args.veri)


if __name__ == "__main__":
    main()
