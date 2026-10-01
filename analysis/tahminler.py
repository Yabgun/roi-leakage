"""Görüntü başına gerçek etiket, tahmin ve olasılıklar; etiket sözlüğü ve örnek görüntüler (danışman maddesi 5).

Makaledeki ana modellerin tohum 0 koşularının kayıtlı tahminleri (`results/preds`) görüntü kimlikleriyle birleştirilip
okunur tablolara çevrilir. Yeniden eğitim yapılmaz. Tahmin edilen etiket, olasılığı en yüksek sınıftır.

Çıktılar:
- `results/tahminler/<veri>_<model>_s0.csv`: kimlik, hasta/kat ya da resmi bölme, gerçek etiket, tahmin, doğru mu,
  sınıf olasılıkları
- `results/tahminler/DOSYALAR.md`: dosya listesi ve sütunların anlamı
- `docs/ETIKETLER.md`: her veri kümesinde etiket numarası ↔ sınıf adı ve her modelin girdisi/çıktısı
- `results/figures/tahmin_ornekleri_{beyin,covidqu}.png|pdf`: aynı görüntüde özgün hâl, sunucunun Π_ROI'de gördüğü
  hâl ve saldırganın tahmini, FoveaHE'nin şifrelediği odak ve FoveaHE modelinin tahmini

Çalıştırma: .venv\\Scripts\\python -m analysis.tahminler
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

import config  # noqa: E402
from common.tez_bicim import SINIF_TEZ, kaydet  # noqa: E402
from foveahe.data import STORE, load_dataset  # noqa: E402

PREDS = config.RESULTS / "preds"
OUT = config.RESULTS / "tahminler"
DOCS = config.ROOT / "docs"
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "surface": "#fcfcfb"}
ROI_CIZGI = "#eb6834"

# (dosya adı parçası, makaledeki yeri, model, girdi, kayıtlı tahmin deseni)
MODELLER = {
    "brain": [
        ("meta_veri_saldirgani", "Tablo II", "Meta veri saldırganı (HistGB)", "ROI konum/büyüklük/şekil + girdi boyutu",
         "saldiri_A_beyin_tumor_tipi_tumu_s0.npz"),
        ("baglam_saldirgani_tam", "Tablo III", "Bağlam saldırganı (ResNet-18)", "Tam görüntü (şifreleme yok)",
         "brain_tam_s0_gnorm.npy"),
        ("baglam_saldirgani_piroi", "Tablo III", "Bağlam saldırganı (ResNet-18)", "Π_ROI görüşü: tümör gizli",
         "brain_baglam_s0_gnorm.npy"),
        ("model_D_tam_goruntu", "Tablo V", "Model D (= Π_ROI'nin modeli)", "Tam görüntü 512×512 (şifreli)",
         "fovea_model_brain_D_U512_s0.npy"),
        ("model_D_foveahe_F32_G16", "Tablo V, VIII", "Model D", "FoveaHE F32_G16 (şifreli)",
         "fovea_model_brain_D_F32_G16_s0.npy"),
        ("model_C_foveahe_F32_G16", "Tablo V", "Model C", "FoveaHE F32_G16 (şifreli)",
         "fovea_model_brain_C_F32_G16_s0.npy"),
        ("sifreli_ozet_model_D", "Tablo VIII", "Şifreli özet + Model D", "ResNet-18 özeti, 512 değer (şifreli)",
         "ozet_brain_D_s0.npy"),
    ],
    "covidqu": [
        ("meta_veri_saldirgani", "Tablo II", "Meta veri saldırganı (HistGB)", "Akciğer maskesinin konum/büyüklük/şekli",
         "saldiri_A_covidqu_teshis_tumu_s0.npz"),
        ("baglam_saldirgani_tam", "Tablo III", "Bağlam saldırganı (ResNet-18)", "Tam görüntü (şifreleme yok)",
         "covidqu_tam_s0.npy"),
        ("baglam_saldirgani_piroi", "Tablo III", "Bağlam saldırganı (ResNet-18)", "Π_ROI görüşü: akciğerler gizli",
         "covidqu_baglam_s0.npy"),
        ("model_D2_tam_goruntu", "Tablo V", "Model D2", "Tam görüntü 256×256 (şifreli)",
         "fovea_model_covidqu_D2_U256_s0.npy"),
        ("model_D2_foveahe_F32_G16", "Tablo V, VIII", "Model D2", "FoveaHE F32_G16 (şifreli)",
         "fovea_model_covidqu_D2_F32_G16_s0.npy"),
        ("model_C_foveahe_F32_G16", "Tablo V", "Model C", "FoveaHE F32_G16 (şifreli)",
         "fovea_model_covidqu_C_F32_G16_s0.npy"),
        ("sifreli_ozet_model_D2", "Tablo VIII", "Şifreli özet + Model D2", "ResNet-18 özeti, 512 değer (şifreli)",
         "ozet_covidqu_D2_s0.npy"),
    ],
}
VERI_KISA = {"brain": "beyin", "covidqu": "covidqu"}


def kimlikler(ds_name: str) -> pd.DataFrame:
    if ds_name == "brain":
        meta = pd.read_csv(config.DATA_PROC / "brain" / "meta.csv")
        return pd.DataFrame({"goruntu_no": meta["id"], "dosya": "brain_tumor_figshare/" + meta["id"].astype(str) + ".mat",
                             "hasta": meta["pid"], "kat": meta["fold"]})
    man = pd.read_csv(config.DATA_PROC / "covidqu_manifest.csv")
    man = man[man.lung_mask_path.notna() & (man.lung_mask_path != "")].reset_index(drop=True)
    yol = man.img_path.str.replace("\\", "/", regex=False)
    return pd.DataFrame({"dosya": yol.str.split("extracted/").str[-1], "resmi_bolme": man["split"]})


def tahmin_oku(desen: str, ds):
    path = PREDS / desen
    if path.suffix == ".npz":
        with np.load(path) as z:
            return z["idx"], z["proba"]
    p = np.load(path)
    if p.shape[0] == len(ds.y):
        idx = np.flatnonzero(~np.isnan(p[:, 0]))
        return idx, p[idx]
    return ds.test_idx, p


def tablolar(ds_name: str, ds) -> dict:
    kim = kimlikler(ds_name)
    siniflar = [SINIF_TEZ[c] for c in ds.labels]
    sonuc = {}
    for parca, tablo, model, girdi, desen in MODELLER[ds_name]:
        idx, p = tahmin_oku(desen, ds)
        yp = p.argmax(1)
        df = kim.iloc[idx].reset_index(drop=True)
        df.insert(0, "satir", idx)
        df["gercek_etiket_no"] = ds.y[idx]
        df["gercek_etiket"] = [siniflar[i] for i in ds.y[idx]]
        df["tahmin_no"] = yp
        df["tahmin"] = [siniflar[i] for i in yp]
        df["dogru_mu"] = np.where(yp == ds.y[idx], "evet", "hayır")
        for j, ad in enumerate(siniflar):
            df[f"olasilik_{ad}"] = np.round(p[:, j], 6)
        ad = f"{VERI_KISA[ds_name]}_{parca}_s0.csv"
        df.to_csv(OUT / ad, index=False, encoding="utf-8-sig")  # Excel'de Türkçe karakterler bozulmasın
        sonuc[parca] = {"dosya": ad, "tablo": tablo, "model": model, "girdi": girdi, "n": len(df),
                        "dogruluk": float((yp == ds.y[idx]).mean()), "idx": idx, "p": p}
    return sonuc


def dosyalar_md(ozet: dict) -> None:
    lines = ["# Görüntü başına tahminler (tohum 0)", "",
             "Her dosyada bir satır bir test görüntüsüdür. Tahmin edilen etiket, olasılığı en yüksek sınıftır. Dosyalar "
             "`python -m analysis.tahminler` ile kayıtlı tahminlerden üretilir; yeniden eğitim yapılmaz.", "",
             "| Dosya | Makalede | Model | Girdi | Görüntü | Doğruluk |", "|---|---|---|---|---|---|"]
    for ds_name, s in ozet.items():
        for v in s.values():
            lines.append(f"| `{v['dosya']}` | {v['tablo']} | {v['model']} | {v['girdi']} | {v['n']} | {v['dogruluk']:.3f} |")
    lines += ["", "## Sütunlar", "",
              "- `satir`: manifestodaki satır numarası (kodun içindeki sıra).",
              "- Beyin MR: `goruntu_no` (figshare dosya numarası), `dosya`, `hasta` (hasta kimliği), `kat` (resmi 5 kattan "
              "hangisinde test edildiği).",
              "- COVID-QU-Ex: `dosya` (veri kümesindeki yol), `resmi_bolme` (Test).",
              "- `gercek_etiket_no`, `gercek_etiket`: veri kümesinin etiketi (numaralar `docs/ETIKETLER.md`'de).",
              "- `tahmin_no`, `tahmin`: modelin tahmini; `dogru_mu`: tahmin gerçek etiketle aynı mı.",
              "- `olasilik_<sınıf>`: modelin o sınıfa verdiği olasılık; satır toplamı 1.", "",
              "Beyin MR'da her kesit, hastası test katındayken bir kez tahmin edilir; 5 katın tahminleri birleşince bütün "
              "kesitler tabloda yer alır. COVID-QU-Ex'te yalnızca resmi Test bölmesi tahmin edilir."]
    (OUT / "DOSYALAR.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def etiketler_md(dsets: dict) -> None:
    b, c = dsets["brain"], dsets["covidqu"]
    lines = ["# Etiketler ve modellerin girdi/çıktısı", "",
             "Bu dosya `python -m analysis.tahminler` ile koddaki etiket listelerinden üretilir.", "",
             "## Etiket numaraları", "",
             "| Veri kümesi | Numara | Sınıf (koddaki ad) | Türkçe ad | Kaynak etiketi |", "|---|---|---|---|---|"]
    for i, ad in enumerate(b.labels):
        lines.append(f"| Beyin MR (figshare 1512427) | {i} | {ad} | {SINIF_TEZ[ad]} | `cjdata.label` = {i + 1} |")
    for i, ad in enumerate(c.labels):
        lines.append(f"| COVID-QU-Ex | {i} | {ad} | {SINIF_TEZ[ad]} | klasör adı `{ad}` |")
    lines += ["| Kaggle CXR (prashant268) | – | COVID19 / NORMAL / PNEUMONIA | COVID-19 / Normal / Pnömoni | klasör adı |", "",
              "Kaggle kümesi meta veri (girdi boyutu) deneyinde, gerçekçilik testinde (bir yönde saldırganın eğitim "
              "kümesi, diğer yönde test kümesi) ve genelleme testinde (yalnız test) kullanılır.", "",
              "İkili görevler: Tablo II'deki \"normal/pnömoni\" satırları ve gerçekçilik testi normal ile pnömoniyi "
              "ayırır; normal = 0, pnömoni = 1 (COVID-QU-Ex'te `Non-COVID`, Kaggle'da `PNEUMONIA`).", "",
              "## Her model neyi görür, neyi tahmin eder", "",
              "Makalenin ana tablolarında hedef (etiket) bütün modellerde aynıdır: beyin MR'da tümör tipi, göğüs "
              "röntgeninde 3 sınıflı teşhis. Modeller **girdileri** bakımından ayrılır.", "",
              "| Model | Makalede | Girdi | Çıktı |", "|---|---|---|---|",
              "| Meta veri saldırganı (HistGradientBoosting) | Tablo II | Π_ROI'nin sunucuya açtığı ROI konumu, büyüklüğü, "
              "şekli ve girdi boyutu (sayısal öznitelikler) | 3 sınıf olasılığı |",
              "| Bağlam saldırganı (ImageNet ön eğitimli ResNet-18) | Tablo III | Sunucunun gördüğü 224×224 görüntü: ROI "
              "pikselleri sıfır + gizli bölge göstergesi kanalı | 3 sınıf olasılığı |",
              "| Model D / D2 / C | Tablo V, VIII | Tam görüntü (Π_ROI'nin teşhis modeli) ya da FoveaHE temsili; şifreli "
              "çalışır (CKKS) | 3 sınıf skoru (şifreli; istemci çözer) |",
              "| Şifreli özet + Model D / D2 | Tablo VIII | İstemcide ResNet-18'in çıkardığı 512 değerlik özet; şifreli | "
              "3 sınıf skoru (şifreli) |",
              "| U-Net | Metin (gerçekçilik testi) | Göğüs röntgeni | Akciğer maskesi (etiket değil) |", "",
              "Saldırganlar sunucunun yerine geçer: gizlenmiş bilgiyi (teşhisi) sunucunun görebildiğinden tahmin etmeye "
              "çalışır. Başarılı olmaları sızıntı demektir. Model D, D2, C ve şifreli özet ise hastanenin istediği teşhis "
              "hizmetidir; başarılı olmaları istenir."]
    (DOCS / "ETIKETLER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def goruntu(ds_name: str, ds, i: int):
    with Image.open(ds.img_paths[i]) as im:
        img = np.asarray(im.convert("L"), dtype=np.float32) / 255
    with Image.open(ds.mask_paths[i]) as mm:
        mask = np.asarray(mm.convert("L")) > 127
    return img, mask


def ornek_sec(ds, saldirgan, fovea, rng, adet: int = 2) -> list[int]:
    """Her sınıftan, hiçbir modelin sonucuna bakmadan, sabit tohumla rastgele `adet` test görüntüsü.

    Saldırganın doğru/yanlış olduğu örneklere göre seçmek (ilk sürüm) zor görüntüleri öne çıkarıyor ve diğer modelin
    başarısını olduğundan düşük gösteriyordu; hatalar karışıklık matrislerinde ve tahmin tablolarında tam olarak var.
    """
    ortak = np.intersect1d(saldirgan["idx"], fovea["idx"])
    secim = []
    for c in range(len(ds.labels)):
        secim += [int(i) for i in rng.choice(ortak[ds.y[ortak] == c], adet, replace=False)]
    return secim


def ornek_sekli(ds_name: str, ds, s: dict) -> None:
    saldirgan = s["baglam_saldirgani_piroi"]
    fovea = s["model_D_foveahe_F32_G16" if ds_name == "brain" else "model_D2_foveahe_F32_G16"]
    fovea_ad = "Model D" if ds_name == "brain" else "Model D2"
    secim = ornek_sec(ds, saldirgan, fovea, np.random.default_rng(0))
    odak = np.load(STORE / ds_name / "f32.npy", mmap_mode="r")
    siniflar = [SINIF_TEZ[c] for c in ds.labels]
    fig, axes = plt.subplots(3, len(secim), figsize=(2.15 * len(secim), 7.6), facecolor=INK["surface"])

    def tahmin_yazi(kaynak, i):
        pos = int(np.flatnonzero(kaynak["idx"] == i)[0])
        p = kaynak["p"][pos]
        k = int(p.argmax())
        durum = "doğru" if k == ds.y[i] else "YANLIŞ"
        return f"{siniflar[k]} (%{100 * p[k]:.0f})\n{durum}"

    for col, i in enumerate(secim):
        img, mask = goruntu(ds_name, ds, i)
        ax = axes[0, col]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.contour(mask, levels=[0.5], colors=ROI_CIZGI, linewidths=0.9)
        ax.set_title(f"Gerçek: {siniflar[ds.y[i]]}", fontsize=8.5, color=INK["primary"])
        ax = axes[1, col]
        ax.imshow(np.where(mask, 0.0, img), cmap="gray", vmin=0, vmax=1)
        ax.set_title(tahmin_yazi(saldirgan, i), fontsize=8, color=INK["primary"])
        ax = axes[2, col]
        ax.imshow(odak[i], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(tahmin_yazi(fovea, i), fontsize=8, color=INK["primary"])
        for r in range(3):
            axes[r, col].set_xticks([])
            axes[r, col].set_yticks([])
            for sp in axes[r, col].spines.values():
                sp.set_color(INK["muted"])
    satir_adi = ["Özgün görüntü\n(turuncu: ROI)", "Sunucunun gördüğü\n(Π_ROI) ve\nsaldırganın tahmini",
                 f"FoveaHE'nin şifrelediği\nodak (32×32) ve\n{fovea_ad} tahmini"]
    for r, ad in enumerate(satir_adi):
        axes[r, 0].set_ylabel(ad, fontsize=8.5, color=INK["secondary"])
    veri_ad = "Beyin MR" if ds_name == "brain" else "COVID-QU-Ex"
    fig.suptitle(f"{veri_ad}: aynı test görüntüsünde gerçek etiket ve tahminler (modeller tohum 0; her sınıftan "
                 "rastgele 2 test görüntüsü, seçim modellerin sonucuna bakmadan)", fontsize=9.5, color=INK["primary"])
    fig.tight_layout()
    kaydet(fig, config.FIGURES / f"tahmin_ornekleri_{VERI_KISA[ds_name]}", facecolor=INK["surface"])
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    dsets = {n: load_dataset(n) for n in ("brain", "covidqu")}
    ozet = {}
    for n, ds in dsets.items():
        ozet[n] = tablolar(n, ds)
        ornek_sekli(n, ds, ozet[n])
        for v in ozet[n].values():
            print(f"{v['dosya']:48s} n={v['n']:5d} doğruluk={v['dogruluk']:.3f}")
    dosyalar_md(ozet)
    etiketler_md(dsets)


if __name__ == "__main__":
    main()
