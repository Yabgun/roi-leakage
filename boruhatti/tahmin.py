"""Yeni görüntülere tahmin: projede hiç kullanılmamış görüntüler dahil (danışman maddeleri 5, 6, 7).

Her görüntü için üç modelin tahmini verilir; hepsi aynı hedefi (beyin MR'da tümör tipi, göğüs röntgeninde teşhis)
tahmin eder, yalnızca gördükleri farklıdır:
- `saldirgan_tam`: bağlam saldırganı, tam görüntü (şifreleme yok; ROI maskesi gerekmez)
- `saldirgan_baglam`: bağlam saldırganı, Π_ROI'de sunucunun gördüğü görüntü (ROI gizli; ROI maskesi gerekir)
- `foveahe`: önerilen yöntem (beyin MR Model D, göğüs röntgeni Model D2; F32_G16). İstemcinin şifreleyeceği odaklı
  temsilden tahmin; ROI maskesi gerekir. `--sifreli` ile gerçekten CKKS şifreli çalıştırılır.

Beyin MR'da 5 katın modelleri ortalanır (topluluk); göğüs röntgeninde tek model vardır.

ROI maskesi: `--maske` ile verilir (görüntüyle aynı boyutta ikili PNG; beyaz = ROI). Göğüs röntgeninde
`--otomatik-maske` U-Net ile akciğer maskesini kendisi çıkarır. Beyin MR'da tümör maskesi kullanıcıdan gelmelidir.

Girdi beklentileri: `docs/GIRDI_SOZLESMESI.md`. Kısaca:
- Beyin MR: T1 ağırlıklı kontrastlı 2B kesit, gri ton.
- Göğüs röntgeni: ön-arka/arka-ön grafi, gri ton.
- Görüntü kare değilse kare yapılmadan yeniden boyutlandırılır.

Modeller `modeller/` altında aranır (`boruhatti.egit` çıktıları ve indirilen makale ağırlıkları).

Örnekler:
  python -m boruhatti.tahmin --veri covidqu --goruntu ornek.png --otomatik-maske
  python -m boruhatti.tahmin --veri beyin --goruntu kesit.png --maske tumor.png --sifreli
  python -m boruhatti.tahmin --veri covidqu --klasor yeni_goruntuler/ --otomatik-maske   # klasördeki hepsi → CSV
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from boruhatti.ortak import CIKTI, MODELLER, aygit, is_parcacigi  # noqa: E402
from common.tez_bicim import SINIF_TEZ, kaydet  # noqa: E402

SINIF = {"beyin": ["meningioma", "glioma", "pituitary"], "covidqu": ["COVID-19", "Non-COVID", "Normal"]}
OZGUN = {"beyin": 512, "covidqu": 256}  # FoveaHE temsilinin çıkarıldığı (veri kümesinin dağıtıldığı) çözünürlük
FOVEA_MODEL = {"beyin": ("brain", "D"), "covidqu": ("covidqu", "D2")}
UZANTI = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MODEL_ETIKET = {"saldirgan_tam": "Saldırgan, tam görüntü", "saldirgan_baglam": "Saldırgan, Π_ROI görüşü",
                "foveahe": "FoveaHE"}


def gri_oku(yol, boyut: int | None = None) -> np.ndarray:
    with Image.open(yol) as im:
        g = im.convert("L")
        if boyut is not None and g.size != (boyut, boyut):
            g = g.resize((boyut, boyut), Image.BILINEAR)
        return np.asarray(g, dtype=np.float32) / 255.0


def maske_oku(yol, boyut: int) -> np.ndarray:
    with Image.open(yol) as im:
        g = im.convert("L")
        if g.size != (boyut, boyut):
            g = g.resize((boyut, boyut), Image.NEAREST)
        return np.asarray(g) > 127


def yuzdelik_olcekle(img: np.ndarray) -> np.ndarray:
    """Beyin MR ön işlemesiyle aynı: 0.5–99.5 yüzdelikleri arası [0, 1] (experiments/prepare_data.py:158-159)."""
    lo, hi = np.percentile(img, [0.5, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def otomatik_akciger_maskesi(img224: np.ndarray, dev: str) -> np.ndarray:
    from common.segmentation import postprocess
    from experiments.lung_segmenter import unet_yukle
    yol = next((p for p in (MODELLER / "lung_unet.pt", MODELLER / "makale" / "lung_unet.pt",
                            Path("results/checkpoints/lung_unet.pt")) if p.exists()), None)
    if yol is None:
        raise SystemExit("U-Net ağırlığı bulunamadı (modeller/lung_unet.pt); README'deki model indirme adımına bakın.")
    model = unet_yukle(yol, dev)
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(img224)[None, None].to(dev)).float())[0, 0].cpu().numpy()
    return postprocess(p > 0.5)


class Saldirgan:
    def __init__(self, veri: str, gorus: str, dev: str):
        from attacks.context_cnn import make_model
        klasor = MODELLER / f"saldirgan_{veri}_{gorus}_tez_s0"
        dosyalar = sorted(klasor.glob("*.pt"))
        if not dosyalar:
            raise SystemExit(f"{klasor} altında model yok: önce `python -m boruhatti.egit --model saldirgan --veri {veri} "
                             f"--gorus {gorus}` ya da modelleri indirin.")
        self.gorus, self.veri, self.dev, self.modeller = gorus, veri, dev, []
        for d in dosyalar:
            m = make_model(3)
            m.load_state_dict(torch.load(d, map_location=dev, weights_only=True))
            self.modeller.append(m.to(dev).eval())
        self.ad = f"{len(dosyalar)} model" + (" (5 katın ortalaması)" if len(dosyalar) > 1 else "")

    @torch.no_grad()
    def __call__(self, img224: np.ndarray, maske224: np.ndarray):
        from attacks.context_cnn import hidden_region, normalize_visible, server_view
        x = torch.from_numpy(img224)[None, None].to(self.dev)
        m = torch.from_numpy(maske224)[None, None].to(self.dev)
        if self.veri == "beyin":  # makaledeki görünür piksel normalizasyonu: ölçek yalnız sunucunun gördüğünden
            x = normalize_visible(x, hidden_region(m, self.gorus))
        girdi = server_view(x, m, self.gorus)
        p = np.mean([torch.softmax(mod(girdi).double(), 1)[0].cpu().numpy() for mod in self.modeller], axis=0)
        gorunen = girdi[0, 0].cpu().numpy() * 0.229 + 0.485  # gösterim için normalizasyonu geri al
        return p, np.where(maske224 if self.gorus == "baglam" else False, 0.0, gorunen)


class FoveaHE:
    def __init__(self, veri: str, sifreli: bool):
        from boruhatti.degerlendir import agirlik_yolu
        from foveahe.he_models import load_weights
        ds_ad, model = FOVEA_MODEL[veri]
        bolumler = [f"k{k}" for k in range(1, 6)] if veri == "beyin" else ["test"]
        self.agirliklar = [load_weights(agirlik_yolu(f"fovea_{ds_ad}_{model}_F32_G16_s0_{b}.npz")) for b in bolumler]
        self.ad = f"Model {model}, F32_G16" + (" (5 katın ortalaması)" if len(bolumler) > 1 else "")
        self.ctx = None
        if sifreli:
            from he.piroi import make_context
            self.ctx = make_context()

    def temsil(self, img: np.ndarray, maske: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Önbellekle aynı yol: alan ortalamalı katman → 8 bit nicemleme → [0, 1]; vektör [odak 32², genel 16², geometri]."""
        from foveahe.representation import extract, roi_geometry
        x = torch.from_numpy(img)[None, None]
        g = roi_geometry(torch.from_numpy(maske)[None])
        katmanlar = [(extract(x, g, k)[0, 0].mul(255).round().clamp(0, 255) / 255).numpy() for k in ("f32", "g16")]
        vektor = np.concatenate([katmanlar[0].ravel(), katmanlar[1].ravel(), g[0].numpy()]).astype(np.float64)
        return vektor, katmanlar[0]

    def __call__(self, img: np.ndarray, maske: np.ndarray):
        from boruhatti.degerlendir import ileri, softmax
        vektor, odak = self.temsil(img, maske)
        if self.ctx is None:
            p = np.mean([softmax(ileri(w, vektor[None]))[0] for w in self.agirliklar], axis=0)
            return p, odak, None
        from foveahe.he_infer import FoveaHEInference
        import time
        t0 = time.perf_counter()
        logit = [FoveaHEInference(self.ctx, w).run(vektor, measure_bytes=False)[0] for w in self.agirliklar]
        sure = (time.perf_counter() - t0) / len(logit)
        return np.mean([softmax(z[None])[0] for z in logit], axis=0), odak, sure


def bir_goruntu(yol: Path, maske_yolu, args, modeller, dev) -> dict:
    veri = args.veri
    ozgun = gri_oku(yol, OZGUN[veri])
    img224 = gri_oku(yol, 224)
    if veri == "beyin":  # beyin MR kesitleri yoğunluk ölçeklenir (ön işlemeyle aynı)
        ozgun, img224 = yuzdelik_olcekle(ozgun), yuzdelik_olcekle(img224)
    if maske_yolu is not None:
        maske224, maske_ozgun = maske_oku(maske_yolu, 224), maske_oku(maske_yolu, OZGUN[veri])
    elif args.otomatik_maske and veri == "covidqu":
        maske224 = otomatik_akciger_maskesi(img224, dev)
        maske_ozgun = np.asarray(Image.fromarray(maske224.astype(np.uint8) * 255).resize(
            (OZGUN[veri],) * 2, Image.NEAREST)) > 127
    else:
        maske224 = maske_ozgun = None
    siniflar = [SINIF_TEZ[c] for c in SINIF[veri]]
    satir, panel = {"goruntu": str(yol)}, [("Özgün görüntü", ozgun, maske_ozgun)]
    for ad, model in modeller.items():
        if ad != "saldirgan_tam" and maske224 is None:
            satir[f"{ad}_tahmin"] = "ROI maskesi gerekli"
            continue
        if ad.startswith("saldirgan"):
            p, gorunen = model(img224, maske224 if maske224 is not None else np.zeros((224, 224), bool))
            if ad == "saldirgan_baglam":
                panel.append(("Sunucunun gördüğü (Π_ROI)", gorunen, None))
        else:
            p, odak, sure = model(ozgun, maske_ozgun)
            panel.append(("FoveaHE odağı (şifrelenen)", odak, None))
            if sure is not None:
                satir["foveahe_sifreli_sure_s"] = round(sure, 3)
        satir[f"{ad}_tahmin"] = siniflar[int(np.argmax(p))]
        for j, s in enumerate(siniflar):
            satir[f"{ad}_olasilik_{s}"] = round(float(p[j]), 4)
    if not args.sekilsiz:
        fig, axes = plt.subplots(1, len(panel), figsize=(3.3 * len(panel), 3.6), facecolor="#fcfcfb")
        for ax, (baslik, im, mk) in zip(np.atleast_1d(axes), panel):
            ax.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            if mk is not None:
                ax.contour(mk, levels=[0.5], colors="#eb6834", linewidths=0.9)
            ax.set_title(baslik, fontsize=9)
            ax.axis("off")
        yazi = " | ".join(f"{MODEL_ETIKET.get(a, a)}{' (CKKS)' if a == 'foveahe' and args.sifreli else ''}: "
                          f"{satir[f'{a}_tahmin']}" for a in modeller if f"{a}_tahmin" in satir)
        fig.suptitle(f"{Path(yol).name} — {yazi}", fontsize=8.5)
        fig.tight_layout()
        kaydet(fig, args.cikti / f"tahmin_{Path(yol).stem}", facecolor="#fcfcfb")
        plt.close(fig)
    return satir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--veri", required=True, choices=list(SINIF))
    ap.add_argument("--goruntu", type=Path)
    ap.add_argument("--maske", type=Path)
    ap.add_argument("--klasor", type=Path, help="klasördeki bütün görüntüler (maskeler aynı adla --maske-klasor altında)")
    ap.add_argument("--maske-klasor", type=Path)
    ap.add_argument("--otomatik-maske", action="store_true", help="göğüs röntgeninde U-Net akciğer maskesi")
    ap.add_argument("--modeller", nargs="*", default=["saldirgan_tam", "saldirgan_baglam", "foveahe"])
    ap.add_argument("--sifreli", action="store_true", help="FoveaHE tahminini gerçek CKKS şifreli çalıştır")
    ap.add_argument("--cikti", type=Path, default=CIKTI / "tahmin")
    ap.add_argument("--sekilsiz", action="store_true")
    ap.add_argument("--aygit", default=None)
    args = ap.parse_args()
    if not args.goruntu and not args.klasor:
        ap.error("--goruntu ya da --klasor verin")
    is_parcacigi(4)
    dev = aygit(args.aygit)
    args.cikti.mkdir(parents=True, exist_ok=True)
    modeller = {}
    for ad in args.modeller:
        if ad == "foveahe":
            modeller[ad] = FoveaHE(args.veri, args.sifreli)
        else:
            modeller[ad] = Saldirgan(args.veri, ad.split("_", 1)[1], dev)
        print(f"[model] {ad}: {modeller[ad].ad}")
    if args.goruntu:
        isler = [(args.goruntu, args.maske)]
    else:
        isler = [(p, (args.maske_klasor / p.name) if args.maske_klasor and (args.maske_klasor / p.name).exists() else None)
                 for p in sorted(args.klasor.iterdir()) if p.suffix.lower() in UZANTI]
    satirlar = [bir_goruntu(Path(g), m, args, modeller, dev) for g, m in isler]
    df = pd.DataFrame(satirlar)
    df.to_csv(args.cikti / "tahminler.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.max_columns", 30, "display.width", 200):
        print(df.drop(columns=[c for c in df if "olasilik" in c]).to_string(index=False))
    print(f"\nAyrıntı ve olasılıklar: {args.cikti / 'tahminler.csv'}" + ("" if args.sekilsiz else f"; şekiller: {args.cikti}"))


if __name__ == "__main__":
    main()
