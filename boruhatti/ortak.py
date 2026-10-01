"""Boru hattının ortak yardımcıları: aygıt seçimi, yollar, JSON, eğitim eğrisi kaydı (yerel CSV/PNG + wandb)."""
from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

import config

MODELLER = config.ROOT / "modeller"             # eğitilen ya da indirilen model dosyaları (git dışı)
CIKTI = config.RESULTS / "boruhatti"            # değerlendirme çıktıları
BEKLENEN = config.RESULTS / "beklenen" / "makale_degerleri.json"
WANDB_PROJE = os.environ.get("WANDB_PROJECT", "roi-leakage")


def aygit(istek: str | None = None) -> str:
    """'cuda' varsa onu, yoksa 'cpu' döndürür. İstenen aygıt yoksa açık bir hata verir."""
    import torch
    if istek in (None, "", "otomatik"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if istek == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA istendi ama ekran kartı bulunamadı. `python -m boruhatti.ortam_kontrol` çalıştırın.")
    return istek


def is_parcacigi(n: int = 4) -> None:
    """İşlemciyi kilitlememek için PyTorch iş parçacığı sayısını sınırlar."""
    import torch
    torch.set_num_threads(max(1, min(n, os.cpu_count() or n)))


def git_surumu() -> str:
    try:
        return subprocess.run(["git", "-C", str(config.ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or "bilinmiyor"
    except Exception:
        return "bilinmiyor"


def json_yaz(yol, veri) -> None:
    yol = Path(yol)
    yol.parent.mkdir(parents=True, exist_ok=True)
    yol.write_text(json.dumps(veri, ensure_ascii=False, indent=2, default=_json_cevir), encoding="utf-8")


def json_oku(yol):
    return json.loads(Path(yol).read_text(encoding="utf-8"))


def _json_cevir(v):
    import numpy as np
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, Path):
        return str(v)
    raise TypeError(type(v))


class Izleyici:
    """Eğitim eğrisi kaydı. Her zaman yerel CSV'ye yazar; wandb kuruluysa ve giriş yapılmışsa wandb'ye de gönderir.

    wandb'yi kapatmak için `WANDB_MODE=disabled`; takım/kişi için `WANDB_ENTITY`; proje adı `WANDB_PROJECT`
    (varsayılan roi-leakage). Adım ölçüleri "adim/..." anahtarlarıyla `adim` eksenine, epoch ölçüleri
    "<önek>epoch/..." anahtarlarıyla kendi epoch eksenine çizilir.
    """

    def __init__(self, ad: str, grup: str, ayar: dict, etiketler: list[str], klasor: Path, wandb_ac: bool = True):
        self.ad, self.klasor = ad, Path(klasor)
        self.klasor.mkdir(parents=True, exist_ok=True)
        self.adimlar, self.epochlar, self.run, self._eksenler = [], [], None, set()
        self._adim = 0
        if not wandb_ac or os.environ.get("WANDB_MODE") == "disabled":
            return
        try:
            import wandb
            self.run = wandb.init(project=WANDB_PROJE, entity=os.environ.get("WANDB_ENTITY") or None, name=ad,
                                  group=grup, config=ayar, tags=etiketler, dir=str(config.LOGS),
                                  reinit="finish_previous")
            self.run.define_metric("adim")
            self.run.define_metric("adim/*", step_metric="adim")
        except Exception as e:  # giriş yapılmamışsa ya da bağlantı yoksa yerel kayıtla sürer
            print(f"[wandb] başlatılamadı, yalnız yerel kayıt yapılacak: {type(e).__name__}: {e}")
            self.run = None

    @property
    def url(self) -> str | None:
        return self.run.url if self.run is not None else None

    def adim(self, veri: dict) -> None:
        self._adim += 1
        satir = {"adim": self._adim, **veri}
        self.adimlar.append(satir)
        if self.run is not None and self._adim % 10 == 0:  # her 10 adımda bir: grafik okunur, kayıt hafif kalır
            self.run.log({"adim": self._adim, **{f"adim/{k}": v for k, v in veri.items()}})

    def epoch(self, epoch: int, veri: dict, onek: str = "") -> None:
        satir = {"onek": onek, "epoch": epoch, **veri}
        self.epochlar.append(satir)
        if self.run is not None:
            eksen = f"{onek}epoch"
            if eksen not in self._eksenler:
                self.run.define_metric(eksen)
                self.run.define_metric(f"{onek}epoch/*", step_metric=eksen)
                self._eksenler.add(eksen)
            self.run.log({eksen: epoch, **{f"{onek}epoch/{k}": v for k, v in veri.items()}})

    def ozet(self, veri: dict) -> None:
        if self.run is not None:
            self.run.summary.update(veri)

    def karisiklik(self, y, yp, siniflar) -> None:
        if self.run is not None:
            import wandb
            self.run.log({"test/karisiklik_matrisi": wandb.plot.confusion_matrix(
                y_true=list(map(int, y)), preds=list(map(int, yp)), class_names=list(siniflar))})

    def bitir(self) -> None:
        for ad, satirlar in (("egitim_egrisi_adim.csv", self.adimlar), ("egitim_egrisi_epoch.csv", self.epochlar)):
            if satirlar:
                alanlar = list(dict.fromkeys(k for s in satirlar for k in s))
                with open(self.klasor / ad, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=alanlar)
                    w.writeheader()
                    w.writerows(satirlar)
        if self.run is not None:
            self.run.finish()
            self.run = None
