"""Ortam denetimi: Python, paket sürümleri, ekran kartı, disk ve proje yolu (kurulumdan sonra ilk komut).

Çalıştırma: python -m boruhatti.ortam_kontrol
"""
from __future__ import annotations

import importlib.metadata as im
import platform
import shutil
import subprocess
import sys

import config

# Makaledeki sonuçların üretildiği ortam (requirements.txt ile aynı)
BEKLENEN_PAKET = {"torch": "2.14.0", "torchvision": "0.29.0", "tenseal": "0.3.16", "numpy": "2.5.2",
                  "pandas": "3.0.5", "scikit-learn": "1.9.1", "scipy": "1.18.1", "pillow": "12.3.0",
                  "matplotlib": "3.11.2", "h5py": "3.16.0", "wandb": "0.30.0", "kagglehub": "1.0.2",
                  "huggingface_hub": "2.0.0"}
GEREKEN_DISK_GB = 25          # ham veri ~6 GB, işlenmiş veri ve önbellek ~9 GB, modeller ~1 GB, pay
COVIDQU_EN_UZUN_GORELI = 130  # data/raw/covid_qu_ex/extracted/ altındaki en uzun göreli dosya yolu (ölçüldü)


def satir(durum: str, ad: str, bilgi: str) -> None:
    print(f"[{durum:5s}] {ad:24s} {bilgi}")


def main():
    uyari = 0
    print(f"Proje: {config.ROOT}")
    v = sys.version_info
    durum = "TAMAM" if (v.major, v.minor) == (3, 13) else "UYARI"
    uyari += durum != "TAMAM"
    satir(durum, "Python", f"{platform.python_version()} (sonuçlar 3.13.0 ile üretildi)")
    for paket, beklenen in BEKLENEN_PAKET.items():
        try:
            surum = im.version(paket)
        except im.PackageNotFoundError:
            satir("EKSIK", paket, f"kurulu değil (beklenen {beklenen}); README'deki kurulum komutlarını çalıştırın")
            uyari += 1
            continue
        esit = surum.split("+")[0] == beklenen
        uyari += not esit
        satir("TAMAM" if esit else "UYARI", paket, surum + ("" if esit else f" (beklenen {beklenen})"))
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            gb = p.total_memory / 1024 ** 3
            durum = "TAMAM" if gb >= 6 else "UYARI"
            uyari += durum != "TAMAM"
            satir(durum, "Ekran kartı", f"{p.name}, {gb:.1f} GB bellek, CUDA {torch.version.cuda}"
                  + ("" if gb >= 6 else " (6 GB altı: --yigin ile yığın boyutunu düşürün)"))
            try:
                surucu = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                        capture_output=True, text=True, timeout=20).stdout.strip()
                satir("TAMAM", "NVIDIA sürücüsü", surucu or "okunamadı")
            except Exception:
                satir("UYARI", "NVIDIA sürücüsü", "nvidia-smi çalıştırılamadı")
        else:
            uyari += 1
            satir("UYARI", "Ekran kartı", "CUDA bulunamadı: eğitim çok yavaş olur; değerlendirme ve tahmin işlemcide çalışır. "
                  "NVIDIA kartı varsa torch'un CUDA sürümünü kurun (README).")
    except ImportError:
        pass
    bos = shutil.disk_usage(config.ROOT).free / 1024 ** 3
    durum = "TAMAM" if bos >= GEREKEN_DISK_GB else "UYARI"
    uyari += durum != "TAMAM"
    satir(durum, "Boş disk", f"{bos:.0f} GB (gereken ~{GEREKEN_DISK_GB} GB)")
    uzunluk = len(str(config.DATA_RAW.resolve())) + len("/covid_qu_ex/extracted/") + COVIDQU_EN_UZUN_GORELI
    durum = "TAMAM" if uzunluk <= 255 else "UYARI"
    uyari += durum != "TAMAM"
    satir(durum, "Dosya yolu uzunluğu", f"en uzun veri yolu ~{uzunluk} karakter (Windows sınırı 260)"
          + ("" if uzunluk <= 255 else "; projeyi kısa bir klasöre taşıyın, ör. C:\\roi-leakage"))
    try:
        import wandb  # noqa: F401
        from pathlib import Path
        giris = bool(__import__("os").environ.get("WANDB_API_KEY")) or any(
            (Path.home() / n).exists() and "api.wandb.ai" in (Path.home() / n).read_text(errors="ignore")
            for n in ("_netrc", ".netrc"))
        satir("TAMAM" if giris else "BILGI", "wandb", "giriş yapılmış" if giris else
              "giriş yok (isteğe bağlı; eğitim eğrileri yine yerelde kaydedilir)")
    except ImportError:
        pass
    print("\nSonuç:", "her şey hazır." if uyari == 0 else f"{uyari} uyarı var; yukarıdaki satırlara bakın.")


if __name__ == "__main__":
    main()
