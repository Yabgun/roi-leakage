"""Tez şekilleri için ortak biçim: Türkçe veri adları, sayı yazımı ve PNG + PDF (vektör) kayıt.

Sayı yazımı tezle aynıdır: ondalık ayracı nokta; dört basamaklı tam sayılarda ayraç yok (8192), beş ve daha fazla
basamakta binlik ayracı boşluk (16 384). Böylece "1.024" gibi hem binlik hem ondalık okunabilen yazım oluşmaz.
"""
from __future__ import annotations

VERI_TEZ = {"beyin MR (Cheng)": "Beyin MR", "akciğer grafisi (COVID-QU-Ex)": "Göğüs röntgeni (COVID-QU-Ex)",
            "brain": "Beyin MR", "covidqu": "Göğüs röntgeni (COVID-QU-Ex)"}
SINIF_TEZ = {"COVID-19": "COVID-19", "Non-COVID": "COVID dışı pnömoni", "Normal": "Normal",
             "meningioma": "Meningiom", "glioma": "Gliom", "pituitary": "Hipofiz tümörü"}


def sayi(n) -> str:
    """Tam sayı: 4 basamağa kadar ayraçsız, 5 ve üstü binlik boşluklu (16 384)."""
    n = int(round(float(n)))
    s = str(abs(n))
    if len(s) >= 5:
        groups = []
        while s:
            groups.append(s[-3:])
            s = s[:-3]
        s = " ".join(reversed(groups))
    return ("-" if n < 0 else "") + s


def kaydet(fig, png_path, dpi: int = 200, **kwargs) -> None:
    """Aynı adla PNG ve PDF kaydeder (tez PDF'i baskıda net olsun diye vektör sürüm)."""
    from pathlib import Path
    png_path = Path(png_path)
    fig.savefig(png_path.with_suffix(".png"), dpi=dpi, **kwargs)
    fig.savefig(png_path.with_suffix(".pdf"), **kwargs)
