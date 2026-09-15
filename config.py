"""Proje genelinde kullanılan yollar ve sabitler. Deneyler proje kökünden `python -m experiments.<ad>` ile çalışır."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
KAGGLE_CXR = DATA_RAW / "kaggle_cxr"
COVIDQU = DATA_RAW / "covid_qu_ex"
BRAIN = DATA_RAW / "brain_tumor_figshare"
STROKE_CSV = DATA_RAW / "stroke" / "healthcare-dataset-stroke-data.csv"

RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
LOGS = RESULTS / "logs"
CHECKPOINTS = RESULTS / "checkpoints"

SEEDS = [0, 1, 2, 3, 4]

# Π_ROI (Bakas & Schoinianakis, ePrint 2026/103) ile aynı CKKS parametreleri
PIROI_POLY_MODULUS = 16384
PIROI_COEFF_MOD_BITS = [60, 40, 40, 40, 60]
PIROI_SCALE_BITS = 40

for _d in (DATA_PROC, TABLES, FIGURES, LOGS, CHECKPOINTS):
    _d.mkdir(parents=True, exist_ok=True)
