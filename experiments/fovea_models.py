"""Adım 2 (çözüm): şifreli çalışabilen modeller, şifresiz eğitim ve adil karşılaştırma.

Model sınıfları (`foveahe.he_models`): D (Π_ROI sınıfı: iki doğrusal katman, gizli 20) ve D2 (arada kare aktivasyon).
Aynı sınıf hem odaklı temsil vektöründe hem tam görüntüde (orijinal çözünürlük: beyin 512, COVID-QU-Ex 256 px)
eğitilir. Seçici şifreleme model çıktısını değiştirmediği için tam görüntü satırı Π_ROI'nin doğruluğudur.
Temsiller: tam görüntü (orijinal U512/U256 ve 224 px), eş örnekli küçültmeler (U), odaklı yapılandırmalar, yalnız odak,
bilgisiz girdi (sabit, AUC 0.5). Bölmeler: beyinde test katı k, doğrulama katı k+1 (hasta bazlı), eğitim kalan üç kat;
COVID-QU-Ex'te resmi Train / Val / Test. Her temsil ve katta ağırlık azaltma WD_GRID içinden doğrulama kaybıyla seçilir
ve doğrulama kaybıyla erken durdurulur (tam görüntü dahil her temsil kendi en iyi ayarını alır). Ölçüt
(COZUM_PLANI §6): aynı model sınıfında odaklı temsil ile tam görüntü farkı ≤ 0.03 AUC. Ağırlıklar katlanmış numpy
olarak results/checkpoints/.

Çalıştırma: .venv\\Scripts\\python -m experiments.fovea_models [--dataset brain covidqu] [--models D D2]
            [--configs ...] [--seeds 0] [--device cpu] [--threads 6] [--quick]
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import config
from common.evaluation import auc_score, bootstrap_ci
from common.report import write_markdown_table
from foveahe.data import Dataset, load_dataset, load_layers
from foveahe.he_models import STD_MIN, export, forward_numpy, make_model, save_weights
from foveahe.representation import FoveaSpec

LOG = config.LOGS / "fovea_models.log"
ORIGINAL = {"brain": 512, "covidqu": 256}
MAX_LOSS = 0.03
WD_GRID = (1e-4, 1e-2, 1.0)


def log(msg: str):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def default_configs(name: str) -> list[str]:
    return [f"U{ORIGINAL[name]}", "tam", "U90", "U64", "F64_G32", "F64_P32k2_G32", "F64", "sabit"]


def kind_of(name: str, dataset: str) -> str:
    spec = FoveaSpec.parse(name)
    if name == "sabit":
        return "sabit"
    if spec.focus or spec.periphery:
        return "odakli"
    if spec.glob == ORIGINAL[dataset]:
        return "tam_orijinal"
    return "tam_224" if name == "tam" else "es_ornekli"


class VectorData:
    """Temsil vektörü: katman pikselleri uint8 (N, p) + geometri float32 (N, 3); yığın başına float'a çevrilir."""

    def __init__(self, ds: Dataset, spec: FoveaSpec):
        # Bellek eşlemeli okuma: tam görüntü vektörleri (beyin 512 px 0.8 GB, COVID-QU-Ex 256 px 2.2 GB) RAM'e sığmayabilir
        layers, geom = load_layers(ds, spec.layer_keys, mmap=True, log=log)
        n = len(ds.y)
        parts = [layers[k].reshape(n, -1) for k in spec.layer_keys]
        if not parts:
            parts = [np.zeros((n, 1), dtype=np.uint8)]  # sabit: tek sabit özellik
        self.pix = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)
        self.geom = geom if spec.uses_geometry else None
        self.n_features = self.pix.shape[1] + (3 if self.geom is not None else 0)

    def get(self, idx) -> torch.Tensor:
        idx = np.asarray(idx)
        x = torch.from_numpy(self.pix[idx]).float().div_(255)
        if self.geom is not None:
            x = torch.cat([x, torch.from_numpy(self.geom[idx])], dim=1)
        return x

    def moments(self, idx, chunk: int = 128):
        s = np.zeros(self.n_features)
        s2 = np.zeros(self.n_features)
        for i in range(0, len(idx), chunk):
            x = self.get(idx[i:i + chunk])
            s += x.sum(0).double().numpy()
            s2 += (x * x).sum(0).double().numpy()
        mean = s / len(idx)
        std = np.sqrt(np.maximum(s2 / len(idx) - mean ** 2, 0.0))
        return torch.tensor(mean, dtype=torch.float32), torch.tensor(np.maximum(std, STD_MIN), dtype=torch.float32)


def mean_loss(model, data, idx, y, device, batch: int = 512) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            total += float(F.cross_entropy(model(data.get(b).to(device)), y[b].to(device), reduction="sum"))
    return total / len(idx)


def fit(kind, data, y, tr, va, n_classes, seed, device, max_epochs, patience, mean, std, wd, batch=128, lr=1e-3):
    """AdamW ile eğitim; doğrulama kaybı en düşük epoch'un ağırlıkları döner: (model, epoch, doğrulama kaybı)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model(kind, data.n_features, n_classes, mean, std).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    yt = torch.from_numpy(y.astype(np.int64))
    best_loss, best_state, best_epoch, bad = np.inf, None, 0, 0
    for ep in range(max_epochs):
        model.train()
        perm = rng.permutation(tr)
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            if len(b) < 2:
                continue  # BN tek örnekle çalışmaz
            loss = F.cross_entropy(model(data.get(b).to(device)), yt[b].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        val = mean_loss(model, data, va, yt, device)
        if val < best_loss - 1e-4:
            best_loss, best_state, best_epoch, bad = val, copy.deepcopy(model.state_dict()), ep + 1, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_epoch, best_loss


@torch.no_grad()
def predict(model, data, idx, device, seed, n_classes, batch: int = 512):
    """fp64 softmax, karışık sırada (değerlendirme hijyeni)."""
    model.eval()
    order = np.random.default_rng(seed + 12345).permutation(len(idx))
    out = np.zeros((len(idx), n_classes))
    for i in range(0, len(idx), batch):
        part = order[i:i + batch]
        out[part] = torch.softmax(model(data.get(idx[part]).to(device)).double(), 1).cpu().numpy()
    return out


def check_export(model, weights, data, idx, device) -> tuple[float, float]:
    """Katlanmış numpy ağırlıklar PyTorch logitlerini veriyor mu; CKKS için en büyük |W1 x| (yanlılık eklenmeden)."""
    model.eval()
    x = data.get(idx[:64])
    with torch.no_grad():
        ref = model(x.to(device)).double().cpu().numpy()
    got = forward_numpy(weights, x.double().numpy())
    dot = np.abs(x.double().numpy() @ weights["W1"].T).max()
    return float(np.abs(got - ref).max()), float(dot)


def splits(ds: Dataset, quick: bool):
    """(ad, eğitim, doğrulama, test) dörtlüleri."""
    if ds.name == "brain":
        folds = np.unique(ds.folds)
        out = []
        for i, k in enumerate(folds[:1] if quick else folds):
            v = folds[(i + 1) % len(folds)]
            out.append((f"k{k}", np.flatnonzero(~np.isin(ds.folds, [k, v])), np.flatnonzero(ds.folds == v),
                        np.flatnonzero(ds.folds == k)))
        return out
    tr, va, te = (np.flatnonzero(ds.split == s) for s in ("Train", "Val", "Test"))
    if quick:
        rng = np.random.default_rng(0)
        tr, va, te = (rng.choice(a, m, replace=False) for a, m in ((tr, 3000), (va, 500), (te, 1000)))
    return [("test", tr, va, te)]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Veri × model × temsil başına tohum ortalaması; fark aynı model sınıfının tam görüntü (orijinal) satırına göre."""
    rows = []
    for (veri, model), d in df.groupby(["veri", "model"], sort=False):
        ref = d[d.temsil_turu == "tam_orijinal"]
        for name, g in d.groupby("temsil", sort=False):
            seeds = set(g.tohum)
            r = ref[ref.tohum.isin(seeds)]
            loss = float(r.auc.mean() - g.auc.mean()) if len(r) else np.nan
            rows.append({"veri": veri, "model": model, "temsil": name, "temsil_turu": g.temsil_turu.iloc[0],
                         "sifreli_deger": int(g.sifreli_deger.iloc[0]), "tohum_sayisi": len(seeds),
                         "auc_ort": float(g.auc.mean()), "auc_std": float(g.auc.std(ddof=1)) if len(g) > 1 else np.nan,
                         "tam_fark": loss,
                         "olcut": "-" if g.temsil_turu.iloc[0] in ("tam_orijinal", "sabit") or np.isnan(loss)
                         else ("evet" if loss <= MAX_LOSS else "hayır")})
    return pd.DataFrame(rows).sort_values(["veri", "model", "sifreli_deger"], kind="stable").reset_index(drop=True)


def save_row(row: dict, out_csv):
    if out_csv.exists():
        df = pd.read_csv(out_csv)
        same = ((df.veri == row["veri"]) & (df.model == row["model"]) & (df.temsil == row["temsil"])
                & (df.tohum == row["tohum"]))
        df = pd.concat([df[~same], pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(out_csv, index=False)
    write_markdown_table(summarize(df), out_csv.with_suffix(".md"), floatfmt="{:.4f}")


def run_dataset(name, models, configs, seeds, device, quick, out_csv, tag):
    ds = load_dataset(name)
    n_cls = len(ds.labels)
    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = set(zip(prev.veri, prev.model, prev.temsil, prev.tohum.astype(int)))
    max_epochs, patience = (3, 2) if quick else (60, 8)
    for cfg in configs:
        spec = FoveaSpec.parse(cfg)
        todo = [(m, s) for m in models for s in seeds if (ds.display, m, cfg, s) not in done]
        if not todo:
            continue
        data = VectorData(ds, spec)
        for kind, seed in todo:
            t0 = time.perf_counter()
            probs = np.full((len(ds.y), n_cls), np.nan)
            epochs, wds, export_err, max_dot = [], [], 0.0, 0.0
            for split_name, tr, va, te in splits(ds, quick):
                mean, std = data.moments(tr)
                best = None
                for wd in WD_GRID:
                    model, ep, val = fit(kind, data, ds.y, tr, va, n_cls, seed, device, max_epochs, patience, mean,
                                         std, wd)
                    if best is None or val < best[2]:
                        best = (model, ep, val, wd)
                model, ep, _, wd = best
                probs[te] = predict(model, data, te, device, seed, n_cls)
                weights = export(model)
                err, dot = check_export(model, weights, data, te, device)
                export_err, max_dot = max(export_err, err), max(max_dot, dot)
                save_weights(weights, config.CHECKPOINTS / f"fovea_{name}_{kind}_{cfg}_s{seed}_{split_name}{tag}.npz")
                epochs.append(ep)
                wds.append(wd)
            te_all = np.flatnonzero(~np.isnan(probs[:, 0]))
            p, yt = probs[te_all], ds.y[te_all]
            groups = ds.groups[te_all] if ds.groups is not None else None
            auc = auc_score(yt, p)
            lo, hi = bootstrap_ci(yt, p, groups=groups, n_boot=500)
            row = {"veri": ds.display, "model": kind, "temsil": cfg, "temsil_turu": kind_of(cfg, name),
                   "sifreli_deger": data.n_features if cfg != "sabit" else 0, "tohum": seed, "auc": auc,
                   "ci95_alt": lo, "ci95_ust": hi, "n": len(te_all), "epoch_ort": float(np.mean(epochs)),
                   "wd_secilen": "/".join(f"{w:g}" for w in wds), "aktarim_hatasi": export_err,
                   "en_buyuk_nokta_carpim": max_dot, "sure_s": time.perf_counter() - t0}
            np.save(config.RESULTS / "preds" / f"fovea_model_{name}_{kind}_{cfg}_s{seed}{tag}.npy", probs)
            save_row(row, out_csv)
            log(f"[{name}] {kind:2s} {cfg:14s} tohum={seed} değer={row['sifreli_deger']} AUC={auc:.4f} "
                f"(%95 GA {lo:.4f}-{hi:.4f}) epoch={row['epoch_ort']:.1f} wd={row['wd_secilen']} "
                f"aktarım hatası={export_err:.1e} en büyük |W1x|={max_dot:.1f} {row['sure_s']:.0f} s")
        del data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="*", default=["brain", "covidqu"])
    ap.add_argument("--models", nargs="*", default=["D", "D2"])
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    tag = "_hizli" if args.quick else ""
    out_csv = config.TABLES / f"cozum_modeller{tag}.csv"
    for name in args.dataset:
        run_dataset(name, args.models, args.configs or default_configs(name), args.seeds, args.device, args.quick,
                    out_csv, tag)
    if out_csv.exists():
        print(summarize(pd.read_csv(out_csv)).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
