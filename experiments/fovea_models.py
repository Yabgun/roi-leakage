"""Adım 2 (çözüm): şifreli çalışabilen modeller, şifresiz eğitim ve adil karşılaştırma.

Model sınıfları:
- D (`foveahe.he_models`): Π_ROI sınıfı, iki doğrusal katman (gizli 20). Aktivasyon olmadığı için ≤ 20 sınıfta çok
  sınıflı doğrusal sınıflandırıcıya (lojistik regresyon) denktir; gizli katman ifade gücü eklemez.
- D2: arada kare aktivasyon.
- C (`foveahe.he_cnn`): katman başına tek evrişim (adım = çekirdek) + BN + kare + tam bağlantılı; geometri kullanılmaz.
Aynı sınıf hem odaklı temsilde hem tam görüntüde (orijinal çözünürlük: beyin 512, COVID-QU-Ex 256 px) eğitilir. Seçici
şifreleme model çıktısını değiştirmediği için tam görüntü satırı Π_ROI'nin doğruluğudur. Temsiller: tam görüntü (orijinal
U512/U256 ve 224 px), eş örnekli küçültmeler (U), odaklı yapılandırmalar, yalnız odak, bilgisiz girdi (sabit; C'de yok).
Bölmeler: beyinde test katı k, doğrulama katı k+1 (hasta bazlı), eğitim kalan üç kat; COVID-QU-Ex'te resmi
Train / Val / Test. Beyinde AUC havuzlanmış kat dışı tahminlerden; kat ortalaması `auc_kat_ort` ayrıca (havuzlanmış AUC
bilgisiz girdide 0.47–0.48'e sapabilir, kat ortalaması 0.500).

Adil ayar (tam görüntü dahil her temsil kendi en iyi ayarını alır): AdamW, doğrulama kaybıyla erken durdurma (en çok
MAX_EPOCHS, sabır PATIENCE). Ağırlık azaltma WD_GRID içinden doğrulama kaybıyla seçilir; en iyi değer üst sınırdaysa
10'ar kat WD_MAX'a kadar genişler. WD_MAX'a dayanırsa öğrenme oranı 10 kat düşürülüp {WD_MAX, 10·WD_MAX} de denenir
(ağırlık azaltma × öğrenme oranı ≤ 0.1; AdamW'de bu çarpım 1'e yaklaşınca güncelleme bozulur). Sınıra dayanan seçimler
`sinirda` sütununda işaretlenir.
Sürümler: v1 (CPU, ağırlık azaltma ≤ 1, 60 epoch) `cozum_modeller_v1_cpu.csv`; v2 (GPU, ≤ 100, KUYRUK3 ile GPU paylaşımı
çok yavaşlattığı için durduruldu) kısmi, `cozum_modeller_v2_gpu.csv`; v3 bu sürüm (Model C ve düşük öğrenme oranı).
Veri GPU belleğine uint8 olarak yüklenir (beyin 512 px 0.8 GB, COVID-QU-Ex 256 px 2.2 GB).

Ölçüt (COZUM_PLANI §6): aynı model sınıfında odaklı temsil ile tam görüntü farkı ≤ 0.03 AUC. Ağırlıklar katlanmış numpy
olarak results/checkpoints/.

Çalıştırma: .venv\\Scripts\\python -m experiments.fovea_models [--dataset brain covidqu] [--models D D2 C]
            [--configs ...] [--seeds 0] [--device cuda] [--threads 6] [--quick]
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
from experiments.fovea_info import fold_mean_auc
from foveahe.data import DISPLAY, Dataset, load_dataset, load_layers
from foveahe.he_cnn import ModelC, export_c, forward_numpy_c, intermediate_max, layer_plan
from foveahe.he_models import STD_MIN, export, forward_numpy, make_model, save_weights
from foveahe.representation import FoveaSpec

LOG = config.LOGS / "fovea_models.log"
PREDS = config.RESULTS / "preds"
ORIGINAL = {"brain": 512, "covidqu": 256}
MAX_LOSS = 0.03
LR = 1e-3
WD_GRID = (1e-4, 1e-2, 1.0)
WD_MAX = 100.0
MAX_EPOCHS, PATIENCE = 200, 10


def log(msg: str):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def default_configs(name: str) -> list[str]:
    return [f"U{ORIGINAL[name]}", "tam", "U90", "U64", "F64_G32", "F64_P32k2_G32", "F32_G16", "F64", "sabit"]


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
    """Temsil vektörü: katman pikselleri uint8 (N, p) + geometri (N, 3); yığın başına float32'ye çevrilir.

    device "cpu" ise diskten bellek eşlemeli okunur; aksi hâlde tamamı parça parça GPU belleğine yüklenir.
    """

    def __init__(self, ds: Dataset, spec: FoveaSpec, device: str = "cpu"):
        layers, geom = load_layers(ds, spec.layer_keys, mmap=True, log=log)
        n = len(ds.y)
        parts = [layers[k].reshape(n, -1) for k in spec.layer_keys] or [np.zeros((n, 1), dtype=np.uint8)]
        self.device = device
        self.n_pix = sum(p.shape[1] for p in parts)
        if device == "cpu":
            self.pix = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)
            self.geom = geom if spec.uses_geometry else None
        else:
            self.pix = torch.empty((n, self.n_pix), dtype=torch.uint8, device=device)
            col = 0
            for p in parts:
                rows = max(1, 64_000_000 // p.shape[1])  # parça başına ~64 MB RAM
                for s in range(0, n, rows):
                    self.pix[s:s + rows, col:col + p.shape[1]] = torch.from_numpy(np.ascontiguousarray(p[s:s + rows])).to(device)
                col += p.shape[1]
            self.geom = torch.from_numpy(np.ascontiguousarray(geom)).to(device) if spec.uses_geometry else None
        self.n_features = self.n_pix + (3 if self.geom is not None else 0)

    def get(self, idx) -> torch.Tensor:
        idx = np.asarray(idx)
        if self.device == "cpu":
            x = torch.from_numpy(self.pix[idx]).float().div_(255)
            if self.geom is not None:
                x = torch.cat([x, torch.from_numpy(self.geom[idx])], dim=1)
            return x
        it = torch.as_tensor(idx, device=self.device)
        x = self.pix.index_select(0, it).float().div_(255)
        if self.geom is not None:
            x = torch.cat([x, self.geom.index_select(0, it)], dim=1)
        return x

    def moments(self, idx, chunk: int = 128):
        """Özellik başına ortalama ve std (std alttan STD_MIN ile sınırlı)."""
        s = torch.zeros(self.n_features, dtype=torch.float64)
        s2 = torch.zeros(self.n_features, dtype=torch.float64)
        for i in range(0, len(idx), chunk):
            x = self.get(idx[i:i + chunk])
            s += x.sum(0).double().cpu()
            s2 += (x * x).sum(0).double().cpu()
        mean = s / len(idx)
        std = torch.sqrt(torch.clamp(s2 / len(idx) - mean ** 2, min=0.0))
        return mean.float(), torch.clamp(std, min=STD_MIN).float()

    def layer_stats(self, idx, plan, chunk: int = 256):
        """Model C: katman başına tek ölçek (ortalama, std)."""
        offs = np.cumsum([0] + [s * s for _, s, _ in plan])
        s1, s2, n = np.zeros(len(plan)), np.zeros(len(plan)), np.zeros(len(plan))
        for i in range(0, len(idx), chunk):
            x = self.get(idx[i:i + chunk])
            for j in range(len(plan)):
                part = x[:, offs[j]:offs[j + 1]]
                s1[j] += float(part.sum(dtype=torch.float64))
                s2[j] += float((part * part).sum(dtype=torch.float64))
                n[j] += part.numel()
        mean = s1 / n
        return [(float(m), float(max(np.sqrt(max(v - m * m, 0.0)), STD_MIN))) for m, v in zip(mean, s2 / n)]


def mean_loss(model, data, idx, y, device, batch: int = 512) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            total += float(F.cross_entropy(model(data.get(b).to(device)), y[b].to(device), reduction="sum"))
    return total / len(idx)


def fit(build, data, y, tr, va, seed, device, max_epochs, patience, wd, lr, batch=128):
    """AdamW; doğrulama kaybı en düşük epoch'un ağırlıkları döner: (model, epoch, doğrulama kaybı)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build().to(device)
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


def fit_select(kind, spec, data, y, tr, va, n_classes, seed, device, max_epochs, patience):
    """Ağırlık azaltma ve (sınırda) öğrenme oranı seçimi. Döner: (model, epoch, wd, lr)."""
    if kind == "C":
        plan = layer_plan(spec)
        stats = data.layer_stats(tr, plan)
        build = lambda: ModelC(plan, n_classes, stats)
    else:
        mean, std = data.moments(tr)
        build = lambda: make_model(kind, data.n_features, n_classes, mean, std)
    results = {}

    def run(wd, lr):
        if (wd, lr) not in results:
            results[(wd, lr)] = fit(build, data, y, tr, va, seed, device, max_epochs, patience, wd, lr)

    for wd in WD_GRID:
        run(wd, LR)
    best = min(results, key=lambda key: results[key][2])
    while best[0] == max(w for w, _ in results) and best[0] * 10 <= WD_MAX:
        run(best[0] * 10, LR)
        best = min(results, key=lambda key: results[key][2])
    if best[0] >= WD_MAX:
        for wd in (WD_MAX, WD_MAX * 10):
            run(wd, LR / 10)
        best = min(results, key=lambda key: results[key][2])
    model, epoch, _ = results[best]
    return model, epoch, best[0], best[1]


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


def export_any(kind: str, model) -> dict:
    return export_c(model) if kind == "C" else export(model)


def check_export(kind, model, weights, data, idx, device) -> tuple[float, float]:
    """Katlanmış numpy ağırlıklar PyTorch logitlerini veriyor mu; CKKS için en büyük ara değer büyüklüğü."""
    model.eval()
    x = data.get(idx[:64])
    with torch.no_grad():
        ref = model(x.to(device)).double().cpu().numpy()
    xd = x.double().cpu().numpy()
    if kind == "C":
        return float(np.abs(forward_numpy_c(weights, xd) - ref).max()), intermediate_max(weights, xd)
    return float(np.abs(forward_numpy(weights, xd) - ref).max()), float(np.abs(xd @ weights["W1"].T).max())


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


def fill_fold_auc(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Kat ortalaması AUC'si eksik beyin satırlarını kayıtlı tahminlerden tamamlar."""
    if "auc_kat_ort" not in df:
        df["auc_kat_ort"] = np.nan
    need = (df.veri == DISPLAY["brain"]) & df.auc_kat_ort.isna()
    if need.any():
        ds = load_dataset("brain")
        for i in df.index[need]:
            path = PREDS / f"fovea_model_brain_{df.at[i, 'model']}_{df.at[i, 'temsil']}_s{int(df.at[i, 'tohum'])}{tag}.npy"
            if path.exists():
                df.at[i, "auc_kat_ort"] = fold_mean_auc(ds, np.load(path))
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Veri × model × temsil başına tohum ortalaması; fark aynı model sınıfının tam görüntü (orijinal) satırına göre."""
    rows = []
    for (veri, model), d in df.groupby(["veri", "model"], sort=False):
        ref = d[d.temsil_turu == "tam_orijinal"]
        for name, g in d.groupby("temsil", sort=False):
            seeds = set(g.tohum)
            r = ref[ref.tohum.isin(seeds)]
            loss = float(r.auc.mean() - g.auc.mean()) if len(r) else np.nan
            fold = g.auc_kat_ort if "auc_kat_ort" in g else pd.Series(dtype=float)
            rows.append({"veri": veri, "model": model, "temsil": name, "temsil_turu": g.temsil_turu.iloc[0],
                         "sifreli_deger": int(g.sifreli_deger.iloc[0]), "tohum_sayisi": len(seeds),
                         "auc_ort": float(g.auc.mean()), "auc_std": float(g.auc.std(ddof=1)) if len(g) > 1 else np.nan,
                         "auc_kat_ort": float(fold.mean()) if fold.notna().any() else np.nan,
                         "tam_fark": loss, "wd_secilen": g.wd_secilen.iloc[0],
                         "lr_secilen": g.lr_secilen.iloc[0] if "lr_secilen" in g else "",
                         "epoch_ort": float(g.epoch_ort.mean()), "sinirda": g.sinirda.iloc[0] if "sinirda" in g else "",
                         "olcut": "-" if g.temsil_turu.iloc[0] in ("tam_orijinal", "sabit") or np.isnan(loss)
                         else ("evet" if loss <= MAX_LOSS else "hayır")})
    return pd.DataFrame(rows).sort_values(["veri", "model", "sifreli_deger"], kind="stable").reset_index(drop=True)


def read_table(path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def save_row(row: dict, out_csv, tag: str):
    if out_csv.exists():
        df = read_table(out_csv)
        same = ((df.veri == row["veri"]) & (df.model == row["model"]) & (df.temsil == row["temsil"])
                & (df.tohum == row["tohum"]))
        df = pd.concat([df[~same], pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df = fill_fold_auc(df, tag)
    df.to_csv(out_csv, index=False)
    write_markdown_table(summarize(df), out_csv.with_suffix(".md"), floatfmt="{:.4f}")


def run_dataset(name, models, configs, seeds, device, quick, out_csv, tag):
    ds = load_dataset(name)
    n_cls = len(ds.labels)
    done = set()
    if out_csv.exists():
        prev = read_table(out_csv)
        done = set(zip(prev.veri, prev.model, prev.temsil, prev.tohum.astype(int)))
    max_epochs, patience = (3, 2) if quick else (MAX_EPOCHS, PATIENCE)
    for cfg in configs:
        spec = FoveaSpec.parse(cfg)
        todo = [(m, s) for m in models for s in seeds
                if (ds.display, m, cfg, s) not in done and not (m == "C" and not spec.layer_keys)]
        if not todo:
            continue
        t_load = time.perf_counter()
        data = VectorData(ds, spec, device)
        log(f"[{name}] {cfg}: {data.n_features} öznitelik {device} belleğine yüklendi ({time.perf_counter() - t_load:.0f} s)")
        for kind, seed in todo:
            t0 = time.perf_counter()
            probs = np.full((len(ds.y), n_cls), np.nan)
            epochs, wds, lrs, export_err, max_mid = [], [], [], 0.0, 0.0
            for split_name, tr, va, te in splits(ds, quick):
                model, ep, wd, lr = fit_select(kind, spec, data, ds.y, tr, va, n_cls, seed, device, max_epochs,
                                               patience)
                probs[te] = predict(model, data, te, device, seed, n_cls)
                weights = export_any(kind, model)
                err, mid = check_export(kind, model, weights, data, te, device)
                export_err, max_mid = max(export_err, err), max(max_mid, mid)
                save_weights(weights, config.CHECKPOINTS / f"fovea_{name}_{kind}_{cfg}_s{seed}_{split_name}{tag}.npz")
                epochs.append(ep)
                wds.append(wd)
                lrs.append(lr)
            te_all = np.flatnonzero(~np.isnan(probs[:, 0]))
            p, yt = probs[te_all], ds.y[te_all]
            groups = ds.groups[te_all] if ds.groups is not None else None
            auc = auc_score(yt, p)
            lo, hi = bootstrap_ci(yt, p, groups=groups, n_boot=500)
            edge = []
            if max(epochs) >= max_epochs:
                edge.append("epoch")
            if max(wds) >= WD_MAX * 10:
                edge.append("wd")
            n_values = sum(s * s for _, s, _ in layer_plan(spec)) if kind == "C" else \
                (data.n_features if cfg != "sabit" else 0)
            row = {"veri": ds.display, "model": kind, "temsil": cfg, "temsil_turu": kind_of(cfg, name),
                   "sifreli_deger": n_values, "tohum": seed, "auc": auc, "ci95_alt": lo, "ci95_ust": hi,
                   "n": len(te_all), "epoch_ort": float(np.mean(epochs)), "wd_secilen": "/".join(f"{w:g}" for w in wds),
                   "lr_secilen": "/".join(f"{v:g}" for v in lrs), "sinirda": ",".join(edge),
                   "aktarim_hatasi": export_err, "en_buyuk_ara_deger": max_mid, "sure_s": time.perf_counter() - t0,
                   "auc_kat_ort": fold_mean_auc(ds, probs) if name == "brain" else np.nan}
            np.save(PREDS / f"fovea_model_{name}_{kind}_{cfg}_s{seed}{tag}.npy", probs)
            save_row(row, out_csv, tag)
            log(f"[{name}{tag}] {kind:2s} {cfg:14s} tohum={seed} değer={n_values} AUC={auc:.4f} "
                f"(%95 GA {lo:.4f}-{hi:.4f}) kat ort={row['auc_kat_ort']:.4f} epoch={row['epoch_ort']:.1f} "
                f"wd={row['wd_secilen']} lr={row['lr_secilen']} sınırda={row['sinirda'] or '-'} "
                f"aktarım hatası={export_err:.1e} en büyük ara değer={max_mid:.1f} {row['sure_s']:.0f} s")
        del data
        if device != "cpu":
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="*", default=["brain", "covidqu"])
    ap.add_argument("--models", nargs="*", default=["D", "D2", "C"])
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
        print(summarize(fill_fold_auc(read_table(out_csv), tag)).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
