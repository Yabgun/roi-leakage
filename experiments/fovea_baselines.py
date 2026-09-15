"""Adım 5 (çözüm, rakip): şifreli özet (HETAL tarzı).

İstemci herkese açık, dondurulmuş ImageNet ResNet-18 (torchvision IMAGENET1K_V1) ile 224 px görüntünün 512 boyutlu
özetini çıkarır ve yalnızca özeti CKKS ile şifreler; sunucu Adım 2'nin D ve D2 sınıflarını özet üzerinde şifreli
çalıştırır. Açıkta piksel ve değişken boyutlu meta veri yoktur (FoveaHE gibi); fark, istemcide büyük bir ağın (11.2 M
parametre) çalışması ve özetin göreve göre eğitilmemiş olmasıdır (istemcide göreve eğitilmiş ağ olsaydı sunucuya gerek
kalmazdı). Aynı bölmeler, aynı ayar protokolü (`fovea_models.fit_select`), aynı şifreli çıkarım kodu
(`FoveaHEInference`) ve aynı CKKS parametreleri.
Ölçülenler: teşhis AUC'si, şifreli-şifresiz eşleşme, maliyet dökümü (makine boşken), istemci CPU süresi (PNG okuma +
küçültme + ResNet-18 ileri geçişi; tüm çekirdekler ve tek çekirdek).
Çıktılar: data/processed/fovea/<veri>/emb_r18.npy, results/tables/cozum_rakipler.csv|md.

Çalıştırma: .venv\\Scripts\\python -m experiments.fovea_baselines [--dataset brain covidqu] [--models D D2]
            [--reps 3] [--n-match 60] [--skip-he] [--quick]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
from torchvision.models import ResNet18_Weights, resnet18

import config
from common.evaluation import auc_score, bootstrap_ci
from common.images import load_gray
from common.report import write_markdown_table
from experiments.attack_context import build_cache
from experiments.fovea_cost import softmax, stratified, test_indices
from experiments.fovea_models import MAX_EPOCHS, PATIENCE, check_export, export_any, fit_select, predict, splits
from foveahe.data import STORE, Dataset, load_dataset
from foveahe.he_infer import FoveaHEInference, public_context_bytes
from foveahe.he_models import STD_MIN, forward_numpy, save_weights
from he.piroi import make_context

LOG = config.LOGS / "fovea_baselines.log"
METHOD = "şifreli özet (ImageNet ResNet-18, 512)"
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def log(msg: str):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def encoder(device: str) -> torch.nn.Module:
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    return net.eval().to(device)


@torch.no_grad()
def embeddings(ds: Dataset, device: str = "cuda", batch: int = 256) -> np.ndarray:
    """(N, 512) float32 özet; 224 px önbellekten, fp32, sonuç diske yazılır."""
    path = STORE / ds.name / "emb_r18.npy"
    if path.exists():
        return np.load(path)
    t0 = time.perf_counter()
    imgs, _ = build_cache(ds.name, ds.img_paths, ds.mask_paths)
    net = encoder(device)
    out = np.zeros((len(imgs), 512), dtype=np.float32)
    for i in range(0, len(imgs), batch):
        x = torch.from_numpy(imgs[i:i + batch]).to(device).float().div_(255).unsqueeze(1).expand(-1, 3, -1, -1)
        out[i:i + batch] = net((x - _MEAN.to(device)) / _STD.to(device)).float().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    log(f"[{ds.name}] özetler çıkarıldı: {out.shape} ({time.perf_counter() - t0:.0f} s)")
    return out


class ArrayData:
    """`fovea_models.VectorData` arayüzü (get, moments, n_features, device) float özellik matrisi için."""

    def __init__(self, X: np.ndarray, device: str):
        self.device = device
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(device)
        self.n_features = X.shape[1]

    def get(self, idx) -> torch.Tensor:
        return self.X.index_select(0, torch.as_tensor(np.asarray(idx), device=self.device))

    def moments(self, idx, chunk: int = 4096):
        x = self.get(idx).double()
        mean = x.mean(0)
        std = torch.sqrt(torch.clamp((x * x).mean(0) - mean ** 2, min=0.0))
        return mean.float().cpu(), torch.clamp(std, min=STD_MIN).float().cpu()


def client_cpu_time(ds: Dataset, n: int = 20) -> tuple[float, float]:
    """İstemci: PNG okuma + 224 px'e küçültme + ResNet-18 ileri geçişi (CPU, tek görüntü); medyan (tüm, tek çekirdek)."""
    net = encoder("cpu")
    idx = np.random.default_rng(1).choice(len(ds.y), n + 1, replace=False)
    threads = torch.get_num_threads()
    result = []
    with torch.no_grad():
        for n_threads in (threads, 1):
            torch.set_num_threads(n_threads)
            times = []
            for j, i in enumerate(idx):
                t0 = time.perf_counter()
                x = torch.from_numpy(load_gray(ds.img_paths[i], 224))[None, None].expand(1, 3, -1, -1)
                net((x - _MEAN) / _STD)
                if j:  # ilk çağrı ısınma
                    times.append(time.perf_counter() - t0)
            result.append(float(np.median(times)))
    torch.set_num_threads(threads)
    return result[0], result[1]


def he_measure(ctx, ds: Dataset, weights: dict, data: ArrayData, n_match: int, reps: int) -> dict:
    infer = FoveaHEInference(ctx, weights)
    idx = stratified(test_indices(ds), ds.y, n_match, seed=0)
    x = data.get(idx).double().cpu().numpy()
    plain = forward_numpy(weights, x)
    enc = np.array([infer.run(xi, measure_bytes=False)[0] for xi in x])
    y = ds.y[idx]
    out = {"maks_mutlak_logit_hatasi": float(np.abs(enc - plain).max()),
           "argmax_uyumu": float((enc.argmax(1) == plain.argmax(1)).mean()),
           "auc_farki_eslesme": abs(auc_score(y, softmax(enc)) - auc_score(y, softmax(plain)))}
    costs = []
    for i in np.random.default_rng(2).choice(test_indices(ds), reps, replace=False):
        _, c = infer.run(data.get([i]).double().cpu().numpy()[0], measure_bytes=True)
        costs.append(c)
    out.update({"ciphertext": costs[0].n_ciphertexts, "sifreleme_s": float(np.mean([c.enc_s for c in costs])),
                "sunucu_s": float(np.mean([c.server_s for c in costs])),
                "cozme_s": float(np.mean([c.dec_s for c in costs])),
                "toplam_s": float(np.mean([c.total_s for c in costs])),
                "yukleme_MB": costs[0].upload_bytes / 1e6, "indirme_MB": costs[0].extra["indirme_bayt"] / 1e6})
    return out


def run_dataset(ctx, name: str, models, seed: int, n_match: int, reps: int, skip_he: bool, quick: bool, rows: list,
                tag: str):
    ds = load_dataset(name)
    data = ArrayData(embeddings(ds), "cuda")
    n_cls = len(ds.labels)
    max_epochs, patience = (3, 2) if quick else (MAX_EPOCHS, PATIENCE)
    cpu_all, cpu_one = client_cpu_time(ds, n=5 if quick else 20)
    log(f"[{name}] istemci CPU (okuma + ResNet-18): tüm çekirdekler {cpu_all * 1000:.0f} ms, tek çekirdek "
        f"{cpu_one * 1000:.0f} ms")
    for kind in models:
        t0 = time.perf_counter()
        probs = np.full((len(ds.y), n_cls), np.nan)
        he_weights, wds = None, []
        for split_name, tr, va, te in splits(ds, quick):
            model, ep, wd, lr = fit_select(kind, None, data, ds.y, tr, va, n_cls, seed, "cuda", max_epochs, patience)
            probs[te] = predict(model, data, te, "cuda", seed, n_cls)
            weights = export_any(kind, model)
            err, _ = check_export(kind, model, weights, data, te, "cuda")
            save_weights(weights, config.CHECKPOINTS / f"ozet_{name}_{kind}_s{seed}_{split_name}{tag}.npz")
            wds.append(f"{wd:g}@{lr:g}")
            if he_weights is None:
                he_weights = weights  # beyinde kat 1, CXR'de resmi test modeli (fovea_cost ile aynı)
        te_all = np.flatnonzero(~np.isnan(probs[:, 0]))
        p, yt = probs[te_all], ds.y[te_all]
        groups = ds.groups[te_all] if ds.groups is not None else None
        auc = auc_score(yt, p)
        lo, hi = bootstrap_ci(yt, p, groups=groups, n_boot=500)
        row = {"veri": ds.display, "yontem": METHOD, "model": kind, "tohum": seed, "auc": auc, "ci95_alt": lo,
               "ci95_ust": hi, "n": len(te_all), "sifreli_deger": 512, "wd_lr": "/".join(wds),
               "aktarim_hatasi": err, "istemci_on_isleme_s": cpu_all, "istemci_on_isleme_tek_cekirdek_s": cpu_one,
               "egitim_s": time.perf_counter() - t0}
        if not skip_he:
            row.update(he_measure(ctx, ds, he_weights, data, n_match, reps))
            row["uctan_uca_s"] = cpu_all + row["toplam_s"]
        np.save(config.RESULTS / "preds" / f"ozet_{name}_{kind}_s{seed}{tag}.npy", probs)
        rows.append(row)
        log(f"[{name}{tag}] {METHOD} {kind}: AUC={auc:.4f} (%95 GA {lo:.4f}-{hi:.4f})"
            + ("" if skip_he else f", şifreli toplam {row['toplam_s']:.2f} s, logit hatası {row['maks_mutlak_logit_hatasi']:.1e}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="*", default=["brain", "covidqu"])
    ap.add_argument("--models", nargs="*", default=["D", "D2"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-match", type=int, default=60)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--skip-he", action="store_true", help="şifreli ölçümü atla (yalnızca doğruluk)")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "CUDA bulunamadı"
    tag = "_hizli" if args.quick else ""
    n_match, reps = (6, 1) if args.quick else (args.n_match, args.reps)
    ctx = None if args.skip_he else make_context()
    rows = []
    for name in args.dataset:
        run_dataset(ctx, name, args.models, args.seed, n_match, reps, args.skip_he, args.quick, rows, tag)
    df = pd.DataFrame(rows)
    out = config.TABLES / f"cozum_rakipler{tag}.csv"
    if out.exists():
        old = pd.read_csv(out)
        keep = ~old.set_index(["veri", "yontem", "model", "tohum"]).index.isin(
            df.set_index(["veri", "yontem", "model", "tohum"]).index)
        df = pd.concat([old[keep], df], ignore_index=True)
    df.to_csv(out, index=False)
    write_markdown_table(df, out.with_suffix(".md"), floatfmt="{:.4f}")
    if ctx is not None:
        log(f"genel bağlam {public_context_bytes(ctx) / 1e6:.1f} MB")
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
