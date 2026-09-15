"""Adım 3 (çözüm): gerçekten şifreli çalıştırma, doğruluk eşleşmesi ve maliyet dökümü.

1) Doğruluk eşleşmesi: Adım 2 ağırlıklarıyla (beyinde kat 1 modeli ve kat 1 testi, COVID-QU-Ex'te resmi test) sınıflara
   eşit dağıtılmış rastgele test örneklerinde şifreli ve şifresiz logitler: en büyük mutlak hata, argmax uyumu, AUC
   farkı (§6 ölçütü ≤ 0.005). Modeller: D, D2 (`foveahe.he_infer`), C (`foveahe.he_cnn`).
2) Maliyet (tekrarlı; makine boşken çalıştırılmalı): istemci ön işleme (PNG okuma + temsil), şifreleme, sunucu, çözme,
   iletişim. Model D ayrıca Π_ROI yeniden üretimiyle aynı kodla (`PiROI.run`, maske tümü 1, yanlılıksız) ölçülür.
3) Referanslar aynı betikte, aynı parametrelerle yeniden ölçülür: tam şifreleme (beyin 512, COVID-QU-Ex 256 px; Model D
   ve Model C için ayrı, rastgele ağırlıklarla, çünkü CKKS süresi ağırlık değerinden bağımsızdır) ve gizlilik şartlı
   Π_ROI (savunma deneyi: AUC ≤ 0.8 için CXR'de %98 şifreli, beyinde yalnızca %100). Hız kazancı model ailesi içinde
   hesaplanır: D ve D2 tam şifreleme D referansına, C tam şifreleme C referansına göre.
Çıktılar: results/tables/cozum_dogruluk_eslesme.csv|md, cozum_maliyet_ham.csv, cozum_maliyet.csv|md,
cozum_maliyet_ozet.json.

Çalıştırma: .venv\\Scripts\\python -m experiments.fovea_cost [--dataset brain covidqu] [--configs ...]
            [--models D D2 C] [--n-match 200] [--reps 3] [--skip-match] [--skip-cost] [--quick]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

import config
from common.evaluation import auc_score
from common.images import load_gray, load_mask, square_box_mask
from common.report import write_markdown_table
from experiments.fovea_models import VectorData
from foveahe.data import Dataset, load_dataset
from foveahe.he_cnn import CNNInference, ModelC, export_c, forward_numpy_c, layer_plan
from foveahe.he_infer import FoveaHEInference, public_context_bytes
from foveahe.he_models import forward_numpy, load_weights
from foveahe.representation import FoveaSpec, extract, roi_geometry
from he.piroi import PiROI, make_context, random_weights

LOG = config.LOGS / "fovea_cost.log"
ORIGINAL = {"brain": 512, "covidqu": 256}
PRIVATE_RHO = {"brain": 1.0, "covidqu": 0.98}
# Adım 1 kararı (15 Eyl): ekonomik F32_G16 ve doğru F64_G32; eş bütçeli eş örnekli referans U64
DEFAULT_CONFIGS = {"brain": ["F32_G16", "F64_G32", "U64"], "covidqu": ["F32_G16", "F64_G32", "U64"]}
FAMILY = {"D": "D", "D2": "D", "C": "C"}


def log(msg: str):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def checkpoint(name: str, kind: str, cfg: str, tag: str):
    split = "k1" if name == "brain" else "test"
    return config.CHECKPOINTS / f"fovea_{name}_{kind}_{cfg}_s0_{split}{tag}.npz"


def test_indices(ds: Dataset) -> np.ndarray:
    return np.flatnonzero(ds.folds == 1) if ds.name == "brain" else np.flatnonzero(ds.split == "Test")


def stratified(pool: np.ndarray, y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Her sınıftan eşit sayıda (n / sınıf sayısı) rastgele indeks."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y[pool])
    per = max(1, n // len(classes))
    return np.sort(np.concatenate([rng.choice(pool[y[pool] == c], per, replace=False) for c in classes]))


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def make_inference(ctx, weights: dict):
    return CNNInference(ctx, weights) if str(weights["kind"]) == "C" else FoveaHEInference(ctx, weights)


def plain_logits(weights: dict, x: np.ndarray) -> np.ndarray:
    return forward_numpy_c(weights, x) if str(weights["kind"]) == "C" else forward_numpy(weights, x)


def encrypted_values(kind: str, spec: FoveaSpec) -> int:
    return sum(s * s for _, s, _ in layer_plan(spec)) if kind == "C" else spec.n_values


def accuracy_match(ctx, ds: Dataset, kind: str, cfg: str, n_match: int, tag: str):
    path = checkpoint(ds.name, kind, cfg, tag)
    if not path.exists():
        log(f"[eşleşme] {path.name} yok, atlandı")
        return None
    weights = load_weights(path)
    data = VectorData(ds, FoveaSpec.parse(cfg))
    idx = stratified(test_indices(ds), ds.y, n_match, seed=0)
    x = data.get(idx).double().numpy()
    plain = plain_logits(weights, x)
    infer = make_inference(ctx, weights)
    enc = np.zeros_like(plain)
    t0 = time.perf_counter()
    for i in range(len(idx)):
        enc[i], _ = infer.run(x[i], measure_bytes=False)
    y = ds.y[idx]
    auc_plain, auc_enc = auc_score(y, softmax(plain)), auc_score(y, softmax(enc))
    row = {"veri": ds.display, "model": kind, "temsil": cfg, "n": len(idx),
           "maks_mutlak_logit_hatasi": float(np.abs(enc - plain).max()),
           "ort_mutlak_logit_hatasi": float(np.abs(enc - plain).mean()),
           "argmax_uyumu": float((enc.argmax(1) == plain.argmax(1)).mean()), "auc_sifresiz": auc_plain,
           "auc_sifreli": auc_enc, "auc_farki": abs(auc_enc - auc_plain), "sure_s": time.perf_counter() - t0}
    log(f"[eşleşme] {ds.name} {kind} {cfg}: maks logit hatası {row['maks_mutlak_logit_hatasi']:.2e}, "
        f"argmax uyumu {row['argmax_uyumu']:.3f}, AUC farkı {row['auc_farki']:.2e}")
    return row


def client_times(ds: Dataset, spec: FoveaSpec, n: int = 20) -> tuple[float, float]:
    """Tek görüntü, CPU, medyan: (FoveaHE: okuma + geometri + katmanlar + vektör, tam/Π_ROI: okuma + düzleştirme)."""
    idx = np.random.default_rng(1).choice(len(ds.y), n, replace=False)
    fovea, full = [], []
    with torch.no_grad():
        for i in idx:
            t0 = time.perf_counter()
            img, mask = load_gray(ds.img_paths[i]), load_mask(ds.mask_paths[i])
            read = time.perf_counter() - t0
            t0 = time.perf_counter()
            img.astype(np.float64).ravel()
            full.append(read + time.perf_counter() - t0)
            t0 = time.perf_counter()
            it, mt = torch.from_numpy(img)[None, None], torch.from_numpy(mask)[None]
            g = roi_geometry(mt)
            parts = [extract(it, g, k).flatten() for k in spec.layer_keys]
            if spec.uses_geometry:
                parts.append(g.flatten())
            torch.cat(parts).double().numpy()
            fovea.append(read + time.perf_counter() - t0)
    return float(np.median(fovea)), float(np.median(full))


def cost_row(ds: Dataset, method: str, model: str, rep_name: str, n_values: int, c, client_s: float, rep: int) -> dict:
    return {"veri": ds.display, "yontem": method, "model": model, "temsil": rep_name, "sifreli_deger": n_values,
            "ciphertext": c.n_ciphertexts, "sifreli_slot": c.extra.get("sifreli_slot", np.nan), "tekrar": rep,
            "istemci_on_isleme_s": client_s, "sifreleme_s": c.enc_s, "sunucu_s": c.server_s, "cozme_s": c.dec_s,
            "toplam_s": c.total_s, "uctan_uca_s": client_s + c.total_s, "yukleme_MB": c.upload_bytes / 1e6,
            "indirme_MB": c.extra.get("indirme_bayt", np.nan) / 1e6}


def measure_fovea(ctx, ds: Dataset, kind: str, cfg: str, reps: int, tag: str, rows: list):
    path = checkpoint(ds.name, kind, cfg, tag)
    if not path.exists():
        log(f"[maliyet] {path.name} yok, atlandı")
        return
    weights = load_weights(path)
    spec = FoveaSpec.parse(cfg)
    data = VectorData(ds, spec)
    client_s, _ = client_times(ds, spec)
    infer = make_inference(ctx, weights)
    n_values = encrypted_values(kind, spec)
    for rep, i in enumerate(np.random.default_rng(2).choice(test_indices(ds), reps, replace=False)):
        x = data.get([i]).double().numpy()[0]
        _, c = infer.run(x, measure_bytes=True)
        rows.append(cost_row(ds, f"FoveaHE-{kind}", kind, cfg, n_values, c, client_s, rep))
        if kind == "D":  # Π_ROI yeniden üretimiyle birebir aynı kod yolu: tamamı şifreli maske, yanlılıksız
            _, c = PiROI(ctx, weights["W1"], weights["W2"]).run(x, np.ones(len(x), dtype=bool), measure_bytes=True)
            rows.append(cost_row(ds, "FoveaHE-D (PiROI.run)", kind, cfg, n_values, c, client_s, rep))
        log(f"[maliyet] {ds.name} {kind} {cfg} tekrar={rep}: toplam {rows[-1]['toplam_s']:.2f} s, "
            f"{rows[-1]['ciphertext']} ciphertext")


def measure_references(ctx, ds: Dataset, reps: int, rows: list, quick: bool, models):
    size = 128 if quick else ORIGINAL[ds.name]
    n = size * size
    n_cls = len(ds.labels)
    _, client_s = client_times(ds, FoveaSpec(glob=64), n=5)
    img = load_gray(ds.img_paths[0], size).ravel()
    if {"D", "D2"} & set(models):
        m1, m2 = random_weights(n, seed=size)
        proto = PiROI(ctx, m1, m2[:n_cls])
        refs = [("tam şifreleme", np.ones(n, dtype=bool))]
        if PRIVATE_RHO[ds.name] < 1.0:
            refs.append((f"Π_ROI %{PRIVATE_RHO[ds.name] * 100:.0f} şifreli (gizlilik şartlı)",
                         square_box_mask(size, PRIVATE_RHO[ds.name]).ravel()))
        for label, mask in refs:
            for rep in range(reps):
                _, c = proto.run(img, mask, measure_bytes=True)
                rows.append(cost_row(ds, label, "D", f"{size} px", int(mask.sum()), c, client_s, rep))
                log(f"[maliyet] {ds.name} {label} (D) tekrar={rep}: toplam {c.total_s:.2f} s, {c.n_ciphertexts} ciphertext")
    if "C" in models:
        torch.manual_seed(0)
        plan = layer_plan(FoveaSpec(glob=size))
        infer = CNNInference(ctx, export_c(ModelC(plan, n_cls, [(0.5, 0.25)]).double().eval()))
        for rep in range(reps):
            _, c = infer.run(img, measure_bytes=True)
            rows.append(cost_row(ds, "tam şifreleme", "C", f"{size} px", n, c, client_s, rep))
            log(f"[maliyet] {ds.name} tam şifreleme (C) tekrar={rep}: toplam {c.total_s:.2f} s, {c.n_ciphertexts} ciphertext")


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(aile=df.model.map(FAMILY))
    agg = df.groupby(["veri", "aile", "yontem", "model", "temsil"], sort=False).agg(
        sifreli_deger=("sifreli_deger", "first"), ciphertext=("ciphertext", "first"),
        tekrar=("tekrar", "count"), istemci_on_isleme_s=("istemci_on_isleme_s", "mean"),
        sifreleme_s=("sifreleme_s", "mean"), sunucu_s=("sunucu_s", "mean"), cozme_s=("cozme_s", "mean"),
        toplam_s=("toplam_s", "mean"), toplam_std=("toplam_s", "std"), uctan_uca_s=("uctan_uca_s", "mean"),
        yukleme_MB=("yukleme_MB", "max"), indirme_MB=("indirme_MB", "max")).reset_index()
    for (veri, family), d in agg.groupby(["veri", "aile"]):
        full = d[d.yontem == "tam şifreleme"]
        private = d[d.yontem.str.startswith("Π_ROI")]
        t_full = float(full.toplam_s.iloc[0]) if len(full) else np.nan
        t_private = float(private.toplam_s.iloc[0]) if len(private) else t_full  # beyinde gizlilik şartı = %100
        agg.loc[d.index, "hiz_kazanci_tam"] = t_full / d.toplam_s
        if family == "D":
            agg.loc[d.index, "hiz_kazanci_gizlilik_sartli_piroi"] = t_private / d.toplam_s
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="*", default=["brain", "covidqu"])
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=["D", "D2", "C"])
    ap.add_argument("--n-match", type=int, default=200)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--skip-match", action="store_true")
    ap.add_argument("--skip-cost", action="store_true")
    ap.add_argument("--quick", action="store_true", help="hızlı deneme ağırlıkları, 6 örnek, 1 tekrar, 128 px referans")
    args = ap.parse_args()
    tag = "_hizli" if args.quick else ""
    n_match, reps = (6, 1) if args.quick else (args.n_match, args.reps)
    t0 = time.perf_counter()
    ctx = make_context()
    summary = {"genel_baglam_MB": public_context_bytes(ctx) / 1e6, "N": config.PIROI_POLY_MODULUS,
               "katsayi_modulu_bit": config.PIROI_COEFF_MOD_BITS, "olcek_bit": config.PIROI_SCALE_BITS}
    log(f"CKKS bağlamı hazır ({time.perf_counter() - t0:.1f} s), genel bağlam {summary['genel_baglam_MB']:.1f} MB")
    match_rows, cost_rows = [], []
    for name in args.dataset:
        ds = load_dataset(name)
        configs = args.configs or DEFAULT_CONFIGS[name]
        if not args.skip_match:
            for cfg in configs:
                for kind in args.models:
                    row = accuracy_match(ctx, ds, kind, cfg, n_match, tag)
                    if row:
                        match_rows.append(row)
        if not args.skip_cost:
            measure_references(ctx, ds, reps, cost_rows, args.quick, args.models)
            for cfg in configs:
                for kind in args.models:
                    measure_fovea(ctx, ds, kind, cfg, reps, tag, cost_rows)
    if match_rows:
        m = pd.DataFrame(match_rows)
        m.to_csv(config.TABLES / f"cozum_dogruluk_eslesme{tag}.csv", index=False)
        write_markdown_table(m, config.TABLES / f"cozum_dogruluk_eslesme{tag}.md", floatfmt="{:.3g}")
        print(m.round(6).to_string(index=False))
    if cost_rows:
        raw = pd.DataFrame(cost_rows)
        raw.to_csv(config.TABLES / f"cozum_maliyet_ham{tag}.csv", index=False)
        agg = aggregate(raw)
        agg.to_csv(config.TABLES / f"cozum_maliyet{tag}.csv", index=False)
        write_markdown_table(agg, config.TABLES / f"cozum_maliyet{tag}.md", floatfmt="{:.3f}")
        print(agg.round(3).to_string(index=False))
    with open(config.TABLES / f"cozum_maliyet_ozet{tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
