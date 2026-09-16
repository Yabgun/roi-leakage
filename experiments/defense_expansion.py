"""Adım: Savunma ve gerçek bedel.

Politikalar (her birinde saldırgan o görüşe göre yeniden eğitilir, yani adaptiftir):
- goruntu_roi      : Π_ROI varsayılanı, her görüntünün kendi ROI'si gizli (meta veri sızar)
- kanonik          : tüm görüntülerde aynı ROI (maske kütlesinin %99'u) ∪ kendi ROI'si
- kanonik_genisK   : kanonik ROI K piksel genişletilmiş
- sizinti_gudumlu  : kanonikten başla, occlusion ile en çok sızdıran yamaları ekle, tekrar et
- rastgele         : kanonikten başla, rastgele yamalar ekle (kıyas)
Sonra gizli alan oranı ρ, Π_ROI maliyet tablosu üzerinden hız kazancına çevrilir.

`--norm gorunur` (yalnız beyin, inceleme S1): ham yoğunluklar, yüzdelikler yalnızca görünür piksellerden; çıktılar
`savunma_gnorm.csv|md`, `savunma_ozet_gnorm.json`. `--recost`: deney koşmadan var olan tabloların hız kazancı sütunlarını
güncel `piroi_maliyet.csv` ile yeniden hesaplar ve tez şekillerini çizer (`savunma_{veri}_tez.png|pdf`).

Çalıştırma: .venv\\Scripts\\python -m experiments.defense_expansion [--dataset covidqu brain] [--norm kesit|gorunur]
            [--quick] [--recost]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import config
from attacks.context_cnn import CpuCache, VisibleNormCache, train_and_predict
from common.evaluation import auc_score
from common.report import write_markdown_table
from common.tez_bicim import kaydet
from defenses.roi_expansion import (canonical_roi, const_hidden_fn, dilate_np, hidden_fraction, occlusion_drops,
                                    patch_grid)
from experiments.attack_context import RES, build_cache, log

GRID = 8


def setup(dataset: str, quick: bool):
    rng = np.random.default_rng(0)
    if dataset == "covidqu":
        man = pd.read_csv(config.DATA_PROC / "covidqu_manifest.csv")
        imgs, masks = build_cache("covidqu", man.img_path, man.lung_mask_path)
        labels = ["COVID-19", "Non-COVID", "Normal"]
        y = man.label.map({l: i for i, l in enumerate(labels)}).to_numpy()
        tr_all = np.flatnonzero((man.split == "Train").to_numpy())
        tr = rng.choice(tr_all, 2000 if quick else 10000, replace=False)
        va = rng.choice(np.flatnonzero((man.split == "Val").to_numpy()), 300 if quick else 1500, replace=False)
        te_all = np.flatnonzero((man.split == "Test").to_numpy())
        te = rng.choice(te_all, 800, replace=False) if quick else te_all
        epochs = 1 if quick else 3
        canon_src = tr_all
    else:
        meta = pd.read_csv(config.DATA_PROC / "brain" / "meta.csv")
        base = config.DATA_PROC / "brain"
        imgs, masks = build_cache("brain", [base / p for p in meta.img_path], [base / p for p in meta.mask_path])
        y = meta["label"].to_numpy() - 1
        folds = meta["fold"].to_numpy()
        fids = np.unique(folds)
        te = np.flatnonzero(folds == fids[-1])
        va = np.flatnonzero(folds == fids[-2])
        tr = np.flatnonzero(~np.isin(folds, fids[-2:]))
        epochs = 1 if quick else 12
        canon_src = tr
    return imgs, masks, y, tr, va, te, epochs, canon_src


def evaluate(cache, y, tr, te, hidden, masks, epochs, seed, n_classes):
    probs, model = train_and_predict(cache, tr, te, "tam", n_classes, epochs=epochs, seed=seed,
                                     extra_hidden_fn=const_hidden_fn(hidden), log=lambda *_: None)
    return auc_score(y[te], probs), hidden_fraction(hidden, masks, te), model


def run(dataset: str, quick: bool, seed: int = 0, norm: str = "kesit"):
    imgs, masks, y, tr, va, te, epochs, canon_src = setup(dataset, quick)
    n_classes = len(np.unique(y))
    if norm == "gorunur":
        from experiments.brain_visible_norm import load_raw
        cache = VisibleNormCache(load_raw(), masks, y)
    else:
        cache = CpuCache(imgs, masks, y)
    rows = []
    empty = np.zeros((RES, RES), dtype=bool)

    def record(policy, step, hidden, auc, frac, extra=None):
        rows.append({"veri": dataset, "politika": policy, "adim": step, "gizli_alan": frac, "auc": auc,
                     "sabit_gizli_alan": float(hidden.mean()), "gorunur_piksel": int((~hidden).sum()),
                     **(extra or {})})
        log(f"[savunma:{dataset}] {policy:18s} adım={step:2d} gizli alan={frac:.4f} "
            f"görünür piksel={int((~hidden).sum())} AUC={auc:.3f}")

    t0 = time.perf_counter()
    # Akıl sağlığı kontrolü: tüm görüntü gizliyse girdiler özdeştir, AUC tam 0.5 olmalı
    full = np.ones((RES, RES), dtype=bool)
    auc, frac, _ = evaluate(cache, y, tr, te, full, masks, 1, seed, n_classes)
    record("tamamen_gizli", 0, full, auc, frac)
    auc, frac, _ = evaluate(cache, y, tr, te, empty, masks, epochs, seed, n_classes)
    record("goruntu_roi", 0, empty, auc, frac)

    canon = canonical_roi(masks[canon_src], coverage=0.99)
    for px in [0, 8, 16, 32, 64]:
        hidden = dilate_np(canon, px)
        auc, frac, model = evaluate(cache, y, tr, te, hidden, masks, epochs, seed, n_classes)
        record("kanonik" if px == 0 else f"kanonik_genis{px}", 0, hidden, auc, frac)
        if px == 0:
            canon_model = model

    patches = patch_grid(RES, GRID)
    step_size = 4
    hidden, model, step = canon.copy(), canon_model, 0
    chosen = []
    while True:
        cand = [p for p in range(len(patches)) if (patches[p] & ~hidden).any()]
        if not cand:
            break
        _, drops = occlusion_drops(model, cache, va, y, hidden, patches, cand)
        top = sorted(cand, key=lambda p: drops[p], reverse=True)[:step_size]
        chosen.append(top)
        for p in top:
            hidden = hidden | patches[p]
        step += 1
        auc, frac, model = evaluate(cache, y, tr, te, hidden, masks, epochs, seed, n_classes)
        record("sizinti_gudumlu", step, hidden, auc, frac, {"eklenen_yamalar": " ".join(map(str, top))})
        if auc <= 0.55 or quick and step >= 2:
            break

    rng = np.random.default_rng(seed + 100)
    n_steps = step
    for order_id in range(1 if quick else 2):
        hidden = canon.copy()
        cand = [p for p in range(len(patches)) if (patches[p] & ~hidden).any()]
        order = list(rng.permutation(cand))
        for s in range(1, n_steps + 1):
            for p in order[(s - 1) * step_size:s * step_size]:
                hidden = hidden | patches[p]
            if s % 2 == 0 or s == n_steps:
                auc, frac, _ = evaluate(cache, y, tr, te, hidden, masks, epochs, seed, n_classes)
                record(f"rastgele_{order_id}", s, hidden, auc, frac)

    df = pd.DataFrame(rows)
    df["sure_dk"] = (time.perf_counter() - t0) / 60
    df["normalizasyon"] = norm if dataset == "brain" else "yok (dağıtıldığı biçimde)"
    return df


def attach_cost(df: pd.DataFrame) -> pd.DataFrame:
    """ρ -> Π_ROI yeniden üretimindeki toplam süre üzerinden hız kazancı (256 ve 512 görüntü boyutu)."""
    cost_path = config.TABLES / "piroi_maliyet.csv"
    if not cost_path.exists():
        return df
    cost = pd.read_csv(cost_path)
    for size in [256, 512]:
        c = cost[cost["size"] == size].sort_values("rho")
        if c.empty:
            continue
        t_full = float(c[c.rho_hedef == 1.0].toplam_s.iloc[0])
        df[f"hiz_kazanci_{size}"] = t_full / np.interp(df["gizli_alan"], c["rho"], c["toplam_s"])
    return df


def plot(df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for dataset, d in df.groupby("veri"):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for policy, g in d.groupby(d["politika"].str.replace(r"_\d+$", "", regex=True)):
            g = g.sort_values("gizli_alan")
            style = "-o" if policy in ("sizinti_gudumlu", "rastgele") else "s"
            ax.plot(g["gizli_alan"], g["auc"], style, label=policy, alpha=0.85)
        ax.axhline(0.5, color="gray", lw=0.8, ls=":")
        ax.axhline(0.6, color="red", lw=0.8, ls="--")
        ax.set_xlabel("Şifrelenen alan oranı ρ")
        ax.set_ylabel("Saldırgan AUC")
        ax.set_title(f"Savunma eğrileri: {dataset}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(config.FIGURES / f"savunma_{dataset}.png", dpi=160)


POLICY_TR = {"goruntu_roi": r"$\Pi_{\mathrm{ROI}}$ varsayılanı (kendi ROI'si)", "kanonik": "kanonik ROI (+ genişletme)",
             "sizinti_gudumlu": "sızıntı güdümlü genişletme", "rastgele": "rastgele genişletme",
             "tamamen_gizli": "tamamen gizli (kontrol)"}


def plot_tez(frames: dict):
    """Tez şekilleri: Türkçe gösterge, şekil içi başlık yok, şans düzeyi (0.5) ve gizlilik hedefi (0.8) çizgileri."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    styles = {"goruntu_roi": dict(marker="D", ls="none", color="#b3261e", ms=7),
              "kanonik": dict(marker="s", ls="none", color="#6b5ca5", ms=6),
              "sizinti_gudumlu": dict(marker="o", ls="-", color="#1c5cab", ms=4, lw=1.4),
              "rastgele": dict(marker="^", ls="--", color="#898781", ms=4, lw=1.0),
              "tamamen_gizli": dict(marker="x", ls="none", color="#0b0b0b", ms=7)}
    for dataset, d in frames.items():
        fig, ax = plt.subplots(figsize=(6.4, 3.9))
        base = d["politika"].str.replace(r"_\d+$", "", regex=True).str.replace(r"^kanonik_genis\d+$", "kanonik", regex=True)
        for policy in ["goruntu_roi", "kanonik", "sizinti_gudumlu", "rastgele", "tamamen_gizli"]:
            sel = d[base == policy]
            if sel.empty:
                continue
            first = True
            for _, g in sel.groupby(sel["politika"] if policy == "rastgele" else base[base == policy]):
                g = g.sort_values("gizli_alan")
                ax.plot(g["gizli_alan"], g["auc"], label=POLICY_TR[policy] if first else None, alpha=0.9,
                        **styles[policy])
                first = False
        ax.axhline(0.8, color="#b3261e", lw=0.9, ls="--")
        ax.text(0.01, 0.8, " gizlilik hedefi 0.8", color="#b3261e", fontsize=8, va="bottom",
                transform=ax.get_yaxis_transform())
        ax.axhline(0.5, color="#898781", lw=0.8, ls=":")
        # şans düzeyi etiketi sağda: sol alttaki göstergenin altında kalmasın
        ax.text(0.62, 0.5, "şans düzeyi 0.5 ", color="#52514e", fontsize=8, va="bottom", ha="right",
                transform=ax.get_yaxis_transform())
        ax.set_xlabel(r"Şifrelenen alan oranı $\rho$")
        ax.set_ylabel("Saldırgan AUC")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0.45, 1.01)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.01, 0.13), frameon=True)
        fig.tight_layout()
        kaydet(fig, config.FIGURES / f"savunma_{dataset}_tez.png")
        plt.close(fig)


def summarize_json(df: pd.DataFrame) -> dict:
    summary = {}
    for dataset, d in df.groupby("veri"):
        s = {}
        for target in [0.8, 0.7, 0.6]:
            ok = d[(d.politika == "sizinti_gudumlu") & (d.auc <= target)]
            s[f"AUC<={target} icin min gizli alan (sizinti_gudumlu)"] = float(ok.gizli_alan.min()) if len(ok) else None
        s["goruntu_roi AUC"] = float(d[d.politika == "goruntu_roi"].auc.iloc[0])
        s["kanonik AUC"] = float(d[d.politika == "kanonik"].auc.iloc[0])
        summary[dataset] = s
    return summary


def tez_frames() -> dict:
    """Tezde kullanılan savunma satırları: COVID-QU-Ex `savunma.csv`, beyin varsa `savunma_gnorm.csv` (S1)."""
    frames = {}
    main_csv, gnorm_csv = config.TABLES / "savunma.csv", config.TABLES / "savunma_gnorm.csv"
    if main_csv.exists():
        df = pd.read_csv(main_csv)
        frames = {v: d for v, d in df.groupby("veri")}
    if gnorm_csv.exists():
        g = pd.read_csv(gnorm_csv)
        frames.update({v: d for v, d in g.groupby("veri")})
    return frames


def recost():
    for name in ("savunma.csv", "savunma_gnorm.csv"):
        path = config.TABLES / name
        if not path.exists():
            continue
        df = attach_cost(pd.read_csv(path))
        df.to_csv(path, index=False)
        write_markdown_table(df.drop(columns=[c for c in ["eklenen_yamalar"] if c in df]), path.with_suffix(".md"),
                             floatfmt="{:.3f}")
        print(f"{name}: hız kazancı sütunları güncel piroi_maliyet.csv ile yeniden hesaplandı")
    plot_tez(tez_frames())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="*", default=["covidqu", "brain"])
    ap.add_argument("--norm", choices=["kesit", "gorunur"], default="kesit")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--recost", action="store_true", help="deney koşmadan hız kazancı sütunlarını ve tez şekillerini yenile")
    args = ap.parse_args()
    if args.recost:
        recost()
        return
    if args.norm == "gorunur" and args.dataset != ["brain"]:
        raise SystemExit("--norm gorunur yalnız beyin MR için: --dataset brain")
    tag = ("_hizli" if args.quick else "") + ("_gnorm" if args.norm == "gorunur" else "")
    frames = [run(d, args.quick, norm=args.norm) for d in args.dataset]
    df = attach_cost(pd.concat(frames, ignore_index=True))
    out = config.TABLES / f"savunma{tag}.csv"
    if out.exists() and not args.quick:
        old = pd.read_csv(out)
        df = pd.concat([old[~old.veri.isin(df.veri.unique())], df], ignore_index=True)
    df.to_csv(out, index=False)
    write_markdown_table(df.drop(columns=[c for c in ["eklenen_yamalar"] if c in df]), config.TABLES / f"savunma{tag}.md",
                         floatfmt="{:.3f}")
    if args.norm == "kesit" and not args.quick:
        plot(df)
    summary = summarize_json(df)
    with open(config.TABLES / f"savunma_ozet{tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.quick:
        plot_tez(tez_frames())


if __name__ == "__main__":
    main()
