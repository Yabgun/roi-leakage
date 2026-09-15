"""Adım: Π_ROI'nin TenSEAL ile yeniden üretimi.

1) Doğruluk: gerçek görüntü ve gerçek ROI maskesiyle seçici şifreli çıktı = şifresiz çıktı mı?
2) Maliyet: görüntü boyutu × ROI oranı taraması (Π_ROI Tablo 1-2'nin karşılığı).

Çalıştırma: .venv\\Scripts\\python -m experiments.piroi_benchmark [--quick]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from PIL import Image

import config
from common.images import load_gray, load_mask, square_box_mask
from common.report import write_markdown_table
from he.piroi import PiROI, make_context, random_weights

RHOS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00]


def real_examples(size: int):
    """Hazırlanmış veri varsa gerçek ROI maskeli örnekler; yoksa Kaggle CXR + merkez kutu."""
    ex = []
    brain_meta = config.DATA_PROC / "brain" / "meta.csv"
    if brain_meta.exists():
        m = pd.read_csv(brain_meta).iloc[0]
        ex.append(("beyin MR, tümör maskesi", load_gray(config.DATA_PROC / "brain" / m.img_path, size),
                   load_mask(config.DATA_PROC / "brain" / m.mask_path, size)))
    cq = config.DATA_PROC / "covidqu_manifest.csv"
    if cq.exists():
        m = pd.read_csv(cq).iloc[0]
        ex.append(("akciğer grafisi, akciğer maskesi", load_gray(m.img_path, size), load_mask(m.lung_mask_path, size)))
    if not ex:
        p = next((config.KAGGLE_CXR / "train" / "PNEUMONIA").glob("*"))
        ex.append(("akciğer grafisi (Kaggle), merkez kutu", load_gray(p, size), square_box_mask(size, 0.25)))
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="yalnızca 64×64 ve az tekrar")
    args = ap.parse_args()
    sizes = [64] if args.quick else [64, 128, 256, 512]
    reps = {64: 3, 128: 3, 256: 2, 512: 1}

    t_start = time.perf_counter()
    ctx = make_context()
    print(f"CKKS bağlamı (N={config.PIROI_POLY_MODULUS}) hazır: {time.perf_counter() - t_start:.1f} s", flush=True)

    # 1) Doğruluk
    rows_ok = []
    for size in sizes[:2]:
        m1, m2 = random_weights(size * size, seed=size)
        proto = PiROI(ctx, m1, m2)
        for name, img, mask in real_examples(size):
            out_sel, cost_sel = proto.run(img, mask, measure_bytes=False)
            ref = proto.plaintext(img)
            err = float(np.max(np.abs(out_sel - ref)))
            rows_ok.append({"size": size, "ornek": name, "rho": float(mask.mean()), "maks_mutlak_hata": err,
                            "toplam_s": cost_sel.total_s})
            print(f"[doğruluk] {size}×{size} {name}: rho={mask.mean():.3f} maks hata={err:.2e}", flush=True)
    ok = pd.DataFrame(rows_ok)
    ok.to_csv(config.TABLES / "piroi_dogruluk.csv", index=False)
    write_markdown_table(ok, config.TABLES / "piroi_dogruluk.md", floatfmt="{:.3g}")

    # 2) Maliyet taraması (içerik maliyeti etkilemez; gerçek bir görüntü kullanılır)
    rows = []
    src = Image.fromarray((real_examples(256)[0][1] * 255).astype(np.uint8))
    for size in sizes:
        m1, m2 = random_weights(size * size, seed=size)
        proto = PiROI(ctx, m1, m2)
        img = np.asarray(src.resize((size, size), Image.BILINEAR), dtype=np.float64) / 255.0
        for rho in RHOS:
            mask = square_box_mask(size, rho)
            for rep in range(reps[size]):
                _, c = proto.run(img, mask, measure_bytes=(rep == 0))
                rows.append({"size": size, "rho_hedef": rho, "rho": float(mask.mean()), "tekrar": rep,
                             "sifreleme_s": c.enc_s, "sunucu_s": c.server_s, "cozme_s": c.dec_s,
                             "toplam_s": c.total_s, "ciphertext_sayisi": c.n_ciphertexts,
                             "sifreli_slot": c.extra.get("sifreli_slot", 0),
                             "yukleme_MB": c.upload_bytes / 1e6 if rep == 0 else np.nan})
            last = [r for r in rows if r["size"] == size and r["rho_hedef"] == rho]
            print(f"[maliyet] {size}×{size} rho={rho:.2f}: toplam {np.mean([r['toplam_s'] for r in last]):.2f} s, "
                  f"{last[0]['ciphertext_sayisi']} ciphertext", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(config.TABLES / "piroi_maliyet_ham.csv", index=False)
    agg = df.groupby(["size", "rho_hedef"]).agg(
        rho=("rho", "first"), sifreleme_s=("sifreleme_s", "mean"), sunucu_s=("sunucu_s", "mean"),
        cozme_s=("cozme_s", "mean"), toplam_s=("toplam_s", "mean"), toplam_std=("toplam_s", "std"),
        ciphertext_sayisi=("ciphertext_sayisi", "first"), sifreli_slot=("sifreli_slot", "first"),
        yukleme_MB=("yukleme_MB", "max")).reset_index()
    full = agg[agg.rho_hedef == 1.0].set_index("size")
    agg["hiz_kazanci_toplam"] = [full.loc[s, "toplam_s"] / t for s, t in zip(agg["size"], agg["toplam_s"])]
    agg["hiz_kazanci_sifreleme"] = [full.loc[s, "sifreleme_s"] / t for s, t in zip(agg["size"], agg["sifreleme_s"])]
    agg.to_csv(config.TABLES / "piroi_maliyet.csv", index=False)
    write_markdown_table(agg, config.TABLES / "piroi_maliyet.md", floatfmt="{:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for size in sizes:
        s = agg[agg["size"] == size]
        ax.plot(s["rho"], s["hiz_kazanci_toplam"], marker="o", label=f"{size}×{size}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Şifreli alan oranı ρ")
    ax.set_ylabel("Tam HE'ye göre toplam hız kazancı (×)")
    ax.set_title("Π_ROI yeniden üretimi (TenSEAL CKKS, N=16384)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES / "piroi_hiz_kazanci.png", dpi=160)
    with open(config.LOGS / "piroi_benchmark.json", "w", encoding="utf-8") as f:
        json.dump({"sure_s": time.perf_counter() - t_start, "boyutlar": sizes}, f)
    print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
