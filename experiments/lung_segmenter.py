"""Adım (gerçekçilik testi hazırlığı): akciğer segmentasyon modeli.

1) COVID-QU-Ex Train bölmesindeki akciğer maskeleriyle U-Net eğit, Val'de Dice ölç.
2) Kaggle CXR (train/test) görüntülerine akciğer maskesi üret, `data/processed/kaggle_lung_masks/` altına kaydet.

Çalıştırma: .venv\\Scripts\\python -m experiments.lung_segmenter [--epochs 6]
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import config
from common.images import load_gray
from common.segmentation import UNet, dice_loss, dice_score, postprocess
from experiments.attack_context import RES, build_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda"

    man = pd.read_csv(config.DATA_PROC / "covidqu_manifest.csv")
    imgs, masks = build_cache("covidqu", man.img_path, man.lung_mask_path)
    tr = np.flatnonzero((man.split == "Train").to_numpy())
    va = np.flatnonzero((man.split == "Val").to_numpy())

    model = UNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = args.epochs * math.ceil(len(tr) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(tr)
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            b = torch.from_numpy(perm[i:i + args.batch])
            x = torch.from_numpy(imgs[b.numpy()]).to(dev).float().div_(255).unsqueeze(1)
            y = torch.from_numpy(masks[b.numpy()]).to(dev).float().unsqueeze(1)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss) * len(b)
        print(f"epoch {ep + 1}/{args.epochs} kayıp={tot / len(perm):.4f} ({time.perf_counter() - t0:.0f} s)", flush=True)

    model.eval()
    dices = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(va), 128):
            b = va[i:i + 128]
            x = torch.from_numpy(imgs[b]).to(dev).float().div_(255).unsqueeze(1)
            pred = (torch.sigmoid(model(x).float()) > 0.5).squeeze(1).cpu().numpy()
            dices += [dice_score(postprocess(p), t) for p, t in zip(pred, masks[b])]
    val_dice = float(np.mean(dices))
    print(f"COVID-QU-Ex Val Dice = {val_dice:.4f}", flush=True)
    config.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.CHECKPOINTS / "lung_unet.pt")

    kag = pd.read_csv(config.DATA_PROC / "kaggle_cxr_manifest.csv")
    out_dir = config.DATA_PROC / "kaggle_lung_masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths, fracs = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(kag), 64):
            part = kag.iloc[i:i + 64]
            x = torch.from_numpy(np.stack([load_gray(p, RES) for p in part.img_path])).to(dev).unsqueeze(1)
            pred = (torch.sigmoid(model(x).float()) > 0.5).squeeze(1).cpu().numpy()
            for j, p in zip(part.index, pred):
                m = postprocess(p)
                path = out_dir / f"{j}.png"
                Image.fromarray((m * 255).astype(np.uint8)).save(path)
                paths.append(str(path))
                fracs.append(float(m.mean()))
    kag["lung_mask_path"] = paths
    kag["lung_frac"] = fracs
    kag.to_csv(config.DATA_PROC / "kaggle_cxr_manifest.csv", index=False)
    summary = {"covidqu_val_dice": val_dice, "kaggle_maske_sayisi": len(paths),
               "kaggle_akciger_orani_medyan": float(np.median(fracs)),
               "kaggle_akciger_orani_cok_kucuk(<%5)": int((np.array(fracs) < 0.05).sum())}
    with open(config.TABLES / "akciger_segmentasyon.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(summary)


if __name__ == "__main__":
    main()
