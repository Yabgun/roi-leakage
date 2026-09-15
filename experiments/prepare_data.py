"""Adım: Veri hazırlığı.

- Kaggle CXR: manifesto + görüntü imzaları (dHash ve 32×32 küçük resim)
- COVID-QU-Ex: zip açma, manifesto (görüntü, akciğer maskesi, enfeksiyon maskesi, etiket, resmi bölme) + imzalar
- Beyin tümörü (Cheng, figshare): zip açma, .mat -> PNG görüntü + tümör maskesi + meta.csv, resmi 5 katlı bölme
- Veri setleri arası tekrar eden görüntüler: dHash adayları + küçük resim korelasyonu ile doğrulama

Çalıştırma: .venv\\Scripts\\python -m experiments.prepare_data [--only kaggle covidqu brain dups]
"""
from __future__ import annotations

import argparse
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import config
from common.imhash import near_duplicate_pairs

IMG_EXT = {".png", ".jpg", ".jpeg"}
BRAIN_LABELS = {1: "meningioma", 2: "glioma", 3: "pituitary"}


def extract(zip_path: Path, out_dir: Path) -> None:
    marker = out_dir / ".extracted"
    if marker.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    marker.touch()


def signature(path: str):
    """(genişlik, yükseklik, dHash hex, 32×32 float16 küçük resim)."""
    with Image.open(path) as im:
        w, h = im.size
        try:
            im.draft("L", (128, 128))
        except Exception:
            pass
        g = im.convert("L")
        a9 = np.asarray(g.resize((9, 8), Image.BILINEAR), dtype=np.int16)
        thumb = np.asarray(g.resize((32, 32), Image.BILINEAR), dtype=np.float32)
    bits = (a9[:, 1:] > a9[:, :-1]).ravel()
    h64 = int(np.packbits(bits).view(">u8")[0])
    thumb = (thumb - thumb.mean()) / (thumb.std() + 1e-6)
    return w, h, f"{h64:016x}", thumb.astype(np.float16)


def add_signatures(df: pd.DataFrame, thumbs_path: Path, workers: int = 8) -> pd.DataFrame:
    with ThreadPoolExecutor(workers) as ex:
        sigs = list(ex.map(signature, df["img_path"]))
    df = df.copy()
    df["width"] = [s[0] for s in sigs]
    df["height"] = [s[1] for s in sigs]
    df["dhash"] = [s[2] for s in sigs]
    np.save(thumbs_path, np.stack([s[3] for s in sigs]))
    return df


def prepare_kaggle():
    rows = []
    for split in ["train", "test", "val", "train2", "test2"]:
        d = config.KAGGLE_CXR / split
        if not d.exists():
            continue
        for cls_dir in sorted(p for p in d.iterdir() if p.is_dir()):
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in IMG_EXT:
                    rows.append({"img_path": str(f), "split": split, "label": cls_dir.name})
    df = pd.DataFrame(rows)
    # Asıl kullanılacak bölmeler train/test; train2/test2/val aynı Kermany görüntülerinin başka bölmesi (bilgi amaçlı)
    main = df[df.split.isin(["train", "test"])].reset_index(drop=True)
    main = add_signatures(main, config.DATA_PROC / "kaggle_cxr_thumbs.npy")
    main.to_csv(config.DATA_PROC / "kaggle_cxr_manifest.csv", index=False)
    summary = main.groupby(["split", "label"]).size().to_dict()
    print("[kaggle] görüntü sayıları:", summary)
    return {"kaggle_cxr": {f"{k[0]}/{k[1]}": int(v) for k, v in summary.items()}}


def prepare_covidqu():
    ext = config.COVIDQU / "extracted"
    extract(config.COVIDQU / "covidqu.zip", ext)
    rows = []
    for img_dir in sorted(p for p in ext.rglob("images") if p.is_dir()):
        parts = img_dir.relative_to(ext).parts
        subset = "infection" if any("Infection" in p for p in parts) else "lung"
        split, label = parts[-3], parts[-2]
        lung_dir, inf_dir = img_dir.parent / "lung masks", img_dir.parent / "infection masks"
        for f in sorted(img_dir.iterdir()):
            if f.suffix.lower() not in IMG_EXT:
                continue
            lm, im = lung_dir / f.name, inf_dir / f.name
            rows.append({"subset": subset, "split": split, "label": label, "file": f.name, "img_path": str(f),
                         "lung_mask_path": str(lm) if lm.exists() else "",
                         "inf_mask_path": str(im) if im.exists() else ""})
    df = pd.DataFrame(rows)
    lung = df[df.subset == "lung"].drop(columns=["inf_mask_path"]).reset_index(drop=True)
    inf = df[df.subset == "infection"].reset_index(drop=True)
    lung = lung.merge(inf[["label", "file", "inf_mask_path"]], on=["label", "file"], how="left")
    lung["inf_mask_path"] = lung["inf_mask_path"].fillna("")
    lung = add_signatures(lung, config.DATA_PROC / "covidqu_thumbs.npy")
    lung.to_csv(config.DATA_PROC / "covidqu_manifest.csv", index=False)
    inf.to_csv(config.DATA_PROC / "covidqu_infection_manifest.csv", index=False)
    summary = lung.groupby(["split", "label"]).size().to_dict()
    print("[covidqu] akciğer maskeli görüntüler:", summary)
    print("[covidqu] enfeksiyon maskesi eşleşen:", int((lung.inf_mask_path != "").sum()),
          "| maskesi eksik:", int((lung.lung_mask_path == "").sum()))
    return {"covidqu": {f"{k[0]}/{k[1]}": int(v) for k, v in summary.items()},
            "covidqu_enfeksiyon_maskeli": int((lung.inf_mask_path != "").sum())}


def _read_cjdata(mf: Path):
    try:
        import h5py
        with h5py.File(mf, "r") as f:
            cj = f["cjdata"]
            label = int(np.array(cj["label"]).squeeze())
            pid = "".join(chr(int(c)) for c in np.array(cj["PID"]).ravel())
            img = np.array(cj["image"]).T.astype(np.float32)
            mask = np.array(cj["tumorMask"]).T.astype(bool)
    except OSError:
        from scipy.io import loadmat
        cj = loadmat(mf, squeeze_me=True, struct_as_record=False)["cjdata"]
        label, pid = int(cj.label), str(cj.PID)
        img, mask = np.asarray(cj.image, np.float32), np.asarray(cj.tumorMask, bool)
    return label, pid, img, mask


def _load_cvind(path: Path) -> np.ndarray:
    try:
        from scipy.io import loadmat
        d = loadmat(path)
        key = [k for k in d if not k.startswith("__")][0]
        return np.asarray(d[key]).ravel().astype(int)
    except (NotImplementedError, ValueError):
        import h5py
        with h5py.File(path, "r") as f:
            return np.array(f[list(f.keys())[0]]).ravel().astype(int)


def prepare_brain():
    zdir, mat_dir = config.BRAIN / "zips", config.BRAIN / "mat"
    for z in sorted(zdir.glob("brainTumorDataPublic_*.zip")):
        extract(z, mat_dir / z.stem)
    out = config.DATA_PROC / "brain"
    (out / "img").mkdir(parents=True, exist_ok=True)
    (out / "mask").mkdir(parents=True, exist_ok=True)
    rows = []
    for mf in sorted(mat_dir.rglob("*.mat"), key=lambda p: int(p.stem)):
        label, pid, img, mask = _read_cjdata(mf)
        lo, hi = np.percentile(img, [0.5, 99.5])
        img8 = (np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
        i = int(mf.stem)
        Image.fromarray(img8).save(out / "img" / f"{i}.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(out / "mask" / f"{i}.png")
        rows.append({"id": i, "label": label, "label_name": BRAIN_LABELS[label], "pid": pid,
                     "h": img.shape[0], "w": img.shape[1], "img_path": f"img/{i}.png", "mask_path": f"mask/{i}.png",
                     "tumor_frac": float(mask.mean())})
    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    cv = _load_cvind(config.BRAIN / "zips" / "cvind.mat")
    df["fold"] = cv[df["id"].to_numpy() - 1] if len(cv) >= len(df) else -1
    patients_in_multiple_folds = int((df.groupby("pid")["fold"].nunique() > 1).sum())
    df.to_csv(out / "meta.csv", index=False)
    print(f"[beyin] {len(df)} kesit, {df.pid.nunique()} hasta; sınıflar: {df.label_name.value_counts().to_dict()}")
    print(f"[beyin] boyutlar: {df.groupby(['h', 'w']).size().to_dict()} | birden fazla katta görünen hasta: "
          f"{patients_in_multiple_folds}")
    return {"beyin_kesit": len(df), "beyin_hasta": int(df.pid.nunique()),
            "beyin_sinif": df.label_name.value_counts().to_dict(),
            "beyin_boyutlar": {f"{k[0]}x{k[1]}": int(v) for k, v in df.groupby(["h", "w"]).size().items()},
            "beyin_cok_katli_hasta": patients_in_multiple_folds}


def find_duplicates(max_dist: int = 10, min_corr: float = 0.97):
    kag = pd.read_csv(config.DATA_PROC / "kaggle_cxr_manifest.csv")
    cq = pd.read_csv(config.DATA_PROC / "covidqu_manifest.csv")
    ta = np.load(config.DATA_PROC / "kaggle_cxr_thumbs.npy").astype(np.float32).reshape(len(kag), -1)
    tb = np.load(config.DATA_PROC / "covidqu_thumbs.npy").astype(np.float32).reshape(len(cq), -1)
    ha = np.array([int(h, 16) for h in kag.dhash], dtype=np.uint64)
    hb = np.array([int(h, 16) for h in cq.dhash], dtype=np.uint64)
    cand = near_duplicate_pairs(ha, hb, max_dist=max_dist)
    keep = []
    for i, j, d in cand:
        corr = float(np.dot(ta[i], tb[j]) / ta.shape[1])
        if corr >= min_corr:
            keep.append({"kaggle_idx": i, "covidqu_idx": j, "hamming": d, "korelasyon": corr,
                         "kaggle_label": kag.label[i], "kaggle_split": kag.split[i],
                         "covidqu_label": cq.label[j], "covidqu_split": cq.split[j],
                         "kaggle_path": kag.img_path[i], "covidqu_path": cq.img_path[j]})
    dup = pd.DataFrame(keep)
    dup.to_csv(config.DATA_PROC / "cxr_capraz_tekrarlar.csv", index=False)
    n_kag = int(dup.kaggle_idx.nunique()) if len(dup) else 0
    print(f"[tekrar] aday çift: {len(cand)}, doğrulanan: {len(dup)}, COVID-QU-Ex'te kopyası olan Kaggle görüntüsü: "
          f"{n_kag}/{len(kag)}")
    if len(dup):
        print(dup.groupby(["kaggle_label", "covidqu_label"]).kaggle_idx.nunique().to_string())
    return {"tekrar_aday": len(cand), "tekrar_dogrulanan_cift": len(dup), "kopyasi_olan_kaggle": n_kag,
            "kaggle_toplam": len(kag)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["kaggle", "covidqu", "brain", "dups"])
    args = ap.parse_args()
    summary_path = config.DATA_PROC / "veri_ozeti.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    steps = {"kaggle": prepare_kaggle, "covidqu": prepare_covidqu, "brain": prepare_brain, "dups": find_duplicates}
    for name in args.only:
        summary.update(steps[name]())
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
