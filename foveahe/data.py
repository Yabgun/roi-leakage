"""FoveaHE veri katmanı: veri kümeleri, odaklı temsil katmanlarının disk önbelleği ve eğitim önbelleği.

Katmanlar orijinal çözünürlükten (beyin 512 px, 15 görüntü 256 px; COVID-QU-Ex 256 px) bir kez hesaplanır ve
`data/processed/fovea/<veri>/<anahtar>.npy` altında uint8 (N,S,S) saklanır; satır sırası manifesto sırasıdır
(`experiments.attack_context` ile aynı). Geometri `geom.npy` (N,3) float32. `index.json` parametreleri ve
tamamlanan anahtarları tutar; yarım kalan anahtar yeniden hesaplanır.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from PIL import Image

import config
from attacks.context_cnn import random_affine, server_view
from foveahe.representation import MARGIN, MIN_SIDE, FoveaSpec, extract, parse_layer_key, render, roi_geometry

STORE = config.DATA_PROC / "fovea"
DISPLAY = {"brain": "beyin MR (Cheng)", "covidqu": "akciğer grafisi (COVID-QU-Ex)"}


@dataclass
class Dataset:
    name: str
    y: np.ndarray
    labels: list
    img_paths: list
    mask_paths: list
    folds: np.ndarray | None = None      # beyin: hasta bazlı resmi katlar (1..5)
    groups: np.ndarray | None = None     # beyin: hasta kimliği (bootstrap)
    train_idx: np.ndarray | None = None  # COVID-QU-Ex: resmi Train + Val
    test_idx: np.ndarray | None = None   # COVID-QU-Ex: resmi Test

    @property
    def display(self) -> str:
        return DISPLAY[self.name]


def load_dataset(name: str) -> Dataset:
    """Saldırı B ile aynı satırlar, etiketler ve bölmeler (beyin: tümör ROI'si; COVID-QU-Ex: akciğer ROI'si)."""
    if name == "brain":
        base = config.DATA_PROC / "brain"
        meta = pd.read_csv(base / "meta.csv")
        return Dataset(name, meta["label"].to_numpy() - 1, meta.groupby("label")["label_name"].first().tolist(),
                       [base / p for p in meta.img_path], [base / p for p in meta.mask_path],
                       folds=meta["fold"].to_numpy(), groups=meta["pid"].to_numpy())
    if name == "covidqu":
        man = pd.read_csv(config.DATA_PROC / "covidqu_manifest.csv")
        man = man[man.lung_mask_path.notna() & (man.lung_mask_path != "")].reset_index(drop=True)
        labels = ["COVID-19", "Non-COVID", "Normal"]
        return Dataset(name, man.label.map({l: i for i, l in enumerate(labels)}).to_numpy(), labels,
                       list(man.img_path), list(man.lung_mask_path),
                       train_idx=np.flatnonzero(man.split.isin(["Train", "Val"]).to_numpy()),
                       test_idx=np.flatnonzero((man.split == "Test").to_numpy()))
    raise ValueError(name)


def _read_pair(paths):
    img_path, mask_path = paths
    with Image.open(img_path) as im:
        img = np.asarray(im.convert("L"), dtype=np.uint8)
    with Image.open(mask_path) as mm:
        mask = np.asarray(mm.convert("L")) > 127
    if img.shape != mask.shape:
        raise ValueError(f"görüntü ve maske boyutu farklı: {img_path}")
    return img, mask


def _read_index(ds: Dataset) -> dict:
    path = STORE / ds.name / "index.json"
    params = {"margin": MARGIN, "min_side": MIN_SIDE, "n": int(len(ds.y))}
    if not path.exists():
        return {**params, "keys": [], "geom": False}
    index = json.loads(path.read_text(encoding="utf-8"))
    if {k: index.get(k) for k in params} != params:
        raise RuntimeError(f"{path}: önbellek parametreleri farklı ({index} / {params}); klasörü _arsiv'e taşıyın")
    return index


def build_layers(ds: Dataset, keys, device: str = "cuda", chunk: int = 512, log=print) -> None:
    """Eksik katmanları ve geometriyi tek geçişte orijinal çözünürlükten hesaplayıp diske yazar."""
    out = STORE / ds.name
    out.mkdir(parents=True, exist_ok=True)
    index = _read_index(ds)
    missing = [k for k in dict.fromkeys(keys) if k not in index["keys"]]
    if not missing and index["geom"]:
        return
    n = len(ds.y)
    t0 = time.perf_counter()
    arrays = {k: np.lib.format.open_memmap(out / f"{k}.npy", mode="w+", dtype=np.uint8,
                                           shape=(n, parse_layer_key(k)[1], parse_layer_key(k)[1]))
              for k in missing}
    geom = np.zeros((n, 3), dtype=np.float32)
    pairs = list(zip(ds.img_paths, ds.mask_paths))
    with ThreadPoolExecutor(12) as ex, torch.no_grad():
        for start in range(0, n, chunk):
            ids = np.arange(start, min(n, start + chunk))
            loaded = list(ex.map(_read_pair, [pairs[i] for i in ids]))
            by_shape = {}
            for pos, (img, _) in enumerate(loaded):
                by_shape.setdefault(img.shape, []).append(pos)
            for pos in by_shape.values():
                sel = ids[pos]
                img = torch.from_numpy(np.stack([loaded[p][0] for p in pos])).to(device).float().div_(255).unsqueeze(1)
                mask = torch.from_numpy(np.stack([loaded[p][1] for p in pos])).to(device)
                g = roi_geometry(mask)
                geom[sel] = g.cpu().numpy()
                for k in missing:
                    lay = extract(img, g, k)[:, 0].mul_(255).round_().clamp_(0, 255)
                    arrays[k][sel] = lay.to(torch.uint8).cpu().numpy()
            if start // chunk % 10 == 0:
                log(f"[fovea:{ds.name}] katmanlar {ids[-1] + 1}/{n} ({time.perf_counter() - t0:.0f} s)")
    for arr in arrays.values():
        arr.flush()
    del arrays
    geom_path = out / "geom.npy"
    if index["geom"]:
        if not np.allclose(np.load(geom_path), geom, atol=1e-6):
            raise RuntimeError(f"{geom_path}: geometri önbellekle uyuşmuyor")
    else:
        np.save(geom_path, geom)
    index["keys"] = sorted(set(index["keys"]) | set(missing))
    index["geom"] = True
    (out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    log(f"[fovea:{ds.name}] {len(missing)} katman hazır: {' '.join(missing)} ({time.perf_counter() - t0:.0f} s)")


def load_layers(ds: Dataset, keys, mmap: bool = False, log=print):
    """Katmanlar (anahtar -> uint8 (N,S,S)) ve geometri (N,3); eksikse önce hesaplanır."""
    keys = list(dict.fromkeys(keys))
    build_layers(ds, keys, log=log)
    out = STORE / ds.name
    mode = "r" if mmap else None
    return {k: np.load(out / f"{k}.npy", mmap_mode=mode) for k in keys}, np.load(out / "geom.npy")


class FoveaCache:
    """`attacks.context_cnn.CpuCache` ile aynı arayüz; `train_and_predict` doğrudan kullanır.

    Katmanlar CPU'da uint8 tutulur, her yığın GPU'da 224×224 odaklı görüntüye çevrilir. Gizli bölge yoktur
    (sunucu görüşü "tam"); artırma yeniden oluşturulmuş görüntüye uygulanır.
    """

    def __init__(self, layers: dict, geom: np.ndarray, labels: np.ndarray, spec: FoveaSpec, device: str = "cuda",
                 max_batch: int = 256, out: int = 224):
        self.spec, self.device, self.out = spec, device, out
        self.layers = {k: torch.from_numpy(layers[k]) for k in spec.layer_keys}
        self.geom = torch.from_numpy(np.ascontiguousarray(geom, dtype=np.float32))
        self.y = torch.from_numpy(np.array(labels, dtype=np.int64, copy=True))
        self._buf = {k: torch.empty((max_batch, *t.shape[1:]), dtype=torch.uint8).pin_memory()
                     for k, t in self.layers.items()}

    def images(self, idx) -> torch.Tensor:
        """(n,1,out,out) odaklı görüntü, [0,1], artırmasız."""
        idx_t = torch.from_numpy(np.asarray(idx, dtype=np.int64))
        n = len(idx_t)
        lay = {}
        for k, t in self.layers.items():
            torch.index_select(t, 0, idx_t, out=self._buf[k][:n])
            lay[k] = self._buf[k][:n].to(self.device, non_blocking=True).float().div_(255).unsqueeze(1)
        return render(lay, self.geom[idx_t].to(self.device, non_blocking=True), self.spec, self.out)

    def batch(self, idx, view: str = "tam", train: bool = False, extra_hidden_fn=None):
        x = self.images(idx)
        m = torch.zeros_like(x, dtype=torch.bool)
        if train:
            x, m = random_affine(x, m)
        y = self.y[torch.from_numpy(np.asarray(idx, dtype=np.int64))]
        return server_view(x, m, "tam"), y.to(self.device, non_blocking=True)
