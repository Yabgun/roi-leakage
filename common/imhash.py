"""Algısal fark hash'i (dHash) ile veri setleri arasında tekrar eden görüntüleri bulma."""
from __future__ import annotations

import numpy as np
from PIL import Image


def dhash(path, size: int = 8) -> int:
    """64 bitlik dHash: yeniden boyutlandırma ve JPEG sıkıştırmasına dayanıklı."""
    with Image.open(path) as im:
        try:
            im.draft("L", (size * 16, size * 16))
        except Exception:
            pass
        g = im.convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).ravel()
    return int(np.packbits(bits).view(">u8")[0])


def _popcount64(x: np.ndarray) -> np.ndarray:
    return np.unpackbits(x.astype(">u8").view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def near_duplicate_pairs(hash_a: np.ndarray, hash_b: np.ndarray, max_dist: int = 6):
    """hash_a, hash_b: uint64 dizileri. Döner: (i, j, mesafe) listesi, i A'da j B'de indeks.

    Güvercin yuvası ilkesi: mesafe < 8 ise 8 baytın en az biri birebir aynıdır. Her bayt bandı için
    aynı değere sahip A ve B grupları küçük matrislerle karşılaştırılır."""
    hash_a = np.asarray(hash_a, dtype=np.uint64)
    hash_b = np.asarray(hash_b, dtype=np.uint64)
    found = {}
    for band in range(8):
        ka = ((hash_a >> np.uint64(8 * band)) & np.uint64(0xFF)).astype(np.int64)
        kb = ((hash_b >> np.uint64(8 * band)) & np.uint64(0xFF)).astype(np.int64)
        order_b = np.argsort(kb, kind="stable")
        kb_sorted = kb[order_b]
        for value in np.unique(ka):
            ia = np.flatnonzero(ka == value)
            lo, hi = np.searchsorted(kb_sorted, [value, value + 1])
            if hi <= lo:
                continue
            ib = order_b[lo:hi]
            x = hash_a[ia][:, None] ^ hash_b[ib][None, :]
            d = _popcount64(x.ravel()).reshape(x.shape)
            for r, c in zip(*np.nonzero(d <= max_dist)):
                found[(int(ia[r]), int(ib[c]))] = int(d[r, c])
    return [(i, j, d) for (i, j), d in found.items()]
