"""Yalnizca goruntu boyutu (sifreli ROI'de bile sunucunun gordugu meta veri) etiketi sizdiriyor mu?"""
import glob, os, numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "kaggle_cxr", "train")
sz = {}
for c in ["NORMAL", "PNEUMONIA", "COVID19"]:
    a = []
    for f in sorted(glob.glob(os.path.join(B, c, "*")))[:900]:
        with Image.open(f) as im:
            a.append(im.size)
    sz[c] = np.array(a, float)
    w, h = sz[c][:, 0], sz[c][:, 1]
    print(f"{c:10s} n={len(a)} genislik medyan={np.median(w):.0f} yukseklik medyan={np.median(h):.0f} en/boy medyan={np.median(w / h):.2f}")

for a, b in [("NORMAL", "PNEUMONIA"), ("NORMAL", "COVID19")]:
    y = np.r_[np.zeros(len(sz[a])), np.ones(len(sz[b]))]
    r = np.r_[sz[a][:, 0] / sz[a][:, 1], sz[b][:, 0] / sz[b][:, 1]]
    px = np.r_[sz[a].prod(1), sz[b].prod(1)]
    f = lambda s: max(roc_auc_score(y, s), 1 - roc_auc_score(y, s))
    print(f"{a} vs {b}: yalnizca en/boy orani AUC={f(r):.3f} | yalnizca piksel sayisi AUC={f(px):.3f}")
