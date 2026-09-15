"""Pilot: ROI tabanli secici sifrelemede sifrelenmemis arka plan tani etiketini sizdiriyor mu?
Kaba yaklasim: akciger bolgesini merkezi kutu ile 'sifreli' say (sunucu goremez), yalnizca kenar
piksellerinden etiketi tahmin et. Gercek deneyde kutu yerine akciger segmentasyon maskesi kullanilmali."""
import os, glob, random, time
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

random.seed(0)
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "kaggle_cxr", "train")
S = 64

def load(cls, n):
    files = sorted(glob.glob(os.path.join(BASE, cls, "*")))
    random.shuffle(files)
    out = []
    for f in files[:n]:
        im = Image.open(f); im.draft("L", (S * 2, S * 2))
        out.append(np.asarray(im.convert("L").resize((S, S)), dtype=np.float32) / 255.0)
    return np.stack(out)

t0 = time.time()
imgs = {"NORMAL": load("NORMAL", 900), "PNEUMONIA": load("PNEUMONIA", 900), "COVID19": load("COVID19", 460)}
print("yukleme: %.0f s" % (time.time() - t0), {k: v.shape[0] for k, v in imgs.items()})

def box(r0, r1, c0, c1):
    m = np.zeros((S, S), bool)
    m[int(r0 * S):int(r1 * S), int(c0 * S):int(c1 * S)] = True
    return m

MASKS = {
    "tam goruntu (sifresiz)": np.ones((S, S), bool),
    "kutu %64 gizli, kenar gorunur": ~box(0.10, 0.90, 0.10, 0.90),
    "kutu %81 gizli, ince kenar":   ~box(0.05, 0.95, 0.05, 0.95),
    "yalnizca ust %15 serit":       box(0.0, 0.15, 0.0, 1.0),
}

def auc(a, b, mask):
    X = np.concatenate([imgs[a], imgs[b]])[:, mask]
    y = np.r_[np.zeros(len(imgs[a])), np.ones(len(imgs[b]))]
    clf = make_pipeline(StandardScaler(), PCA(n_components=min(100, X.shape[1] - 1), random_state=0),
                        LogisticRegression(max_iter=3000, C=0.1))
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return roc_auc_score(y, p)

print("\n%-32s %8s %18s %16s" % ("sunucunun gordugu bolge", "piksel%", "NORMAL vs PNOMONI", "NORMAL vs COVID"))
for name, m in MASKS.items():
    print("%-32s %7.0f%% %18.3f %16.3f" % (name, 100 * m.mean(), auc("NORMAL", "PNEUMONIA", m), auc("NORMAL", "COVID19", m)))
