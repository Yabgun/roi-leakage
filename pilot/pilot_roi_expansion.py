"""Pilot: Teshisi gizlemek icin goruntunun ne kadarini sifrelemek gerekir?
Normal vs Pnomoni (ayni kaynak), 64x64, 8x8 yama izgarasi (64 yama).
Baslangic ROI = merkezdeki 6x6 yama (%56, kaba akciger bolgesi).
Politikalar: sizinti-gudumlu (occlusion ile en cok sizdiran yamayi ekle, saldirgani yeniden egit),
rastgele sira, esit genisletme (dilation). Iki onisleme: naif yeniden boyutlandirma ve normalize
(en-boy korunarak merkez kirpma + histogram esitleme)."""
import os, glob, random
import numpy as np
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import roc_auc_score

random.seed(0); np.random.seed(0)
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "kaggle_cxr", "train")
S, P = 64, 8  # goruntu boyutu, yama boyutu
G = S // P

def load(cls, n, mode):
    files = sorted(glob.glob(os.path.join(BASE, cls, "*"))); random.Random(1).shuffle(files)
    out = []
    for f in files[:n]:
        im = Image.open(f); im.draft("L", (S * 4, S * 4)); im = im.convert("L")
        if mode == "normalize":
            w, h = im.size; m = min(w, h)
            im = im.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m))
            im = ImageOps.equalize(im.resize((S, S)))
        else:
            im = im.resize((S, S))
        out.append(np.asarray(im, dtype=np.float32) / 255.0)
    return np.stack(out)

def patch_mask(hidden):
    m = np.ones((S, S), bool)
    for (r, c) in hidden:
        m[r * P:(r + 1) * P, c * P:(c + 1) * P] = False
    return m

def clf():
    return make_pipeline(StandardScaler(), PCA(n_components=40, random_state=0), LogisticRegression(max_iter=2000, C=0.1))

def cv_auc(X, y, visible):
    if visible.sum() == 0:
        return 0.5
    p = cross_val_predict(clf(), X[:, visible], y, cv=StratifiedKFold(3, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return roc_auc_score(y, p)

ALL = [(r, c) for r in range(G) for c in range(G)]
ROI0 = [(r, c) for r in range(1, 7) for c in range(1, 7)]  # merkez 6x6 yama

def run(mode):
    X0 = np.concatenate([load("NORMAL", 900, mode), load("PNEUMONIA", 900, mode)]).reshape(1800, -1)
    y = np.r_[np.zeros(900), np.ones(900)]
    print(f"\n### onisleme = {mode}")
    print("tam goruntu AUC: %.3f | yalnizca ROI0 disi (baslangic) AUC: %.3f" % (cv_auc(X0, y, np.ones(S * S, bool)), cv_auc(X0, y, patch_mask(ROI0).ravel())))
    rest = [p for p in ALL if p not in ROI0]
    results = {}
    # 1) sizinti-gudumlu (adaptif): her adimda saldirgani egit, occlusion ile en onemli 2 yamayi gizle
    hidden = list(ROI0); curve = []
    Xtr, Xva, ytr, yva = train_test_split(X0, y, test_size=0.3, stratify=y, random_state=0)
    while True:
        vis = patch_mask(hidden).ravel()
        curve.append((len(hidden) / len(ALL), cv_auc(X0, y, vis)))
        cand = [p for p in ALL if p not in hidden]
        if not cand:
            break
        model = clf().fit(Xtr[:, vis], ytr)
        base = roc_auc_score(yva, model.predict_proba(Xva[:, vis])[:, 1])
        scores = []
        idx_map = np.flatnonzero(vis)
        mean_tr = Xtr[:, vis].mean(0)
        for p in cand:
            pm = ~patch_mask([p]).ravel()           # bu yamanin pikselleri
            cols = np.flatnonzero(pm[idx_map])      # gorunur vektordeki konumlari
            Xo = Xva[:, vis].copy(); Xo[:, cols] = mean_tr[cols]
            scores.append(base - roc_auc_score(yva, model.predict_proba(Xo)[:, 1]))
        order = np.argsort(scores)[::-1]
        hidden += [cand[i] for i in order[:2]]
    results["sizinti-gudumlu"] = curve
    # 2) rastgele sira (3 tekrar ortalamasi)
    rc = []
    for rep in range(3):
        order = rest.copy(); random.Random(10 + rep).shuffle(order)
        hidden = list(ROI0); cur = []
        for k in range(0, len(order) + 1, 2):
            h = hidden + order[:k]
            cur.append((len(h) / len(ALL), cv_auc(X0, y, patch_mask(h).ravel())))
        rc.append(cur)
    results["rastgele"] = [(rc[0][i][0], np.mean([r[i][1] for r in rc])) for i in range(len(rc[0]))]
    # 3) esit genisletme: once kenar halkasi (tum 28 yama) tek adim; ara adim: ust+alt satirlar
    dil = [(len(ROI0) / 64, cv_auc(X0, y, patch_mask(ROI0).ravel()))]
    top_bottom = [(r, c) for r in (0, 7) for c in range(G)]
    dil.append(((len(ROI0) + len(top_bottom)) / 64, cv_auc(X0, y, patch_mask(ROI0 + top_bottom).ravel())))
    dil.append((1.0, 0.5))
    results["esit genisletme"] = dil
    targets = [0.80, 0.70, 0.60]
    for name, cur in results.items():
        pts = " ".join(f"{r:.2f}:{a:.2f}" for r, a in cur[::2])
        need = []
        for t in targets:
            ok = [r for r, a in cur if a <= t]
            need.append(f"AUC<={t:.2f} icin rho>={min(ok):.2f}" if ok else f"AUC<={t:.2f} icin tamami")
        print(f"{name:16s} | {' | '.join(need)}")
        print(f"   (rho:AUC) {pts}")

for mode in ["naif", "normalize"]:
    run(mode)
