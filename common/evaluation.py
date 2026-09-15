"""Değerlendirme yardımcıları: çok sınıflı AUC, grup (hasta) bazlı bootstrap güven aralığı."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auc_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """İkili sınıfta pozitif sınıf AUC'si, çok sınıfta makro one-vs-rest AUC."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    if proba.ndim == 1 or proba.shape[1] == 2:
        p = proba if proba.ndim == 1 else proba[:, 1]
        return float(roc_auc_score(y_true, p))
    return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))


def bootstrap_ci(y_true, proba, groups=None, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """AUC için %95 güven aralığı. groups verilirse yeniden örnekleme grup (hasta) düzeyinde yapılır."""
    rng = np.random.default_rng(seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    if groups is None:
        units = np.arange(len(y_true))
        members = {u: np.array([u]) for u in units}
    else:
        groups = np.asarray(groups)
        units = np.unique(groups)
        members = {u: np.flatnonzero(groups == u) for u in units}
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(units, size=len(units), replace=True)
        idx = np.concatenate([members[u] for u in pick])
        if len(np.unique(y_true[idx])) < len(np.unique(y_true)):
            continue
        try:
            stats.append(auc_score(y_true[idx], proba[idx]))
        except ValueError:
            continue
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
