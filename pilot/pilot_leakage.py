"""Pilot: TEZ_PLANI_v2 varsayimlarinin gercek Stroke verisinde hizli sinamasi.
H1c  : sifreli skor + acik yari-tanimlayicilardan hassas oznitelik sizintisi
H2   : leakage_k (greedy) vs random_k vs manual
H3   : LR'ye karsi uyelik cikarimi taban cizgisi (savunmasiz)
"""
import itertools, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, KFold
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")
import os
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "stroke", "healthcare-dataset-stroke-data.csv")
df = pd.read_csv(DATA)
df = df[df.gender != "Other"].copy()
df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
df["bmi"] = df["bmi"].fillna(df["bmi"].median())
print("satir:", len(df), "| stroke orani: %.3f" % df.stroke.mean())
print("smoking_status dagilimi:\n", df.smoking_status.value_counts(normalize=True).round(3).to_string())
print("Unknown sigara durumu olanlarin yas<18 orani: %.2f" % (df[df.smoking_status == "Unknown"].age < 18).mean())
print("yas<18 olanlarin Unknown orani: %.2f" % (df[df.age < 18].smoking_status == "Unknown").mean())

# ---- yari-tanimlayici (QI) blok tanimlari ----
QI_BLOCKS = {
    "age": lambda d: d[["age"]],
    "gender": lambda d: (d[["gender"]] == "Male").astype(int),
    "bmi": lambda d: d[["bmi"]],
    "ever_married": lambda d: (d[["ever_married"]] == "Yes").astype(int),
    "work_type": lambda d: pd.get_dummies(d["work_type"], prefix="wt").astype(int),
    "Residence_type": lambda d: (d[["Residence_type"]] == "Urban").astype(int),
}

def qi_matrix(d, blocks):
    if not blocks:
        return None
    return pd.concat([QI_BLOCKS[b](d) for b in blocks], axis=1).values.astype(float)

known = df.smoking_status != "Unknown"
TARGETS = {
    "hypertension": (df.index, df.hypertension.values),
    "heart_disease": (df.index, df.heart_disease.values),
    "high_glucose(>=140)": (df.index, (df.avg_glucose_level >= 140).astype(int).values),
    "smoking=Unknown": (df.index, (df.smoking_status == "Unknown").astype(int).values),
    "ever_smoker|known": (df[known].index, df[known].smoking_status.isin(["smokes", "formerly smoked"]).astype(int).values),
    "current_smoker|known": (df[known].index, (df[known].smoking_status == "smokes").astype(int).values),
}

def attack_auc(target, blocks, model="lr", seed=0):
    idx, y = TARGETS[target]
    if not blocks:
        return 0.5
    X = qi_matrix(df.loc[idx], blocks)
    clf = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
           if model == "lr" else HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)

ALL_QI = list(QI_BLOCKS)
print("\n=== H1c: skor sifreli, S tamamen sifreli; saldirgan yalnizca 6 QI'yi goruyor ===")
print("%-22s %8s %8s" % ("hedef", "LR-AUC", "GBM-AUC"))
for t in TARGETS:
    print("%-22s %8.3f %8.3f" % (t, attack_auc(t, ALL_QI, "lr"), attack_auc(t, ALL_QI, "gbm")))
# surekli glikoz icin R^2
Xg = qi_matrix(df, ALL_QI)
pg = cross_val_predict(make_pipeline(StandardScaler(), Ridge()), Xg, df.avg_glucose_level.values, cv=KFold(5, shuffle=True, random_state=0))
print("avg_glucose_level (surekli) R^2 (Ridge): %.3f" % r2_score(df.avg_glucose_level.values, pg))

# ---- H2: hangi ek QI gizlenmeli? (tum alt kumeler, ayrik 6 QI) ----
H2_TARGETS = ["hypertension", "heart_disease", "high_glucose(>=140)", "ever_smoker|known"]
def mean_leak(visible):
    return np.mean([attack_auc(t, visible, "lr") for t in H2_TARGETS])

print("\n=== H2: ek gizlenen QI sayisi m -> hedeflerin ortalama saldiri AUC'si (LR saldirgan) ===")
cache = {}
for m in range(0, 4):
    vals = []
    for hidden in itertools.combinations(ALL_QI, m):
        vis = [q for q in ALL_QI if q not in hidden]
        v = mean_leak(vis)
        cache[hidden] = v
        vals.append((v, hidden))
    vals.sort()
    arr = np.array([v for v, _ in vals])
    print(f"m={m}: manual/random ortalama={arr.mean():.3f}  en iyi={arr[0]:.3f} {vals[0][1]}  en kotu={arr[-1]:.3f} {vals[-1][1]}")

# greedy
hidden = []
for step in range(3):
    best = min((q for q in ALL_QI if q not in hidden), key=lambda q: mean_leak([x for x in ALL_QI if x not in hidden + [q]]))
    hidden.append(best)
    print(f"greedy adim {step+1}: gizle {best:15s} -> ortalama AUC {mean_leak([x for x in ALL_QI if x not in hidden]):.3f}")

# ---- H3: LR'ye karsi uyelik cikarimi (kayip esigi, Yeom tarzi) ----
print("\n=== H3: savunmasiz modelde uyelik cikarimi AUC (kayip esigi), 5 tohum ===")
def model_matrix(d):
    num = d[["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]].astype(float)
    cat = pd.get_dummies(d[["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]]).astype(float)
    return pd.concat([num, cat], axis=1).values
Xm, ym = model_matrix(df), df.stroke.values
for name, mk in [("LogReg(balanced)", lambda s: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))),
                 ("RandomForest(derin)", lambda s: RandomForestClassifier(n_estimators=200, random_state=s))]:
    aucs, tpr_at = [], []
    for s in range(5):
        rng = np.random.RandomState(s)
        mem = rng.rand(len(ym)) < 0.5
        clf = mk(s).fit(Xm[mem], ym[mem])
        p = np.clip(clf.predict_proba(Xm)[:, 1], 1e-7, 1 - 1e-7)
        loss = -(ym * np.log(p) + (1 - ym) * np.log(1 - p))
        a = roc_auc_score(mem.astype(int), -loss)
        aucs.append(a)
        # TPR @ FPR=1%
        thr = np.quantile(-loss[~mem], 0.99)
        tpr_at.append(((-loss[mem]) > thr).mean())
        util = roc_auc_score(ym[~mem], p[~mem])
    print(f"{name:20s} MIA-AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}  TPR@FPR1%={np.mean(tpr_at):.3f}  (model test AUC ~{util:.3f})")
