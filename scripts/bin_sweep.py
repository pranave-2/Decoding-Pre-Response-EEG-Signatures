"""
Bin-size sweep for RQ1 (choice) and RQ2 (accuracy) decoding.
Tests n_bins = 2,3,4,5,6,8,10,15,20 with IDENTICAL classifier/CV settings,
so the comparison is fair. Run yourself: python3 bin_sweep.py

WARNING: this takes a while (24 subjects x ~9 bin settings x 2 targets x
5 CV repeats). Reduce N_REPEATS or the bin list if you want it faster.
"""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

d = np.load("/mnt/user-data/uploads/eeg_epochs_compact.npz", allow_pickle=True)
X = d["X"]; subj = np.array([str(s) for s in d["subject"]])
choice = d["choice"]; accuracy = d["accuracy"]
SUBS = sorted(set(subj))

BIN_LIST = [2, 3, 4, 5, 6, 8, 10, 15, 20]
N_REPEATS = 5
N_FOLDS = 5

def feats(Xi, nb):
    n, c, t = Xi.shape
    e = np.linspace(0, t, nb + 1).astype(int)
    return np.stack([Xi[:, :, e[i]:e[i+1]].mean(2) for i in range(nb)], 2).reshape(n, -1)

def clf():
    return make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))

def cv_auc(F, y, nrep=N_REPEATS, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(nrep):
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=int(rng.integers(1e6)))
        oof = np.full(len(y), np.nan)
        for tr, te in skf.split(F, y):
            if len(np.unique(y[tr])) < 2:
                continue
            c = clf(); c.fit(F[tr], y[tr])
            oof[te] = c.predict_proba(F[te])[:, 1]
        ok = ~np.isnan(oof)
        if ok.sum() > 10:
            out.append(roc_auc_score(y[ok], oof[ok]))
    return np.mean(out) if out else np.nan

print(f"{'n_bins':>7} {'n_feat':>7} {'choice_AUC':>11} {'accuracy_AUC':>13}")
results = []
for nb in BIN_LIST:
    ch_scores, ac_scores = [], []
    for s in SUBS:
        m = subj == s
        F = feats(X[m], nb)
        ch_scores.append(cv_auc(F, (choice[m] == 2).astype(int), seed=hash((s,nb)) % 9999))
        ac_scores.append(cv_auc(F, accuracy[m].astype(int), seed=hash((s,nb,1)) % 9999))
    ch_mean = np.nanmean(ch_scores); ac_mean = np.nanmean(ac_scores)
    n_feat = X.shape[1] * nb
    results.append((nb, n_feat, ch_mean, ac_mean))
    print(f"{nb:>7} {n_feat:>7} {ch_mean:>11.4f} {ac_mean:>13.4f}")

print("\nBest choice AUC:", max(results, key=lambda r: r[2]))
print("Best accuracy AUC:", max(results, key=lambda r: r[3]))
