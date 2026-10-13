"""
Compares fixed-width time bins against PCA (data-driven dimensionality
reduction) as an alternative to uniform averaging.
PCA is fit INSIDE each training fold only -- fitting it on all data first
would leak test-set structure into the features, which is a classic mistake.
"""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

d = np.load("/mnt/user-data/uploads/eeg_epochs_compact.npz", allow_pickle=True)
X = d["X"]; subj = np.array([str(s) for s in d["subject"]])
choice = d["choice"]; accuracy = d["accuracy"]
SUBS = sorted(set(subj))

def fixed_bins(Xi, nb=3):
    n, c, t = Xi.shape
    e = np.linspace(0, t, nb + 1).astype(int)
    return np.stack([Xi[:, :, e[i]:e[i+1]].mean(2) for i in range(nb)], 2).reshape(n, -1)

def raw_flat(Xi):
    """No binning at all -- every raw timepoint as a feature, let PCA do the
    dimensionality reduction data-adaptively instead of us pre-deciding bins."""
    n, c, t = Xi.shape
    return Xi.reshape(n, c * t)

def cv_auc_with_pca(F_raw, y, n_components=15, nrep=3, seed=0):
    """PCA fit fresh inside each fold -- this is the leak-free version."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(nrep):
        skf = StratifiedKFold(5, shuffle=True, random_state=int(rng.integers(1e6)))
        oof = np.full(len(y), np.nan)
        for tr, te in skf.split(F_raw, y):
            if len(np.unique(y[tr])) < 2: continue
            pipe = make_pipeline(
                StandardScaler(),
                PCA(n_components=min(n_components, len(tr)-2)),
                LDA(solver="lsqr", shrinkage="auto"),
            )
            pipe.fit(F_raw[tr], y[tr])
            oof[te] = pipe.predict_proba(F_raw[te])[:, 1]
        ok = ~np.isnan(oof)
        if ok.sum() > 10:
            out.append(roc_auc_score(y[ok], oof[ok]))
    return np.mean(out) if out else np.nan

def cv_auc_fixed(F, y, nrep=3, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(nrep):
        skf = StratifiedKFold(5, shuffle=True, random_state=int(rng.integers(1e6)))
        oof = np.full(len(y), np.nan)
        for tr, te in skf.split(F, y):
            if len(np.unique(y[tr])) < 2: continue
            c = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
            c.fit(F[tr], y[tr]); oof[te] = c.predict_proba(F[te])[:, 1]
        ok = ~np.isnan(oof)
        if ok.sum() > 10:
            out.append(roc_auc_score(y[ok], oof[ok]))
    return np.mean(out) if out else np.nan

print("Comparing fixed-bin features vs PCA-on-raw-timepoints (leak-free), per subject")
print(f"{'sub':>5} {'fixed-3bin':>11} {'PCA-15comp':>11}")
fixed_scores, pca_scores = [], []
for s in SUBS[:8]:   # subset for speed -- remove [:8] to run all 24 yourself
    m = subj == s
    Ff = fixed_bins(X[m], nb=3)
    Fr = raw_flat(X[m])
    y = (choice[m] == 2).astype(int)
    a = cv_auc_fixed(Ff, y, seed=hash(s)%999)
    b = cv_auc_with_pca(Fr, y, seed=hash(s)%999)
    fixed_scores.append(a); pca_scores.append(b)
    print(f"{s:>5} {a:>11.4f} {b:>11.4f}")

print(f"\nMean (n={len(fixed_scores)} subjects tested): fixed-bin={np.mean(fixed_scores):.4f}  PCA={np.mean(pca_scores):.4f}")
