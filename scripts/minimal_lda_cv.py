"""
minimal_lda_cv.py — the simplest possible version of our decoding pipeline,
stripped of all the sweep/multiverse machinery, so you can read every line
and run it yourself.

Run: python3 minimal_lda_cv.py
"""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ---- 1. load one subject's data ----
d = np.load("eeg_epochs_compact.npz", allow_pickle=True)  # adjust path as needed
subj = np.array([str(s) for s in d["subject"]])
SUBJECT_TO_TEST = "01"
mask = subj == SUBJECT_TO_TEST

X = d["X"][mask]              # shape (n_trials, n_channels, n_timepoints)
choice = d["choice"][mask]    # 1 or 2
y = (choice == 2).astype(int) # binary label: 0 = choice 1, 1 = choice 2

print(f"Subject {SUBJECT_TO_TEST}: {X.shape[0]} trials, "
      f"{X.shape[1]} channels, {X.shape[2]} timepoints per trial")
print(f"Class balance: {int((y==0).sum())} vs {int((y==1).sum())}")

# ---- 2. turn each trial into a feature vector ----
# We average voltage within 3 equal time bins per channel, rather than using
# every raw timepoint (that would be 58*75=4350 features for ~300 trials --
# far too many; the classifier would overfit badly).
N_BINS = 3

def make_features(X, n_bins):
    n_trials, n_ch, n_t = X.shape
    edges = np.linspace(0, n_t, n_bins + 1).astype(int)
    bins = [X[:, :, edges[i]:edges[i+1]].mean(axis=2) for i in range(n_bins)]
    return np.concatenate(bins, axis=1)   # (n_trials, n_ch * n_bins)

F = make_features(X, N_BINS)
print(f"Feature vector length per trial: {F.shape[1]}")

# ---- 3. 5-fold cross-validation ----
# Stratified = each fold keeps the same class balance as the whole dataset.
# We do this 5 times with different random splits and average, since a
# single split can be lucky or unlucky with only ~300 trials.
N_FOLDS = 5
N_REPEATS = 5

all_true, all_pred = [], []
aucs = []

for repeat in range(N_REPEATS):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=repeat)
    fold_true, fold_pred = [], []

    for train_idx, test_idx in skf.split(F, y):
        X_train, X_test = F[train_idx], F[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # StandardScaler + LDA in a pipeline: the scaler is FIT ONLY on
        # training data, then applied to test data. This is what prevents
        # leakage -- the test set never influences how features are scaled.
        model = make_pipeline(
            StandardScaler(),
            LDA(solver="lsqr", shrinkage="auto"),  # shrinkage handles
                                                     # trials < features
        )
        model.fit(X_train, y_train)

        # predict_proba gives P(class=1) per trial; we need this (not just
        # the hard 0/1 prediction) to compute AUC, which cares about ranking
        probs = model.predict_proba(X_test)[:, 1]

        fold_true.extend(y_test)
        fold_pred.extend(probs)

    auc = roc_auc_score(fold_true, fold_pred)
    aucs.append(auc)
    print(f"  Repeat {repeat+1}: AUC = {auc:.4f}")

print(f"\nMean AUC across {N_REPEATS} repeats: {np.mean(aucs):.4f} "
      f"(SD {np.std(aucs):.4f})")

# ---- 4. permutation test: is this AUC actually meaningful? ----
# Shuffle the labels randomly and rerun the EXACT same pipeline. If the real
# AUC isn't clearly higher than this shuffled-label baseline, the result
# could just be noise / an artifact of the pipeline itself.
print("\nRunning permutation test (50 shuffles)...")
rng = np.random.default_rng(0)
null_aucs = []
for i in range(50):
    y_shuffled = rng.permutation(y)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=i)
    fold_true, fold_pred = [], []
    for train_idx, test_idx in skf.split(F, y_shuffled):
        model = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
        model.fit(F[train_idx], y_shuffled[train_idx])
        probs = model.predict_proba(F[test_idx])[:, 1]
        fold_true.extend(y_shuffled[test_idx])
        fold_pred.extend(probs)
    null_aucs.append(roc_auc_score(fold_true, fold_pred))

null_aucs = np.array(null_aucs)
real_auc = np.mean(aucs)
p_value = (1 + np.sum(null_aucs >= real_auc)) / (1 + len(null_aucs))

print(f"Null distribution (shuffled labels): mean = {null_aucs.mean():.4f}, "
      f"should be close to 0.5")
print(f"Real AUC ({real_auc:.4f}) exceeds {int((null_aucs < real_auc).sum())}/50 "
      f"null-shuffle AUCs")
print(f"Empirical p-value: {p_value:.4f}")
