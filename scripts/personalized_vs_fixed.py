"""
Matched comparison: fixed-300ms-window vs personalized-per-subject-window
single-trial LDA decoding, for RQ1 (choice) and RQ2 (accuracy).

Feature count is held FIXED at 3 time-bins per subject in both conditions
(channels x 3), specifically to avoid conflating "longer window = more
temporal resolution = more features = overfitting headroom" with the actual
question: does giving each subject their own RT-feasible window (instead of
one global 300ms window sized to the fastest subject) change decodability.

Method (same as minimal_lda_cv.py, reused for continuity):
  - 3 time-bins -> mean amplitude per channel per bin -> flatten to a feature
    vector of length n_channels_this_subject * 3.
  - StandardScaler (fit train-fold only) + LDA(shrinkage='auto', solver='lsqr')
  - 5-fold stratified CV, repeated 5x with different shuffles -> mean AUC
  - 50-shuffle permutation test (label shuffle, same CV scheme) -> empirical p
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

RNG_SEED = 0
N_REPEATS = 5
N_FOLDS = 5
N_PERM = 20
N_BINS = 3

def bin_features(X_trials, n_bins=N_BINS):
    """X_trials: (n_trials, n_channels, n_samples) possibly with NaN rows/cols
    (NaN channels = unused channel slots, NaN leading samples = outside
    this subject's personalized window). Returns (n_trials, n_valid_channels*n_bins).
    Assumes NaN pattern is IDENTICAL across all trials of a given call
    (true here: fixed per-subject channel count and window length)."""
    n_trials, n_channels, n_samples = X_trials.shape
    # valid channels: not NaN anywhere in any trial, any sample
    chan_valid = ~np.isnan(X_trials).all(axis=(0, 2))
    Xc = X_trials[:, chan_valid, :]
    # valid time samples: not NaN anywhere in any trial, any channel
    time_valid = ~np.isnan(Xc).all(axis=(0, 1))
    Xc = Xc[:, :, time_valid]
    n_valid_samples = Xc.shape[2]
    edges = np.linspace(0, n_valid_samples, n_bins + 1).astype(int)
    bins = []
    for b in range(n_bins):
        seg = Xc[:, :, edges[b]:edges[b+1]]
        bins.append(seg.mean(axis=2))
    feat = np.concatenate(bins, axis=1)  # (n_trials, n_valid_channels*n_bins)
    return feat

def run_subject(X_sub, y, seed_offset=0):
    """Returns dict with mean_auc, std_auc, perm_p, n_trials, n_features."""
    feat = bin_features(X_sub)
    n_trials, n_features = feat.shape
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < N_FOLDS:
        return dict(mean_auc=np.nan, std_auc=np.nan, perm_p=np.nan,
                    n_trials=n_trials, n_features=n_features, skipped=True)

    def cv_auc(X_, y_, seed):
        aucs = []
        for r in range(N_REPEATS):
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                   random_state=seed * 1000 + r)
            for train_idx, test_idx in skf.split(X_, y_):
                sc = StandardScaler().fit(X_[train_idx])
                Xtr = sc.transform(X_[train_idx])
                Xte = sc.transform(X_[test_idx])
                clf = LDA(solver="lsqr", shrinkage="auto")
                clf.fit(Xtr, y_[train_idx])
                p = clf.predict_proba(Xte)[:, 1]
                if len(np.unique(y_[test_idx])) == 2:
                    aucs.append(roc_auc_score(y_[test_idx], p))
        return np.array(aucs)

    real_aucs = cv_auc(feat, y, RNG_SEED + seed_offset)
    mean_auc = real_aucs.mean()
    std_auc = real_aucs.std()

    rng = np.random.RandomState(RNG_SEED + seed_offset + 999)
    perm_means = []
    for p_i in range(N_PERM):
        y_perm = rng.permutation(y)
        perm_aucs = cv_auc(feat, y_perm, RNG_SEED + seed_offset + 1 + p_i)
        perm_means.append(perm_aucs.mean())
    perm_means = np.array(perm_means)
    perm_p = (np.sum(perm_means >= mean_auc) + 1) / (N_PERM + 1)

    return dict(mean_auc=mean_auc, std_auc=std_auc, perm_p=perm_p,
                n_trials=n_trials, n_features=n_features, skipped=False)


if __name__ == "__main__":
    import sys
    print("Loading data...")

    # ---- fixed-window (300ms) dataset ----
    fixed = np.load("/home/claude/work/compact/eeg_epochs_compact.npz", allow_pickle=True)
    Xf = fixed["X"]  # (7341, 58, 75), no NaNs
    subj_f = fixed["subject"]
    choice_f = fixed["choice"]
    acc_f = fixed["accuracy"]

    # ---- personalized-window dataset ----
    Xp = np.load("/home/claude/work/personalized/X.npy")  # (7377, 64, 200)
    subj_p = np.load("/home/claude/work/personalized/subject.npy")
    choice_p = np.load("/home/claude/work/personalized/choice.npy")
    acc_p = np.load("/home/claude/work/personalized/accuracy.npy")

    subjects = sorted(np.unique(subj_f))
    print(f"{len(subjects)} subjects\n")

    results = []
    for s in subjects:
        mf = subj_f == s
        mp = subj_p == s
        yf_choice = (choice_f[mf] == 2).astype(int)
        yf_acc = acc_f[mf].astype(int)
        yp_choice = (choice_p[mp] == 2).astype(int)
        yp_acc = acc_p[mp].astype(int)

        r_fixed_choice = run_subject(Xf[mf], yf_choice, seed_offset=0)
        r_fixed_acc = run_subject(Xf[mf], yf_acc, seed_offset=100)
        r_pers_choice = run_subject(Xp[mp], yp_choice, seed_offset=200)
        r_pers_acc = run_subject(Xp[mp], yp_acc, seed_offset=300)

        row = dict(subject=s,
                    fixed_choice=r_fixed_choice, pers_choice=r_pers_choice,
                    fixed_acc=r_fixed_acc, pers_acc=r_pers_acc)
        results.append(row)
        print(f"sub-{s}: "
              f"CHOICE fixed={r_fixed_choice['mean_auc']:.3f}(n={r_fixed_choice['n_trials']},f={r_fixed_choice['n_features']}) "
              f"pers={r_pers_choice['mean_auc']:.3f}(n={r_pers_choice['n_trials']},f={r_pers_choice['n_features']})  |  "
              f"ACC fixed={r_fixed_acc['mean_auc']:.3f} pers={r_pers_acc['mean_auc']:.3f}")
        sys.stdout.flush()

    import pickle
    with open("/tmp/claude-0/-home-claude/dc50dea1-fc1c-51f9-9d55-a537a3eaf5ff/scratchpad/personalized_vs_fixed_results.pkl", "wb") as fh:
        pickle.dump(results, fh)
    print("\nSaved results.")
