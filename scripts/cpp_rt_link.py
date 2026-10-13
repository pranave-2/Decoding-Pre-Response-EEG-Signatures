"""
The real CPP has a KNOWN property: its build-up rate/amplitude relates to
response time (Kelly & O'Connell 2013). We never tested whether OUR signal
has this property. Here we test it, honestly, as a first mechanistic check
(not full proof, but real evidence either way).
"""
import numpy as np
from scipy import stats

d = np.load("/mnt/user-data/uploads/eeg_epochs_compact.npz", allow_pickle=True)
X = d["X"]; subj = np.array([str(s) for s in d["subject"]]); RT = d["RT"]
SUBS = sorted(set(subj))

# our "decision signal proxy": mean amplitude in the LAST bin (closest to
# response) at posterior channels -- the same slow-drift feature that
# drove decoding
n_ch = X.shape[1]
posterior = slice(n_ch//2, n_ch)

print("Does our decision-signal proxy amplitude correlate with RT, per subject?")
print("(Real CPP: steeper/larger amplitude -> typically FASTER RT)")
print(f"{'sub':>5} {'r':>7} {'p':>7}")
rs = []
for s in SUBS:
    m = subj == s
    sig = X[m][:, posterior, -15:].mean(axis=(1,2))  # last ~60ms, posterior chans
    r, p = stats.pearsonr(sig, RT[m])
    rs.append(r)
    print(f"{s:>5} {r:>7.3f} {p:>7.3f}")

rs = np.array(rs)
t, p = stats.ttest_1samp(rs, 0)
print(f"\nMean correlation across subjects: r={rs.mean():.3f}, t({len(rs)-1})={t:.2f}, p={p:.4f}")
print(f"Subjects with negative r (signal UP -> RT DOWN, as CPP predicts): {(rs<0).sum()}/{len(rs)}")
print("""
INTERPRETATION:
If this came back with a reliable negative correlation (higher signal ->
faster response) consistent across subjects, that WOULD be genuine evidence
for an accumulation-like mechanism, not just "the classifier found something".
""")
