"""
Demonstrates why naive cross-subject pooling (without per-subject
normalization) is dangerous, and shows the fix.
"""
import numpy as np

d = np.load("/mnt/user-data/uploads/eeg_epochs_compact.npz", allow_pickle=True)
X = d["X"]; subj = np.array([str(s) for s in d["subject"]])
SUBS = sorted(set(subj))

print("Per-subject raw signal amplitude (std dev across all samples):")
sds = []
for s in SUBS:
    m = subj == s
    sd = X[m].std()
    sds.append(sd)
    print(f"  sub-{s}: SD = {sd:.2f}")
print(f"\nRange: {min(sds):.2f} to {max(sds):.2f}  -- a {max(sds)/min(sds):.1f}x spread")
print("""
THE FAILURE MODE: if you pool raw voltage across subjects and train ONE
classifier, and subject 4 has 5x the amplitude of subject 18, the classifier
can partly "solve" the task by just learning "big numbers = subject 4's
trials" -- which correlates with subject 4's OWN choice/accuracy base rates,
not with anything about the decision itself. That's subject-identity leakage
disguised as a real effect.

THE FIX WE ACTUALLY USED: our reported RQ1/RQ2 numbers are WITHIN-SUBJECT --
one classifier trained and tested per subject, then averaged. This sidesteps
the problem entirely, because amplitude differences between people never
enter a within-subject model at all.

THE FIX FOR TRUE POOLING (if you want one model across subjects): z-score
each subject's data using THAT SUBJECT's own mean/SD before pooling, so
everyone is on a comparable scale. Standard practice, sometimes called
"subject-wise normalization". Below: a quick check of what raw pooling vs
normalized pooling does to a trivial diagnostic.
""")

# Quick concrete illustration: can you decode WHICH SUBJECT a trial belongs
# to, just from raw amplitude? If yes, that's the leakage channel.
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
feat = X.std(axis=(1,2)).reshape(-1,1)  # single feature: trial's overall amplitude
labels = subj
# just check top-2 subjects for a quick binary illustration
s1, s2 = SUBS[3], SUBS[17]  # arbitrary pair with different SDs from above
m = np.isin(subj, [s1, s2])
y = (subj[m] == s1).astype(int)
score = cross_val_score(LogisticRegression(), feat[m], y, cv=5, scoring="roc_auc").mean()
print(f"Decoding SUBJECT IDENTITY (sub-{s1} vs sub-{s2}) from raw amplitude alone: AUC = {score:.3f}")
print("(if this is well above 0.5, raw amplitude alone reveals who the subject is --")
print(" exactly the leakage channel that pooling-without-normalization would exploit)")
