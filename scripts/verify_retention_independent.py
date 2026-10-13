"""
FULLY INDEPENDENT verification of the 99.5% retention claim.
Uses only the raw BIDS events.tsv files from ds002739 (re-clone fresh),
with NO dependency on any prior extraction or Colab output.

Run: python3 verify_retention_independent.py
Needs: git clone https://github.com/OpenNeuroDatasets/ds002739.git  (~40MB, tsv only)
"""
import glob, os, csv
import numpy as np

REPO = "ds002739"   # path to the cloned repo

WINDOW_MS = 300
BUFFER_MS = 150
MIN_RT = WINDOW_MS + BUFFER_MS

def read_rts(tsv_path):
    rts = []
    with open(tsv_path, newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        for row in reader:
            if len(row) < 4:
                continue
            if row[3].strip().rstrip("\r") != "dot_stim_validtrials":
                continue
            try:
                rts.append(float(row[1]) * 1000.0)   # duration column = RT, sec->ms
            except ValueError:
                continue
    return np.array(rts)

files = sorted(glob.glob(os.path.join(REPO, "sub-*", "func", "*_events.tsv")))
print(f"Found {len(files)} run files")

all_rt = []
for f in files:
    all_rt.append(read_rts(f))
all_rt = np.concatenate(all_rt)

n_total = len(all_rt)
n_kept = int((all_rt >= MIN_RT).sum())
print(f"\nTotal valid trials across all subjects: {n_total}")
print(f"Trials with RT >= {MIN_RT}ms (fit the 300/150 window): {n_kept}")
print(f"Retention: {100*n_kept/n_total:.2f}%")
print(f"\n(Compare to paper's claim: 7341/7377 = 99.51%)")
