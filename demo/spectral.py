"""
spectral.py — the analysis core shared by both demo acts.

Two jobs:
  1. band_power()   -> power in a frequency band (drives the alpha opener)
  2. ssvep_decide() -> which flicker frequency the brain is locked to
                       (drives the attention BCI)

The SSVEP decision uses two methods and you can pick either:
  - "psd" : compare narrowband power at each candidate frequency (+harmonic).
            Simple, transparent, great for explaining on stage.
  - "cca" : canonical correlation against sine/cosine templates. This is the
            standard high-accuracy SSVEP method used in real BCIs.
"""

import numpy as np


def _detrend(x):
    return x - x.mean(axis=0, keepdims=True)


def band_power(x, sfreq, lo, hi, occ_idx=None):
    """Mean power in [lo, hi) Hz. x: (n_samples, n_channels)."""
    if x.shape[0] < 32:
        return 0.0
    if occ_idx is not None:
        x = x[:, occ_idx]
    x = _detrend(x)
    win = np.hanning(x.shape[0])[:, None]
    f = np.fft.rfftfreq(x.shape[0], 1.0 / sfreq)
    P = (np.abs(np.fft.rfft(x * win, axis=0)) ** 2).mean(axis=1)
    sel = (f >= lo) & (f < hi)
    return float(P[sel].mean()) if sel.any() else 0.0


def alpha_index(x, sfreq, occ_idx=None):
    """Relative alpha: alpha power / total 2-40 Hz power. Robust to overall
    amplitude, so the opener gauge behaves the same across people."""
    a = band_power(x, sfreq, 8, 13, occ_idx)
    total = band_power(x, sfreq, 2, 40, occ_idx)
    return float(a / total) if total > 0 else 0.0


def _narrowband_power(x, sfreq, freq, half=0.4, n_harmonics=2):
    """Summed power at freq and its harmonics, in narrow bands."""
    p = 0.0
    for h in range(1, n_harmonics + 1):
        c = freq * h
        p += band_power(x, sfreq, c - half, c + half)
    return p


def ssvep_decide_psd(x, sfreq, freqs, occ_idx=None):
    """Return (best_freq, scores dict) using narrowband power."""
    if occ_idx is not None:
        x = x[:, occ_idx]
    scores = {f: _narrowband_power(x, sfreq, f) for f in freqs}
    # normalize to a soft confidence
    vals = np.array(list(scores.values()))
    best = max(scores, key=scores.get)
    conf = float(vals.max() / vals.sum()) if vals.sum() > 0 else 0.0
    return best, scores, conf


def _cca_corr(X, Y):
    """First canonical correlation between multivariate X and Y."""
    Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    qx, _ = np.linalg.qr(Xc)
    qy, _ = np.linalg.qr(Yc)
    # singular values of qx^T qy are the canonical correlations
    s = np.linalg.svd(qx.T @ qy, compute_uv=False)
    return float(s[0]) if len(s) else 0.0


def ssvep_decide_cca(x, sfreq, freqs, n_harmonics=2, occ_idx=None):
    """Canonical correlation analysis: correlate the multichannel EEG against
    sine/cosine templates at each candidate frequency. Standard SSVEP method."""
    if occ_idx is not None:
        x = x[:, occ_idx]
    x = _detrend(x)
    n = x.shape[0]
    t = np.arange(n) / sfreq
    scores = {}
    for f in freqs:
        cols = []
        for h in range(1, n_harmonics + 1):
            cols.append(np.sin(2 * np.pi * f * h * t))
            cols.append(np.cos(2 * np.pi * f * h * t))
        Y = np.column_stack(cols)
        scores[f] = _cca_corr(x, Y)
    best = max(scores, key=scores.get)
    vals = np.array(list(scores.values()))
    conf = float(vals.max() / vals.sum()) if vals.sum() > 0 else 0.0
    return best, scores, conf


def ssvep_decide(x, sfreq, freqs, method="cca", occ_idx=None):
    if method == "psd":
        return ssvep_decide_psd(x, sfreq, freqs, occ_idx=occ_idx)
    return ssvep_decide_cca(x, sfreq, freqs, occ_idx=occ_idx)


if __name__ == "__main__":
    # Validate against the simulator: does the classifier recover the
    # frequency we injected?
    from eeg_stream import EEGSource
    import time

    FREQS = [10.0, 15.0]
    src = EEGSource(simulate=True, sfreq=250, n_channels=19).start()
    occ = list(range(19 * 2 // 3, 19))
    time.sleep(0.5)

    for target in FREQS:
        src.set_state(ssvep_freq=target)
        time.sleep(2.5)                      # let the buffer fill with the response
        x, _ = src.get_chunk(seconds=2.0)
        for method in ["psd", "cca"]:
            best, scores, conf = ssvep_decide(x, src.sfreq, FREQS, method=method, occ_idx=occ)
            ok = "OK" if abs(best - target) < 0.01 else "WRONG"
            sc = {f: round(v, 3) for f, v in scores.items()}
            print(f"target={target:>4}Hz  {method}: decoded={best:>4}Hz  "
                  f"conf={conf:.2f}  {sc}  [{ok}]")
        src.set_state(ssvep_freq=None)
        time.sleep(1.0)
    src.stop()
