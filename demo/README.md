# Live EEG Demo — Setup & Run Guide

Two acts, both working, both requiring **no accessories** beyond the amplifier,
electrode cap, and your laptop:

- **Act 1 — Alpha opener.** A live gauge that rises when the subject closes
  their eyes. Guaranteed to work on anyone; your safe opening win.
- **Act 2 — SSVEP attention BCI.** Two flickering boxes; the computer reads
  which one the subject is attending to, from their occipital EEG. ~85–95%
  reliable, feels like mind-reading.

---

## 1. Install (once)

```bash
pip install pylsl pygame numpy
```

That's all. `pylsl` needs the liblsl binary; on most systems `pip install pylsl`
bundles it. If it complains about a missing library, install liblsl:
- Windows: it's included in the pylsl wheel; if not, download from the LSL releases page.
- Linux: `conda install -c conda-forge liblsl` or build from source.
- macOS: `brew install labstreaminglayer/tap/lsl`.

## 2. Rehearse WITHOUT the headset (do this first)

```bash
python demo.py
```

This runs in **simulation mode** — a synthetic brain so you can learn the flow
and rehearse your patter with no hardware:

- Press **1** for the alpha act. Press **C** to calibrate, then **hold S** to
  fake "eyes closed" and watch the bar jump.
- Press **2** for the SSVEP act. Press **SPACE**, then **hold LEFT or RIGHT** to
  fake attending to that box; after 4 s it announces the choice.
- **ESC** returns to the menu.

Rehearse until the flow is second nature. On the day you'll do the same key
presses, minus the fake-input keys.

## 3. Go live with the Mitsar

**On the Mitsar acquisition PC:** start EEGStudio (or the SmartBCI app) and turn
on its **LSL output** so the amplifier streams an `EEG` type stream onto the
network. Both PCs must be on the same network (or run everything on one PC).

**Then:**

```bash
python demo.py --live
```

It resolves the LSL `EEG` stream automatically, adopts its true channel count
and sampling rate, and everything else is identical to rehearsal.

## 4. Electrode priorities

Both acts read **occipital** channels (back of the head): **O1, O2, Oz**, and
nearby **P3/P4/Pz** help. If you're short on time prepping the cap, get those
back-of-head electrodes clean first — impedance low, good gel/contact. The demo
uses the last third of the channel list as "occipital"; if your montage orders
channels differently, tell me and I'll set the exact indices for O1/O2/Oz.

## 5. Running each act live

**Alpha (Act 1):**
1. Subject sits still, eyes open, relaxed.
2. Press **C** to capture their eyes-open baseline (2 s — tell them to hold still).
3. Ask them to close their eyes. The bar should climb within a second or two.
4. Open eyes → it falls. This is your reliable crowd-warmer.

**SSVEP (Act 2):**
1. Subject looks at the screen ~60 cm away.
2. Tell them to **fix their gaze on one box and keep staring**.
3. Press **SPACE** *after* they're already staring (the reader uses the last 4 s).
4. The progress bar fills over 4 s; then it announces LEFT or RIGHT.
5. For a stronger effect, dim the room and have them really lock onto one box.

## 6. If SSVEP accuracy is low live

- Increase `DECISION_SECONDS` in `demo.py` (e.g. 4 → 6). Longer stare = cleaner.
- Make sure the two frequencies divide your refresh rate. On 60 Hz, 10 and 15 Hz
  are exact. On a 120 Hz screen you can use 10/12/15/20. Set `REFRESH` correctly.
- Check occipital electrode contact first — this matters more than anything.
- Room lighting lower, boxes bigger, subject closer.

## 7. Files

- `eeg_stream.py` — LSL/simulator data source (same interface for both).
- `spectral.py` — alpha index + SSVEP decoders (PSD and CCA).
- `demo.py` — the pygame app with both acts.

## 8. Known UX detail

The SSVEP reader analyzes the **last 4 seconds before it finishes**, so the
subject must be staring *before and during* the countdown, not after pressing
SPACE. Say "keep staring" while the bar fills.
