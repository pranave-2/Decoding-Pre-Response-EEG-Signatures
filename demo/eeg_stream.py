"""
eeg_stream.py — EEG data source for the demo.

Two modes, same interface, so the rest of the demo never needs to know which
one is running:

    source = EEGSource(simulate=True)    # no hardware, generates plausible EEG
    source = EEGSource(simulate=False)   # connects to the Mitsar via LSL

    source.start()
    chunk, tstamps = source.get_chunk()  # (n_samples, n_channels), (n_samples,)
    source.stop()

The simulator lets you build, test, and rehearse the whole demo with no device
attached, then flip one flag on the day.
"""

import time
import threading
import numpy as np


class EEGSource:
    def __init__(self, simulate=True, sfreq=250.0, n_channels=19,
                 stream_type="EEG", buffer_seconds=10.0):
        self.simulate = simulate
        self.sfreq = float(sfreq)
        self.n_channels = int(n_channels)
        self.stream_type = stream_type
        self.buffer_len = int(buffer_seconds * self.sfreq)

        self._buf = np.zeros((self.buffer_len, self.n_channels), dtype=np.float32)
        self._tbuf = np.zeros(self.buffer_len, dtype=np.float64)
        self._widx = 0
        self._total = 0
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._inlet = None
        self._t0 = None

    # ---------- connection ----------
    def start(self):
        if self.simulate:
            self._t0 = time.time()
            self._running = True
            self._thread = threading.Thread(target=self._sim_loop, daemon=True)
            self._thread.start()
            print(f"[EEGSource] SIMULATION: {self.n_channels}ch @ {self.sfreq}Hz")
        else:
            self._connect_lsl()
            self._running = True
            self._thread = threading.Thread(target=self._lsl_loop, daemon=True)
            self._thread.start()
        return self

    def _connect_lsl(self):
        from pylsl import resolve_streams, StreamInlet, proc_clocksync, proc_dejitter
        print(f"[EEGSource] Resolving LSL streams of type '{self.stream_type}' ...")
        streams = [s for s in resolve_streams(wait_time=3.0)
                   if s.type().upper() == self.stream_type.upper()]
        if not streams:
            # fall back to any stream, print what was found to help debugging
            allst = resolve_streams(wait_time=1.0)
            names = [(s.name(), s.type()) for s in allst]
            raise RuntimeError(
                f"No '{self.stream_type}' LSL stream found. Streams seen: {names}. "
                "On the Mitsar acquisition PC, make sure the LSL output/app is running."
            )
        info = streams[0]
        self._inlet = StreamInlet(info, max_buflen=int(self.buffer_len / self.sfreq) + 1,
                                  processing_flags=proc_clocksync | proc_dejitter)
        self.sfreq = info.nominal_srate() or self.sfreq
        self.n_channels = info.channel_count()
        self.buffer_len = int((self.buffer_len / len(self._buf)) * self.buffer_len) or self.buffer_len
        self._buf = np.zeros((self.buffer_len, self.n_channels), dtype=np.float32)
        self._tbuf = np.zeros(self.buffer_len, dtype=np.float64)
        print(f"[EEGSource] Connected to '{info.name()}': "
              f"{self.n_channels}ch @ {self.sfreq}Hz")

    # ---------- acquisition loops ----------
    def _push(self, samples, tstamps):
        with self._lock:
            for s, t in zip(samples, tstamps):
                self._buf[self._widx] = s[:self.n_channels]
                self._tbuf[self._widx] = t
                self._widx = (self._widx + 1) % self.buffer_len
                self._total += 1

    def _lsl_loop(self):
        while self._running:
            chunk, tstamps = self._inlet.pull_chunk(timeout=0.2, max_samples=64)
            if tstamps:
                self._push(np.asarray(chunk, dtype=np.float32), tstamps)
            else:
                time.sleep(0.005)

    def _sim_loop(self):
        """Generate pink-ish EEG with alpha, plus event-driven transients that
        the demo can inject (a CPP-like ramp, an error blip)."""
        step = 1.0 / self.sfreq
        next_t = time.time()
        phase = 0.0
        while self._running:
            now = time.time()
            n = max(1, int((now - next_t) * self.sfreq) + 1)
            n = min(n, 64)
            t_idx = np.arange(n)
            tt = phase + t_idx * step
            # base: 1/f-ish noise
            noise = np.cumsum(np.random.randn(n, self.n_channels), axis=0) * 0.4
            noise -= noise.mean(0)
            samples = (noise + np.random.randn(n, self.n_channels) * 3.0).astype(np.float32)

            # resting alpha: modest everywhere, LARGE over occipital when eyes closed.
            # occipital = last third of channels (O1/O2/Oz sit at the back).
            occ = np.zeros(self.n_channels); occ[self.n_channels * 2 // 3:] = 1.0
            base_alpha = 3.0 + (18.0 if self.sim_eyes_closed else 0.0)
            alpha = base_alpha * np.sin(2 * np.pi * 10.0 * tt)[:, None] * occ[None, :]
            samples += alpha.astype(np.float32)

            # SSVEP: a steady-state response at the attended flicker frequency,
            # strongest over occipital channels.
            if self.sim_ssvep_freq:
                ss = 10.0 * np.sin(2 * np.pi * self.sim_ssvep_freq * tt)[:, None] * occ[None, :]
                samples += ss.astype(np.float32)
            # inject any queued event signal
            samples = self._apply_events(samples, n)
            tstamps = next_t + t_idx * step
            self._push(samples, tstamps)
            phase += n * step
            next_t += n * step
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)

    # ---- simulator state the demo can drive, so rehearsal looks real ----
    # set these from the demo to make the simulated brain "respond"
    sim_eyes_closed = False     # True -> strong 10 Hz alpha over occipital ch
    sim_ssvep_freq = None       # e.g. 10.0 -> inject a 10 Hz steady-state response
    sim_ssvep_phase = 0.0

    def set_state(self, eyes_closed=None, ssvep_freq=None):
        if eyes_closed is not None:
            self.sim_eyes_closed = eyes_closed
        if ssvep_freq is not None:
            self.sim_ssvep_freq = ssvep_freq

    # event injection for the simulator ----------------------------------
    _event_queue = None

    def inject(self, kind):
        """Queue a synthetic neural event ('ramp' or 'error') for realism."""
        if not self.simulate:
            return
        if self._event_queue is None:
            self._event_queue = []
        # (kind, samples_remaining)
        dur = int(0.4 * self.sfreq)
        self._event_queue.append([kind, dur, dur])

    def _apply_events(self, samples, n):
        if not self._event_queue:
            return samples
        keep = []
        for ev in self._event_queue:
            kind, total, rem = ev
            k = min(n, rem)
            done = total - rem
            idx = np.arange(k)
            if kind == "ramp":
                # slow centro-parietal positivity-like ramp
                amp = (done + idx) / total * 12.0
                samples[:k, self.n_channels // 2:] += amp[:, None].astype(np.float32)
            elif kind == "error":
                # sharp fronto-central deflection ~ ERN/Pe
                amp = 15.0 * np.sin(np.pi * (done + idx) / total)
                samples[:k, :self.n_channels // 3] -= amp[:, None].astype(np.float32)
            ev[2] -= k
            if ev[2] > 0:
                keep.append(ev)
        self._event_queue = keep
        return samples

    # ---------- read ----------
    def get_chunk(self, seconds=None):
        """Return the most recent `seconds` of data as (n_samples, n_channels)."""
        with self._lock:
            n = self.buffer_len if seconds is None else int(seconds * self.sfreq)
            n = min(n, self.buffer_len, self._total)
            if n == 0:
                return (np.zeros((0, self.n_channels), np.float32),
                        np.zeros(0, np.float64))
            idx = (self._widx - n + np.arange(n)) % self.buffer_len
            return self._buf[idx].copy(), self._tbuf[idx].copy()

    def latest_timestamp(self):
        with self._lock:
            if self._total == 0:
                return None
            return self._tbuf[(self._widx - 1) % self.buffer_len]

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._inlet is not None:
            try:
                self._inlet.close_stream()
            except Exception:
                pass
        print("[EEGSource] stopped")


if __name__ == "__main__":
    # quick self-test of the simulator
    src = EEGSource(simulate=True).start()
    time.sleep(1.0)
    chunk, ts = src.get_chunk(seconds=0.5)
    print("got chunk:", chunk.shape, "ts span:",
          None if len(ts) == 0 else round(ts[-1] - ts[0], 3), "s")
    src.inject("ramp")
    time.sleep(0.5)
    chunk, ts = src.get_chunk(seconds=0.5)
    print("after ramp inject, max amp:", round(float(np.abs(chunk).max()), 1))
    src.stop()
