"""
demo.py — the live EEG demo (Act 1: alpha opener, Act 2: SSVEP attention BCI).

Run:
    python demo.py                 # simulation mode (no hardware, for rehearsal)
    python demo.py --live          # connect to the Mitsar via LSL

Keys:
    1  -> Act 1: eyes-open / eyes-closed alpha gauge
    2  -> Act 2: SSVEP attention BCI (two flickering boxes)
    SPACE (in Act 2) -> run one decision
    C  -> calibrate baseline (Act 1)
    S  (simulation only) -> toggle "eyes closed" to fake the effect while rehearsing
    LEFT/RIGHT (simulation only, Act 2) -> fake attending to a box
    ESC -> back to menu / quit

Flicker frequencies are chosen to divide the monitor refresh rate cleanly.
On a 60 Hz monitor, 10 Hz (6 frames) and 15 Hz (4 frames) are exact.
"""

import sys
import time
import numpy as np
import pygame

from eeg_stream import EEGSource
from spectral import alpha_index, ssvep_decide

# ---------------- configuration ----------------
LIVE = "--live" in sys.argv
REFRESH = 60                       # assumed monitor refresh (Hz)
SSVEP_FREQS = [10.0, 15.0]         # 6 and 4 frames at 60 Hz -> exact
DECISION_SECONDS = 4.0             # how long to accumulate before deciding
OCC_FRACTION = (2, 3)              # occipital channels = last third

W, H = 1100, 720
BG = (12, 14, 20)
FG = (235, 238, 245)
ACCENT = (90, 200, 160)
WARN = (235, 170, 80)
DIM = (120, 128, 140)


def occ_indices(n_channels):
    a, b = OCC_FRACTION
    return list(range(n_channels * a // b, n_channels))


class Demo:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("EEG Live Demo")
        self.clock = pygame.time.Clock()
        self.font_xl = pygame.font.SysFont("Arial", 64, bold=True)
        self.font_l = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_m = pygame.font.SysFont("Arial", 26)
        self.font_s = pygame.font.SysFont("Arial", 18)

        self.src = EEGSource(simulate=not LIVE).start()
        time.sleep(1.0)
        self.occ = occ_indices(self.src.n_channels)

        self.mode = "menu"
        self.alpha_baseline = None
        self.frame = 0
        # Act 2 state
        self.decide_active = False
        self.decide_start = None
        self.last_result = None
        self.running = True

    # ---------------- shared drawing ----------------
    def text(self, s, font, color, x, y, center=True):
        surf = font.render(s, True, color)
        r = surf.get_rect()
        if center:
            r.center = (x, y)
        else:
            r.topleft = (x, y)
        self.screen.blit(surf, r)

    # ---------------- menu ----------------
    def draw_menu(self):
        self.screen.fill(BG)
        self.text("Live EEG Demo", self.font_xl, FG, W // 2, 150)
        mode_str = "LIVE (Mitsar via LSL)" if LIVE else "SIMULATION (rehearsal)"
        self.text(mode_str, self.font_m, ACCENT if LIVE else WARN, W // 2, 210)
        self.text("1   —   Alpha opener  (eyes open vs closed)", self.font_l, FG, W // 2, 330)
        self.text("2   —   SSVEP attention BCI  (mind-reading)", self.font_l, FG, W // 2, 400)
        self.text(f"channels: {self.src.n_channels}    sfreq: {self.src.sfreq:.0f} Hz    "
                  f"occipital idx: {self.occ[0]}-{self.occ[-1]}",
                  self.font_s, DIM, W // 2, 520)
        self.text("ESC to quit", self.font_s, DIM, W // 2, 680)

    # ---------------- Act 1: alpha ----------------
    def draw_alpha(self):
        self.screen.fill(BG)
        x, _ = self.src.get_chunk(seconds=2.0)
        ai = alpha_index(x, self.src.sfreq, self.occ) if x.shape[0] > 64 else 0.0

        if self.alpha_baseline is None:
            level = 0.5
            hint = "Press C to calibrate your eyes-OPEN baseline"
        else:
            # ratio above baseline -> 0..1 gauge
            level = np.clip((ai - self.alpha_baseline) /
                            (self.alpha_baseline * 1.2 + 1e-6), 0, 1)
            hint = "Close your eyes → watch the bar rise"

        self.text("Act 1 — Alpha rhythm", self.font_l, FG, W // 2, 70)
        self.text("The back of your brain bursts into a 10 Hz rhythm when your eyes close.",
                  self.font_s, DIM, W // 2, 115)

        # gauge
        bar_w, bar_h = 160, 380
        bx, by = W // 2 - bar_w // 2, 170
        pygame.draw.rect(self.screen, (30, 34, 44), (bx, by, bar_w, bar_h), border_radius=12)
        fill_h = int(bar_h * level)
        col = ACCENT if level > 0.5 else (80, 140, 200)
        pygame.draw.rect(self.screen, col,
                         (bx, by + bar_h - fill_h, bar_w, fill_h), border_radius=12)
        self.text(f"{level*100:4.0f}%", self.font_l, FG, W // 2, by + bar_h + 45)
        self.text(f"alpha index: {ai:.2f}", self.font_s, DIM, W // 2, by + bar_h + 80)
        self.text(hint, self.font_m, WARN, W // 2, 640)
        if not LIVE:
            self.text("[sim] hold S to fake eyes-closed", self.font_s, DIM, W // 2, 690)

    def calibrate_alpha(self):
        vals = []
        t0 = time.time()
        while time.time() - t0 < 2.0:
            x, _ = self.src.get_chunk(seconds=2.0)
            if x.shape[0] > 64:
                vals.append(alpha_index(x, self.src.sfreq, self.occ))
            time.sleep(0.1)
        if vals:
            self.alpha_baseline = float(np.median(vals))

    # ---------------- Act 2: SSVEP ----------------
    def draw_ssvep(self):
        self.screen.fill(BG)
        self.text("Act 2 — Attention BCI", self.font_l, FG, W // 2, 55)
        self.text("Stare at one flickering box. The computer reads which one from your brain.",
                  self.font_s, DIM, W // 2, 95)

        # two flickering boxes; flicker by toggling visibility on frame schedule
        boxes = [(-1, SSVEP_FREQS[0], W // 2 - 320, "LEFT"),
                 (+1, SSVEP_FREQS[1], W // 2 + 120, "RIGHT")]
        size = 200
        top = 180
        for sign, freq, bx, label in boxes:
            frames_per_cycle = REFRESH / freq
            on = (self.frame % frames_per_cycle) < (frames_per_cycle / 2)
            color = FG if on else (35, 38, 48)
            pygame.draw.rect(self.screen, color, (bx, top, size, size), border_radius=8)
            self.text(f"{label}  {freq:.0f} Hz", self.font_s, DIM, bx + size // 2, top + size + 28)

        # decision progress / result
        if self.decide_active:
            elapsed = time.time() - self.decide_start
            prog = min(1.0, elapsed / DECISION_SECONDS)
            pygame.draw.rect(self.screen, (30, 34, 44), (W // 2 - 250, 470, 500, 26), border_radius=8)
            pygame.draw.rect(self.screen, ACCENT, (W // 2 - 250, 470, int(500 * prog), 26), border_radius=8)
            self.text("reading your brain...", self.font_m, WARN, W // 2, 530)
            if prog >= 1.0:
                self.run_decision()
        elif self.last_result is not None:
            freq, conf, scores = self.last_result
            side = "LEFT" if freq == SSVEP_FREQS[0] else "RIGHT"
            self.text(f"You were looking at:  {side}", self.font_l, ACCENT, W // 2, 500)
            self.text(f"({freq:.0f} Hz, confidence {conf*100:.0f}%)", self.font_s, DIM, W // 2, 545)
            self.text("SPACE to go again", self.font_s, DIM, W // 2, 600)
        else:
            self.text("SPACE to read your choice", self.font_m, WARN, W // 2, 520)

        if not LIVE:
            self.text("[sim] hold LEFT or RIGHT to fake attending", self.font_s, DIM, W // 2, 680)

    def run_decision(self):
        x, _ = self.src.get_chunk(seconds=DECISION_SECONDS)
        freq, scores, conf = ssvep_decide(x, self.src.sfreq, SSVEP_FREQS,
                                          method="cca", occ_idx=self.occ)
        self.last_result = (freq, conf, scores)
        self.decide_active = False
        if not LIVE:
            self.src.set_state(ssvep_freq=None)

    # ---------------- event / loop ----------------
    def handle_keys(self, event):
        if event.key == pygame.K_ESCAPE:
            if self.mode == "menu":
                self.running = False
            else:
                self.mode = "menu"
                self.last_result = None
                self.decide_active = False
                if not LIVE:
                    self.src.set_state(eyes_closed=False, ssvep_freq=None)
        elif self.mode == "menu":
            if event.key == pygame.K_1:
                self.mode = "alpha"
            elif event.key == pygame.K_2:
                self.mode = "ssvep"
        elif self.mode == "alpha":
            if event.key == pygame.K_c:
                self.calibrate_alpha()
        elif self.mode == "ssvep":
            if event.key == pygame.K_SPACE and not self.decide_active:
                self.decide_active = True
                self.decide_start = time.time()
                self.last_result = None

    def handle_held_keys(self):
        """Simulation-only shortcuts to fake the neural response for rehearsal."""
        if LIVE:
            return
        keys = pygame.key.get_pressed()
        if self.mode == "alpha":
            self.src.set_state(eyes_closed=keys[pygame.K_s])
        elif self.mode == "ssvep" and self.decide_active:
            if keys[pygame.K_LEFT]:
                self.src.set_state(ssvep_freq=SSVEP_FREQS[0])
            elif keys[pygame.K_RIGHT]:
                self.src.set_state(ssvep_freq=SSVEP_FREQS[1])

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self.handle_keys(event)
            self.handle_held_keys()

            if self.mode == "menu":
                self.draw_menu()
            elif self.mode == "alpha":
                self.draw_alpha()
            elif self.mode == "ssvep":
                self.draw_ssvep()

            pygame.display.flip()
            self.frame += 1
            self.clock.tick(REFRESH)

        self.src.stop()
        pygame.quit()


if __name__ == "__main__":
    Demo().run()
