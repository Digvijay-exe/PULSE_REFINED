"""
wristlift.py - wrist-IMU strength-training tracker (auto set detection + auto rep counting
+ optional exercise recognition). Garmin-style behaviour, built from the public literature
(RecoFit-style pipeline). Not Garmin's code.

Pipeline
  1. Data      : 3-axis accel (+ optional 3-axis gyro), 25-50 Hz, wrist-worn.
  2. Set detect: sliding window -> features -> active/idle -> hysteresis state machine.
  3. Rep count : low-pass -> PCA to 1-D -> autocorrelation period -> peak detection
                 -> consistency gate (>= 4 reps) -> back-fill (whole-set recount).
  4. Exercise  : random forest on set-level features (optional); Keras CNN builder provided.

Units: accel in m/s^2 or g (auto-detected), gyro in rad/s or deg/s (only used for periodicity).

Usage
  python wristlift.py --demo
  python wristlift.py recording.csv --fs 50        # columns: t,ax,ay,az[,gx,gy,gz]
Library
  det = WorkoutDetector(); events = det.push(acc_chunk, gyro_chunk); events += det.flush()
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, correlate, find_peaks, sosfiltfilt

G = 9.80665


# --------------------------------------------------------------------------- config
@dataclass
class Config:
    fs: float = 50.0
    # --- set detection
    win_s: float = 4.0            # window length
    hop_s: float = 1.0            # window hop
    start_windows: int = 3        # consecutive active windows to start a set
    end_idle_s: float = 7.0       # idle time to end a set (5-10 s)
    max_set_s: float = 600.0      # safety cap
    f_lo: float = 0.2             # lifting rep frequency band (Hz)
    f_hi: float = 1.3
    min_energy: float = 0.7       # m/s^2, rms of mean-removed accel in window
    min_conc: float = 0.35        # spectral concentration around dominant freq
    min_ac: float = 0.35          # autocorrelation peak strength
    # --- rep counting
    rep_lp_hz: float = 6.0
    min_rep_s: float = 0.8        # fastest rep
    max_rep_s: float = 6.0        # slowest rep
    prom_frac: float = 0.25       # peak prominence as fraction of signal amplitude
    min_reps: int = 4             # Garmin shows reps only after 4
    interval_tol: float = 0.40    # rep-interval consistency (+-40% of median)
    interval_frac: float = 0.70   # fraction of intervals that must be consistent


# --------------------------------------------------------------------------- helpers
def _lp(x: np.ndarray, fc: float, fs: float, order: int = 3) -> np.ndarray:
    x = np.asarray(x, float)
    if x.shape[0] < 40 or fc >= fs / 2:
        return x
    sos = butter(order, fc / (fs / 2), output="sos")
    return sosfiltfilt(sos, x, axis=0)


def _pca1(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[0]


def _autocorr(sig: np.ndarray) -> np.ndarray:
    n = len(sig)
    ac = correlate(sig, sig, mode="full", method="fft")[n - 1:]
    ac = ac / (ac[0] + 1e-12)
    return ac / np.maximum(1 - np.arange(n) / n, 0.3)  # mild bias correction


# --------------------------------------------------------------------------- stage 2: windows
def window_features(acc_w: np.ndarray, gyro_w: Optional[np.ndarray], fs: float, cfg: Config) -> np.ndarray:
    """-> [energy, dominant_freq, spectral_concentration, autocorr_peak, gyro_rms]"""
    x = _lp(acc_w, 6.0, fs)
    x = x - x.mean(0)
    energy = float(np.sqrt(x.var(0).sum()))
    sig = _pca1(x)
    n = len(sig)

    nfft = 1 << int(np.ceil(np.log2(n * 8)))
    sp = np.abs(np.fft.rfft(sig * np.hanning(n), nfft)) ** 2
    f = np.fft.rfftfreq(nfft, 1 / fs)
    wide = (f >= 0.1) & (f <= 4.0)      # search wide, then require the peak to be in the lifting band
    dom = float(f[np.argmax(np.where(wide, sp, 0))])
    tot = sp[(f >= 0.1) & (f <= 5.0)].sum() + 1e-12
    conc = float(sp[(f >= dom - 0.4) & (f <= dom + 0.4)].sum() / tot)

    ac = _autocorr(sig)
    lo, hi = int(0.5 * fs), n // 2
    acpk = 0.0
    if hi > lo + 2:
        pk, _ = find_peaks(ac[lo:hi])
        if len(pk):
            acpk = float(ac[lo:hi][pk].max())

    gr = 0.0 if gyro_w is None else float(np.sqrt((gyro_w - gyro_w.mean(0)).var(0).sum()))
    return np.array([energy, dom, conc, acpk, gr])


def rule_activity(feats: np.ndarray, cfg: Config) -> bool:
    energy, dom, conc, acpk, _ = feats
    return bool(energy >= cfg.min_energy and cfg.f_lo <= dom <= cfg.f_hi
                and (conc >= cfg.min_conc or acpk >= cfg.min_ac))


def fit_activity_model(acc_list, label_list, fs=50.0, cfg: Optional[Config] = None, gyro_list=None):
    """Optional: train a random forest replacing the threshold rule.
    label_list[i] is a per-sample 0/1 array (1 = exercising). Returns a callable(feats)->bool."""
    from sklearn.ensemble import RandomForestClassifier
    cfg = cfg or Config(fs=fs)
    w, h = int(cfg.win_s * fs), int(cfg.hop_s * fs)
    X, y = [], []
    for i, (acc, lab) in enumerate(zip(acc_list, label_list)):
        gy = None if gyro_list is None else gyro_list[i]
        for e in range(w, len(acc) + 1, h):
            X.append(window_features(acc[e - w:e], None if gy is None else gy[e - w:e], fs, cfg))
            y.append(int(np.mean(lab[e - w:e]) > 0.5))
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=0).fit(X, y)
    return lambda feats: bool(rf.predict(feats.reshape(1, -1))[0])


# --------------------------------------------------------------------------- stage 3: reps
@dataclass
class RepResult:
    count: int
    confirmed: bool
    peak_times: List[float] = field(default_factory=list)  # seconds from segment start
    period: float = 0.0
    strength: float = 0.0
    t_start: float = 0.0   # motion onset, seconds from segment start
    t_end: float = 0.0     # motion offset


def count_reps(acc: np.ndarray, gyro: Optional[np.ndarray], fs: float, cfg: Config) -> RepResult:
    n = len(acc)
    if n < int(4 * cfg.min_rep_s * fs):
        return RepResult(0, False)

    # candidate 1-D signals: accel PCA (keeps wrist orientation => "arm returns to start"),
    # and gyro PCA if present; pick the more periodic one.
    cands = [_pca1(_lp(acc, cfg.rep_lp_hz, fs))]
    if gyro is not None and np.any(gyro):
        cands.append(_pca1(_lp(gyro, cfg.rep_lp_hz, fs)))

    best = None
    for sig in cands:
        ac = _autocorr(sig)
        lo, hi = int(cfg.min_rep_s * fs), min(int(cfg.max_rep_s * fs), n // 2)
        if hi <= lo + 2:
            continue
        seg = ac[lo:hi]
        pk, _ = find_peaks(seg, height=cfg.min_ac)
        if not len(pk):
            continue
        h = seg[pk]
        k = np.where(h >= 0.75 * h.max())[0][0]      # first strong peak = fundamental
        lag, st = lo + int(pk[k]), float(h[k])
        if best is None or st > best[2]:
            best = (sig, lag, st)
    if best is None:
        return RepResult(0, False)

    sig, P, strength = best
    sm = gaussian_filter1d(sig, sigma=P / 8.0)

    # motion onset/offset from a lightly smoothed speed envelope
    fast = gaussian_filter1d(sig, sigma=0.1 * fs)
    env = gaussian_filter1d(np.abs(np.gradient(fast)), sigma=0.15 * fs)
    idx = np.where(env > 0.15 * np.percentile(env, 95))[0]
    if not len(idx):
        return RepResult(0, False)
    ta, tb = int(idx[0]), int(idx[-1])
    n_est = (tb - ta) / P                      # approx. number of rep cycles in the motion window

    # Extrema of both polarities inside the motion window. One polarity sits at the rest
    # posture (a fence-post: N+-1 of them), the other mid-rep (exactly N). Pick the polarity
    # whose count best matches duration / period.
    best_pk, best_err = np.array([], int), 1e9
    for pol in (1.0, -1.0):
        x = pol * sm
        amp = np.percentile(x, 90) - np.percentile(x, 10)
        pk, _ = find_peaks(x, distance=max(1, int(0.6 * P)), prominence=cfg.prom_frac * amp)
        inner = pk[(pk > ta) & (pk < tb)]
        err = abs(len(inner) - n_est)
        if err < best_err - 1e-9 or (abs(err - best_err) < 1e-9 and len(inner) > len(best_pk)):
            best_pk, best_err = inner, err
    pk = best_pk

    ok = False
    if len(pk) >= 2:
        iv = np.diff(pk) / fs
        med = np.median(iv)
        frac = float(np.mean(np.abs(iv - med) <= cfg.interval_tol * med))
        ok = len(pk) >= cfg.min_reps and frac >= cfg.interval_frac
    return RepResult(int(len(pk)), ok, (pk / fs).tolist(), P / fs, strength, ta / fs, tb / fs)


# --------------------------------------------------------------------------- stage 4: exercise
def set_features(acc: np.ndarray, gyro: Optional[np.ndarray], fs: float) -> np.ndarray:
    a = _lp(acc, 6.0, fs)
    g = a.mean(0)
    gn = g / (np.linalg.norm(g) + 1e-9)                     # mean wrist orientation
    x = a - g
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    var_ratio = (s ** 2) / ((s ** 2).sum() + 1e-12)
    axis = np.abs(vt[0])                                    # sign-invariant motion axis
    rng = np.percentile(x, 95, axis=0) - np.percentile(x, 5, axis=0)
    feats = [gn, var_ratio, axis, x.std(0), rng]
    if gyro is not None and np.any(gyro):
        gg = _lp(gyro, 6.0, fs)
        feats += [gg.std(0), np.abs(gg).max(0)]
    else:
        feats += [np.zeros(3), np.zeros(3)]
    return np.concatenate(feats)


class ExerciseClassifier:
    """Random forest over set-level features. fit() with labeled sets, then predict()."""

    def __init__(self, fs: float = 50.0):
        from sklearn.ensemble import RandomForestClassifier
        self.fs = fs
        self.rf = RandomForestClassifier(n_estimators=300, random_state=0)
        self.fitted = False

    def fit(self, sets, labels):
        """sets: list of (acc, gyro_or_None) segments; labels: list of str."""
        X = [set_features(a, g, self.fs) for a, g in sets]
        self.rf.fit(X, labels)
        self.fitted = True
        return self

    def predict(self, acc, gyro=None):
        if not self.fitted:
            return None
        return str(self.rf.predict(set_features(acc, gyro, self.fs).reshape(1, -1))[0])


def build_cnn(n_classes: int, win: int = 150, channels: int = 6):
    """Optional deep model for window-level exercise recognition (needs tensorflow; not
    exercised in this file's demo). Train on (N, win, channels) windows, softmax over classes."""
    from tensorflow import keras
    from tensorflow.keras import layers
    m = keras.Sequential([
        layers.Input((win, channels)),
        layers.Conv1D(32, 5, activation="relu"), layers.MaxPooling1D(2),
        layers.Conv1D(64, 5, activation="relu"), layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation="relu"), layers.GlobalAveragePooling1D(),
        layers.Dropout(0.3), layers.Dense(64, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ])
    m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


# --------------------------------------------------------------------------- streaming detector
class WorkoutDetector:
    """Feed samples as they arrive; get events back.

    Events (dicts):
      {"type":"set_start","t":sec}                       set confirmed (back-dated start)
      {"type":"reps","t":sec,"count":n}                  live rep count (only once >= 4 reps)
      {"type":"set_end","start":s,"end":e,"reps":n,"exercise":str|None,"rest_starts":e}
      {"type":"set_discarded","t":sec}                   motion that wasn't a real set
    """

    def __init__(self, cfg: Optional[Config] = None,
                 activity_model: Optional[Callable[[np.ndarray], bool]] = None,
                 exercise_model: Optional[ExerciseClassifier] = None,
                 accel_in_g: Optional[bool] = None):
        self.cfg = cfg or Config()
        c = self.cfg
        self.win_n, self.hop_n = int(round(c.win_s * c.fs)), int(round(c.hop_s * c.fs))
        self.activity_model = activity_model
        self.exercise_model = exercise_model
        self._in_g = accel_in_g
        self._acc = np.zeros((0, 3))
        self._gyro = np.zeros((0, 3))
        self._has_gyro: Optional[bool] = None
        self._dropped = 0
        self._total = 0
        self._next_end = self.win_n
        self._in_set = False
        self._act_run = 0
        self._idle_s = 0.0
        self._set_start = 0
        self._last_active_end = 0
        self._last_reps = 0

    # ---- public
    def push(self, acc, gyro=None) -> List[dict]:
        acc = np.atleast_2d(np.asarray(acc, float))
        if self._in_g is None:
            self._in_g = bool(np.median(np.linalg.norm(acc, axis=1)) < 3.0)
        if self._in_g:
            acc = acc * G
        if self._has_gyro is None:
            self._has_gyro = gyro is not None
        gyro = np.zeros_like(acc) if gyro is None else np.atleast_2d(np.asarray(gyro, float))
        self._acc = np.vstack([self._acc, acc])
        self._gyro = np.vstack([self._gyro, gyro])
        self._total += len(acc)

        events: List[dict] = []
        while self._next_end <= self._total:
            events += self._step(self._next_end)
            self._next_end += self.hop_n
        self._trim()
        return events

    def flush(self) -> List[dict]:
        """Call at end of recording (or when the user ends the workout)."""
        return self._finalize() if self._in_set else []

    # ---- internals
    def _slice(self, a: int, b: int):
        s, e = a - self._dropped, b - self._dropped
        return self._acc[s:e], (self._gyro[s:e] if self._has_gyro else None)

    def _step(self, end: int) -> List[dict]:
        c, fs = self.cfg, self.cfg.fs
        a, g = self._slice(end - self.win_n, end)
        feats = window_features(a, g, fs, c)
        active = self.activity_model(feats) if self.activity_model else rule_activity(feats, c)
        ev: List[dict] = []

        if not self._in_set:
            self._act_run = self._act_run + 1 if active else 0
            if self._act_run >= c.start_windows:                 # hysteresis: start
                self._in_set = True
                self._set_start = max(self._dropped, end - self.win_n - (c.start_windows - 1) * self.hop_n)
                self._last_active_end = end
                self._idle_s = 0.0
                self._last_reps = 0
                ev.append({"type": "set_start", "t": self._set_start / fs})
            return ev

        if active:
            self._last_active_end, self._idle_s = end, 0.0
        else:
            self._idle_s += c.hop_s

        sa, sg = self._slice(self._set_start, end)
        r = count_reps(sa, sg, fs, c)
        if r.confirmed and r.count != self._last_reps:
            self._last_reps = r.count
            ev.append({"type": "reps", "t": end / fs, "count": r.count})

        if self._idle_s >= c.end_idle_s or (end - self._set_start) / fs > c.max_set_s:
            ev += self._finalize()                                # hysteresis: end
        return ev

    def _finalize(self) -> List[dict]:
        c, fs = self.cfg, self.cfg.fs
        s0, e0 = self._set_start, self._last_active_end
        self._in_set, self._act_run, self._idle_s = False, 0, 0.0
        a, g = self._slice(s0, e0)
        r = count_reps(a, g, fs, c)
        if not r.confirmed:
            return [{"type": "set_discarded", "t": e0 / fs}]
        start = s0 / fs + r.t_start
        end = s0 / fs + r.t_end
        ex = self.exercise_model.predict(a, g) if self.exercise_model else None
        return [{"type": "set_end", "start": round(start, 2), "end": round(end, 2),
                 "reps": r.count, "exercise": ex, "rest_starts": round(end, 2)}]

    def _trim(self):
        c = self.cfg
        keep = self._set_start if self._in_set else self._next_end - self.win_n - c.start_windows * self.hop_n
        keep = max(self._dropped, keep)
        cut = keep - self._dropped
        if cut > 0:
            self._acc, self._gyro = self._acc[cut:], self._gyro[cut:]
            self._dropped = keep


# --------------------------------------------------------------------------- convenience / IO
def detect_sets(acc, gyro=None, fs=50.0, **kw) -> List[dict]:
    det = WorkoutDetector(Config(fs=fs), **kw)
    ev = []
    for i in range(0, len(acc), int(fs)):
        ev += det.push(acc[i:i + int(fs)], None if gyro is None else gyro[i:i + int(fs)])
    ev += det.flush()
    return ev


def load_csv(path: str, fs: float):
    raw = np.genfromtxt(path, delimiter=",", skip_header=1 if not _is_num(open(path).readline()) else 0)
    t, rest = raw[:, 0], raw[:, 1:]
    grid = np.arange(t[0], t[-1], 1 / fs)
    res = np.column_stack([np.interp(grid, t, rest[:, i]) for i in range(rest.shape[1])])
    return res[:, :3], (res[:, 3:6] if res.shape[1] >= 6 else None)


def _is_num(line: str) -> bool:
    try:
        float(line.split(",")[0])
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------- synthetic demo
def synth(fs=50, seed=0, plan=None):
    """Idle / set / idle ... with orientation swing + harmonics + noise. Returns acc (g), gyro, truth."""
    rng = np.random.default_rng(seed)
    plan = plan or [("idle", 10), ("set", 10, 3.0), ("idle", 25), ("set", 12, 2.4),
                    ("idle", 20), ("set", 8, 3.5), ("idle", 12)]
    acc, gyr, truth = [], [], []
    for p in plan:
        if p[0] == "idle":
            n = int(p[1] * fs)
            a = np.column_stack([np.zeros(n), np.zeros(n), np.ones(n)]) + rng.normal(0, 0.01, (n, 3))
            a += np.cumsum(rng.normal(0, 0.0008, (n, 3)), axis=0)
            g = rng.normal(0, 0.01, (n, 3))
        else:
            reps, P = p[1], p[2]
            t = np.arange(int(reps * P * fs)) / fs
            if len(p) > 3 and p[3] == "curl":     # one-sided: rest -> up -> back to rest
                th = 0.9 * (1 - np.cos(2 * np.pi * t / P)) / 2
            else:                                 # two-sided swing around rest
                th = 0.9 * np.sin(2 * np.pi * t / P)
            a = np.column_stack([-np.sin(th) + 0.08 * np.sin(4 * np.pi * t / P), np.zeros(len(t)), np.cos(th)])
            a += rng.normal(0, 0.03, a.shape)
            g = np.column_stack([np.zeros(len(t)), np.gradient(th, 1 / fs), np.zeros(len(t))]) + rng.normal(0, 0.05, a.shape)
            truth.append(reps)
        acc.append(a)
        gyr.append(g)
    return np.vstack(acc), np.vstack(gyr), truth


def demo():
    fs = 50
    acc, gyro, truth = synth(fs)
    ev = detect_sets(acc, gyro, fs)
    print(json.dumps(ev, indent=1))
    got = [e["reps"] for e in ev if e["type"] == "set_end"]
    print("truth:", truth, " detected:", got)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--fs", type=float, default=50.0)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo or not args.csv:
        demo()
    else:
        A, Gy = load_csv(args.csv, args.fs)
        out = [e for e in detect_sets(A, Gy, args.fs) if e["type"] == "set_end"]
        print(json.dumps(out, indent=1))
