"""Three ways to flag anomalies from the per-frame measurements, all calibrated on normal footage only.

1. Typed rules: one interpretable test per anomaly kind, thresholds set from a normal clip.
2. Isolation Forest on the same measurements: no rule to write, but no anomaly type either.
3. PCA on raw pixels (small grey thumbnails): no feature engineering at all.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .synth import FPS, SPEED

FEATS = ["speed", "n_obj", "max_abs_angle", "min_fill", "hue_off", "mach_mean", "mach_std", "brightness",
         "dark_frac", "sharpness"]


def smooth_speed(df: pd.DataFrame, k: int = 5) -> pd.Series:
    """Phase correlation sometimes locks on the wrong stripe (impossible values, e.g. on the first frames):
    drop them, interpolate, then take a rolling median."""
    s = df["speed"].where(df["speed"].abs() <= 2 * SPEED).interpolate(limit_direction="both")
    return s.rolling(k, center=True, min_periods=1).median()


@dataclass
class RuleDetector:
    mach_ref: float = 0.0
    mach_tol: float = 0.0
    angle_max: float = 12.0
    fill_min: float = 0.75
    hue_tol: float = 5.0
    dark_max: float = 0.25
    jam_speed: float = 1.0
    jam_frames: int = 5

    def fit(self, normal: pd.DataFrame, k: float = 6.0):
        """Machine-area reference from normal footage (the other thresholds are physical)."""
        self.mach_ref = float(normal["mach_mean"].median())
        mad = float((normal["mach_mean"] - self.mach_ref).abs().median()) * 1.4826
        self.mach_tol = max(3.0, k * mad)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        lab = np.array(["normal"] * n, dtype=object)
        cam = df["dark_frac"].to_numpy() > self.dark_max
        spd = smooth_speed(df).to_numpy()
        slow = spd < self.jam_speed
        # a stop counts only if it lasts jam_frames frames
        run = np.zeros(n, int)
        for i in range(n):
            run[i] = run[i - 1] + 1 if slow[i] and i else int(slow[i])
        jam = np.zeros(n, bool)
        for i in np.nonzero(run >= self.jam_frames)[0]:
            jam[i - self.jam_frames + 1:i + 1] = True
        smoke = (df["mach_mean"].to_numpy() - self.mach_ref) > self.mach_tol
        foreign = (df["hue_off"].to_numpy() > self.hue_tol) | (df["min_fill"].to_numpy() < self.fill_min)
        mis = df["max_abs_angle"].to_numpy() > self.angle_max
        # order matters: a covered lens invalidates every other measurement
        for kind, m in (("misaligned", mis), ("foreign_object", foreign), ("smoke", smoke), ("jam", jam),
                        ("camera_fault", cam)):
            lab[m] = kind
        return lab


class ForestDetector:
    def __init__(self, quantile: float = 0.995, seed: int = 0):
        self.q = quantile
        self.scaler = StandardScaler()
        self.model = IsolationForest(n_estimators=300, random_state=seed)

    def _X(self, df):
        X = df[FEATS].copy()
        X["speed"] = smooth_speed(df)
        return X.to_numpy(float)

    def fit(self, normal: pd.DataFrame):
        X = self.scaler.fit_transform(self._X(normal))
        self.model.fit(X)
        self.thr = float(np.quantile(-self.model.score_samples(X), self.q))
        return self

    def score(self, df):
        return -self.model.score_samples(self.scaler.transform(self._X(df)))

    def predict(self, df) -> np.ndarray:
        return self.score(df) > self.thr


class EnvelopeDetector:
    """Learns the normal range of every measurement and flags frames that leave it.

    No rule per anomaly, and unlike the Isolation Forest it still reacts to a measurement that never varied
    in the normal footage. It also says which measurement left its range, a first hint of the cause.
    """

    def __init__(self, q: float = 0.001, margin: float = 0.25, warmup: int = 10):
        """warmup: frames ignored at the start of a stream (no previous frame to measure motion against)."""
        self.q, self.margin, self.warmup = q, margin, warmup

    def _X(self, df):
        X = df[FEATS].copy()
        X["speed"] = smooth_speed(df)
        return X

    def fit(self, normal: pd.DataFrame):
        X = self._X(normal).iloc[self.warmup:]
        self.lo, self.hi = X.quantile(self.q), X.quantile(1 - self.q)
        floor = 0.05 * X.median().abs().clip(lower=1.0)
        self.span = (self.hi - self.lo).clip(lower=0) + floor
        return self

    def excess(self, df) -> pd.DataFrame:
        X = self._X(df)
        return ((self.lo - X).clip(lower=0) + (X - self.hi).clip(lower=0)) / self.span

    def predict(self, df) -> np.ndarray:
        flags = self.excess(df).max(axis=1).to_numpy() > self.margin
        flags[:self.warmup] = False
        return flags

    def explain(self, df) -> pd.Series:
        """Measurement furthest outside its normal range, per frame."""
        return self.excess(df).idxmax(axis=1)


class PixelPCADetector:
    def __init__(self, n_components: int = 30, quantile: float = 0.995):
        self.pca = PCA(n_components=n_components, random_state=0)
        self.q = quantile

    def fit(self, thumbs: np.ndarray):
        self.pca.fit(thumbs)
        self.thr = float(np.quantile(self.score(thumbs), self.q))
        return self

    def score(self, thumbs):
        rec = self.pca.inverse_transform(self.pca.transform(thumbs))
        return ((thumbs - rec) ** 2).mean(1)

    def predict(self, thumbs) -> np.ndarray:
        return self.score(thumbs) > self.thr


# ------------------------------------------------------------------ events and scoring
def segments(flags: np.ndarray, min_len: int = 3, gap: int = 5) -> list[tuple[int, int]]:
    """Contiguous runs of True, merging runs separated by fewer than `gap` frames, dropping short ones."""
    idx = np.nonzero(flags)[0]
    if not len(idx):
        return []
    segs, s, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > gap:
            segs.append((s, prev))
            s = i
        prev = i
    segs.append((s, prev))
    return [(a, b) for a, b in segs if b - a + 1 >= min_len]


def score_events(true_events, flags_by_kind: dict, fps: int = FPS) -> pd.DataFrame:
    """Per kind: events found, false alarms (detected segments touching no true event) and delay in seconds.

    flags_by_kind maps a kind (or "any") to a boolean array per frame.
    """
    rows = []
    for kind, flags in flags_by_kind.items():
        truth = [e for e in true_events if kind == "any" or e.kind == kind]
        segs = segments(flags)
        found, delays = 0, []
        for e in truth:
            hit = [a for a, b in segs if a <= e.end and b >= e.start]
            if hit:
                found += 1
                delays.append(max(0, min(hit) - e.start) / fps)
        false = sum(1 for a, b in segs if not any(a <= e.end and b >= e.start for e in true_events))
        rows.append({"kind": kind, "events": len(truth), "found": found, "false_alarms": false,
                     "mean_delay_s": float(np.mean(delays)) if delays else np.nan})
    return pd.DataFrame(rows)
