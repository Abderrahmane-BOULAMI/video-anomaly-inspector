"""Predictive maintenance from video: the wobble of a rotating fan, measured on its hub marker.

Every window of 10 s gives one amplitude (spread of the hub position around its mean). A CUSUM on that
amplitude detects the start of degradation; a trend fitted on the recent windows predicts when the
amplitude will cross the alarm limit, so maintenance can be planned before it does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .synth import FPS


def amplitude(df: pd.DataFrame, window_s: float = 10.0) -> pd.DataFrame:
    w = int(window_s * FPS)
    rows = []
    for s in range(0, len(df) - w + 1, w):
        d = df.iloc[s:s + w][["hub_x", "hub_y"]].dropna()
        if len(d) < w // 2:
            continue
        amp = float(np.sqrt(d["hub_x"].var() + d["hub_y"].var()))  # a circle of radius a gives a std of a/sqrt(2) per axis
        rows.append({"t_s": (s + w) / FPS, "amp_px": amp})
    return pd.DataFrame(rows)


def cusum_onset(amp: pd.DataFrame, ref_windows: int = 20, k: float = 0.5, h: float = 5.0) -> float | None:
    """Time (s) at which an upward CUSUM on the amplitude signals a change; reference from the first windows."""
    a = amp["amp_px"].to_numpy()
    mu, sd = a[:ref_windows].mean(), max(a[:ref_windows].std(), 1e-3)
    s = 0.0
    for t, x in zip(amp["t_s"], a):
        s = max(0.0, s + (x - mu) / sd - k)
        if s > h:
            return float(t)
    return None


def predict_crossing(amp: pd.DataFrame, limit: float, now_idx: int, history: int = 12, deg: int = 1) -> float | None:
    """From the last `history` windows up to now_idx, extrapolate a polynomial trend (linear by default:
    a quadratic reacted too early to noise on this data) and return the predicted crossing time (s)."""
    d = amp.iloc[max(0, now_idx - history + 1):now_idx + 1]
    if len(d) < 6:
        return None
    t, y = d["t_s"].to_numpy(), d["amp_px"].to_numpy()
    c = np.polyfit(t - t[-1], y, deg)
    grid = np.arange(0, 3600, 1.0)
    yy = np.polyval(c, grid)
    above = np.nonzero(yy >= limit)[0]
    return float(t[-1] + grid[above[0]]) if len(above) else None


def maintenance_alert(amp: pd.DataFrame, limit: float, horizon_s: float = 180.0, onset: float | None = None):
    """First time the predicted crossing falls within the horizon (only after a detected onset)."""
    for i in range(len(amp)):
        t = amp["t_s"].iloc[i]
        if onset is not None and t < onset:
            continue
        if amp["amp_px"].iloc[i] >= limit:
            return {"alert_s": float(t), "predicted_cross_s": float(t), "reason": "limit already reached"}
        p = predict_crossing(amp, limit, i)
        if p is not None and p - t <= horizon_s:
            return {"alert_s": float(t), "predicted_cross_s": p, "reason": "trend"}
    return None


def first_crossing(amp: pd.DataFrame, limit: float) -> float | None:
    hit = amp[amp["amp_px"] >= limit]
    return float(hit["t_s"].iloc[0]) if len(hit) else None
