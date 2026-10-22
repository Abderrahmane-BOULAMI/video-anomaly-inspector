"""Per-frame measurements from the camera stream (OpenCV only, no training needed).

Each frame gives an interpretable feature vector: belt speed, objects on the belt (count, angle, shape,
colour), appearance of the machine area, global image quality, and the position of the fan hub marker.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .synth import BELT_Y0, BELT_Y1, FAN_C, FAN_R, MACHINE, W

PART_HUE = (95, 115)   # OpenCV hue range of the blue parts (0..179)
EDGE = 18              # ignore objects cut by the image border


def belt_speed(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    """Horizontal displacement (px/frame) of the belt between two frames, by phase correlation."""
    a = prev_gray[BELT_Y0 + 4:BELT_Y1 - 4].astype(np.float32)
    b = gray[BELT_Y0 + 4:BELT_Y1 - 4].astype(np.float32)
    (dx, _dy), _resp = cv2.phaseCorrelate(a, b)
    return float(dx)


def belt_objects(frame: np.ndarray) -> list[dict]:
    hsv = cv2.cvtColor(frame[BELT_Y0:BELT_Y1], cv2.COLOR_BGR2HSV)
    mask = (hsv[..., 1] > 90).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 60:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if x < EDGE or x + w > W - EDGE:
            continue
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
        if rw < rh:
            ang -= 90
        ang = (ang + 90) % 180 - 90
        ang = ang - 180 if ang > 90 else ang
        m = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(m, [c], -1, 1, -1)
        hue = float(np.median(hsv[..., 0][m > 0]))
        out.append({"cx": cx, "area": area, "angle": float(ang if abs(ang) <= 45 else ang - np.sign(ang) * 90),
                    "fill": area / max(1.0, rw * rh), "hue": hue})
    return out


def frame_features(frame: np.ndarray, prev_gray: np.ndarray | None) -> tuple[dict, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    objs = belt_objects(frame)
    x0, y0, x1, y1 = MACHINE
    mach = gray[y0:y1, x0:x1]
    fan = frame[FAN_C[1] - FAN_R - 10:FAN_C[1] + FAN_R + 10, FAN_C[0] - FAN_R - 10:FAN_C[0] + FAN_R + 10]
    hsv = cv2.cvtColor(fan, cv2.COLOR_BGR2HSV)
    red = ((hsv[..., 0] < 8) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 120) & (hsv[..., 2] > 120)
    ys, xs = np.nonzero(red)
    f = {
        "speed": belt_speed(prev_gray, gray) if prev_gray is not None else np.nan,
        "n_obj": len(objs),
        "max_abs_angle": max((abs(o["angle"]) for o in objs), default=0.0),
        "min_fill": min((o["fill"] for o in objs), default=1.0),
        "hue_off": max((min(abs(o["hue"] - PART_HUE[0]), abs(o["hue"] - PART_HUE[1]))
                        if not PART_HUE[0] <= o["hue"] <= PART_HUE[1] else 0.0 for o in objs), default=0.0),
        "mach_mean": float(mach.mean()),
        "mach_std": float(mach.std()),
        "brightness": float(gray.mean()),
        "dark_frac": float((gray < 20).mean()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
        "hub_x": float(xs.mean()) if len(xs) else np.nan,
        "hub_y": float(ys.mean()) if len(ys) else np.nan,
    }
    return f, gray


def extract(frames, thumbs: bool = False, thumb_size=(40, 30)) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Run over a frame iterator. Optionally keep small grey thumbnails for pixel-based models."""
    rows, small, prev = [], [], None
    for fr in frames:
        f, prev = frame_features(fr, prev)
        rows.append(f)
        if thumbs:
            small.append(cv2.resize(prev, thumb_size, interpolation=cv2.INTER_AREA).ravel() / 255.0)
    df = pd.DataFrame(rows)
    df["speed"] = df["speed"].bfill()
    return df, (np.array(small, np.float32) if thumbs else None)
