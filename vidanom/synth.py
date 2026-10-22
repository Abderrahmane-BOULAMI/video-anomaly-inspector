"""Synthetic camera stream of a production station, with labelled anomalies.

The scene: a conveyor belt carrying parts under a fixed camera, a motor-driven fan above the belt and a
machine housing on the right. Anomalies are injected at known frames so that every detector can be scored:

- jam:            the belt stops (parts stop too)
- misaligned:     one part travels rotated on the belt
- foreign_object: an object that is not a part rides the belt
- smoke:          a grey plume grows in front of the machine housing
- camera_fault:   the lens is partly covered
- wear:           the fan starts to wobble and the wobble grows slowly (predictive maintenance case)

Real footage replaces this module; the detectors only need frames (and labels to be evaluated).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

W, H, FPS = 320, 240, 10
BELT_Y0, BELT_Y1 = 110, 170
PART_W, PART_H = 28, 18
PART_BGR = (200, 120, 40)
FOREIGN_BGR = (40, 40, 170)
FAN_C, FAN_R = (60, 52), 24
MACHINE = (200, 18, 300, 98)  # x0, y0, x1, y1
SPEED = 4.0                    # px per frame
SPACING = 60                   # px between parts
TYPES = ("jam", "misaligned", "foreign_object", "smoke", "camera_fault")


@dataclass
class Event:
    kind: str
    start: int
    end: int  # inclusive

    def as_dict(self):
        return {"kind": self.kind, "start": self.start, "end": self.end}


@dataclass
class Stream:
    """Labels are known up front; frames are rendered on demand (a 10-minute clip would not fit in memory)."""
    n_frames: int
    seed: int
    labels: np.ndarray          # (N,) object array, "normal" or the event kind
    events: list = field(default_factory=list)
    wobble_px: np.ndarray = None  # true fan wobble amplitude per frame
    wear_start: int | None = None

    def frames(self):
        """Yield BGR uint8 frames (H, W, 3) in order."""
        return _render(self)

    def frame_array(self, start: int = 0, stop: int | None = None, step: int = 1) -> np.ndarray:
        stop = self.n_frames if stop is None else stop
        return np.stack([fr for i, fr in enumerate(self.frames()) if start <= i < stop and (i - start) % step == 0])


def schedule(n_frames: int, rng, counts=None) -> list[Event]:
    """Place non-overlapping events at random frames (with 60 frames of margin between them)."""
    counts = counts or {"jam": 3, "misaligned": 4, "foreign_object": 3, "smoke": 2, "camera_fault": 2}
    dur = {"jam": (40, 80), "smoke": (90, 130), "camera_fault": (20, 35), "misaligned": (0, 0),
           "foreign_object": (0, 0)}
    events, taken = [], np.zeros(n_frames, bool)
    taken[:100] = True  # quiet start
    for kind, n in counts.items():
        for _ in range(n):
            for _try in range(500):
                lo, hi = dur[kind]
                length = int(rng.integers(lo, hi + 1)) if hi else int(W / SPEED) + 10
                s = int(rng.integers(100, n_frames - length - 1))
                if not taken[max(0, s - 60):s + length + 60].any():
                    taken[s:s + length] = True
                    events.append(Event(kind, s, s + length - 1))
                    break
    return sorted(events, key=lambda e: e.start)


def _draw_part(img, cx, cy, angle, color, foreign=False):
    if foreign:
        pts = np.array([[-9, -7], [6, -10], [12, 2], [3, 9], [-11, 5]], float)
        c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        pts = pts @ np.array([[c, -s], [s, c]]).T + (cx, cy)
        cv2.fillPoly(img, [pts.astype(np.int32)], color)
        return
    box = cv2.boxPoints(((cx, cy), (PART_W, PART_H), angle)).astype(np.int32)
    cv2.fillPoly(img, [box], color)
    cv2.polylines(img, [box], True, (230, 170, 90), 1)


def generate(n_frames: int = 6000, seed: int = 0, anomalies: bool = True, wear: bool = True) -> Stream:
    rng = np.random.default_rng(seed)
    events = schedule(n_frames, rng) if anomalies else []
    labels = np.array(["normal"] * n_frames, dtype=object)
    for e in events:
        labels[e.start:e.end + 1] = e.kind
    wear_start = int(n_frames * 0.5) if wear else None
    wob = np.full(n_frames, 0.5)
    if wear:
        t = np.arange(n_frames - wear_start)
        wob[wear_start:] = 0.5 + 6.0 * (t / max(1, len(t) - 1)) ** 1.5   # slow, accelerating degradation
    return Stream(n_frames, seed, labels, events, wob, wear_start)


def _base():
    base = np.zeros((H, W, 3), np.uint8)
    base[:] = (35, 38, 42)
    cv2.rectangle(base, (MACHINE[0], MACHINE[1]), (MACHINE[2], MACHINE[3]), (70, 72, 75), -1)
    for i, col in enumerate([(60, 200, 60), (60, 200, 60), (40, 180, 230)]):
        cv2.circle(base, (MACHINE[0] + 15 + 18 * i, MACHINE[1] + 12), 4, col, -1)
    cv2.putText(base, "M-12", (MACHINE[0] + 60, MACHINE[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    return base


def _render(st: Stream):
    rng = np.random.default_rng(st.seed + 10_000)
    n = st.n_frames
    active = {k: st.labels == k for k in TYPES}
    special = {e.start: e.kind for e in st.events if e.kind in ("misaligned", "foreign_object")}
    base = _base()
    belt_off, theta = 0.0, 0.0
    parts = []  # [x, angle, kind]
    next_spawn = 0.0
    smoke_age = 0
    for f in range(n):
        v = 0.0 if active["jam"][f] else SPEED
        belt_off = (belt_off + v) % 20
        next_spawn -= v
        if next_spawn <= 0:
            kind = "part"
            for s0, k in list(special.items()):
                if s0 <= f:
                    kind = k
                    del special[s0]
                    break
            ang = float(rng.uniform(25, 40) * rng.choice([-1, 1])) if kind == "misaligned" else float(rng.normal(0, 1.5))
            parts.append([-PART_W, ang, kind])
            next_spawn += SPACING
        for p in parts:
            p[0] += v
        parts = [p for p in parts if p[0] < W + PART_W]

        img = base.copy()
        cv2.rectangle(img, (0, BELT_Y0), (W, BELT_Y1), (88, 90, 92), -1)
        for x in np.arange(-20 + belt_off, W, 20):
            cv2.line(img, (int(x), BELT_Y0), (int(x), BELT_Y1), (70, 72, 74), 2)
        for x, ang, kind in parts:
            if kind == "foreign_object":
                _draw_part(img, x, (BELT_Y0 + BELT_Y1) / 2, ang * 3, FOREIGN_BGR, foreign=True)
            else:
                _draw_part(img, x, (BELT_Y0 + BELT_Y1) / 2, ang, PART_BGR)
        # fan with wear wobble: the hub moves on a small circle at the rotation frequency
        theta += 0.9
        a = st.wobble_px[f]
        c = (int(round(FAN_C[0] + a * np.cos(theta))), int(round(FAN_C[1] + a * np.sin(theta))))
        cv2.circle(img, c, FAN_R, (110, 112, 115), -1)
        for k in range(3):
            ang = theta + k * 2 * np.pi / 3
            cv2.line(img, c, (int(c[0] + 20 * np.cos(ang)), int(c[1] + 20 * np.sin(ang))), (60, 62, 64), 4)
        cv2.circle(img, c, 4, (30, 30, 220), -1)  # red hub marker
        if active["smoke"][f]:
            smoke_age += 1
            ov = img.copy()
            r = 8 + smoke_age // 3
            cv2.circle(ov, (250, 70), min(r, 45), (175, 175, 175), -1)
            cv2.circle(ov, (232, 52), min(r // 2 + 4, 30), (185, 185, 185), -1)
            alpha = min(0.75, 0.15 + smoke_age / 120)
            img = cv2.addWeighted(ov, alpha, img, 1 - alpha, 0)
        else:
            smoke_age = 0
        if active["camera_fault"][f]:
            cv2.rectangle(img, (0, 0), (int(W * 0.55), H), (8, 8, 8), -1)
        gain = 1.0 + 0.03 * np.sin(f / 37.0)
        noisy = img.astype(np.float32) * gain + rng.normal(0, 4, img.shape)
        yield np.clip(noisy, 0, 255).astype(np.uint8)


def write_video(frames, path: str, fps: int = FPS):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for fr in frames:
        vw.write(fr)
    vw.release()
