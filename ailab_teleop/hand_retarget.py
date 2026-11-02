"""Human hand keypoints (Fast SAM 3D Body / MHR70) -> finger curl -> Unitree Dex3 joint targets.

Curl is hand-agnostic: per finger, the sum of the bend angles along wrist -> base -> 2 -> 3 -> tip, mapped linearly
from an "open" to a "closed" angle onto 0..1. Dex3 has a thumb, an index and a "middle" finger; the middle one
follows the mean of the human middle and ring fingers. 0 = open (all Dex3 joints 0), 1 = SONIC's closed hand.
The model's thumb keypoints barely change between an open hand and a fist (the thumb is small and occluded), so by
default the Dex3 thumb follows the mean curl of index and middle (gripper-like); thumb_mode="own" uses the thumb.
mode="grab" (default) makes each hand a two-state gripper: fully closed once the mean index/middle curl exceeds
`grab_close`, fully open again below `grab_open` (hysteresis, so it does not chatter at the threshold).
"""

import numpy as np

# MHR70 keypoint indices (sam_3d_body/metadata/mhr70.py): per finger [base(third_joint), 2, 3, tip(4)]
MHR70_HAND = {
    "right": {"wrist": 41, "thumb": [24, 23, 22, 21], "index": [28, 27, 26, 25], "middle": [32, 31, 30, 29],
              "ring": [36, 35, 34, 33], "pinky": [40, 39, 38, 37]},
    "left": {"wrist": 62, "thumb": [45, 44, 43, 42], "index": [49, 48, 47, 46], "middle": [53, 52, 51, 50],
             "ring": [57, 56, 55, 54], "pinky": [61, 60, 59, 58]},
}
FINGERS = ("thumb", "index", "middle", "ring", "pinky")

# Dex3 closed pose = SONIC's default hand pose (input_interface.hpp GetHandPose); open = all zeros.
# Order: left  [thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1]
#        right [thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1]
DEX3_CLOSED = {"left": np.array([0.0, 0.0, 1.75, -1.57, -1.75, -1.57, -1.75]),
               "right": np.array([0.0, 0.0, -1.75, 1.57, 1.75, 1.57, 1.75])}
DEX3_SLOT = {"left": {"thumb": [0, 1, 2], "middle": [3, 4], "index": [5, 6]},
             "right": {"thumb": [0, 1, 2], "index": [3, 4], "middle": [5, 6]}}


def finger_bend_deg(kp3d: np.ndarray, side: str) -> dict:
    """Summed bend angle [deg] per finger from 70x3 keypoints."""
    ids = MHR70_HAND[side]
    wrist = kp3d[ids["wrist"]]
    out = {}
    for f in FINGERS:
        pts = np.concatenate([wrist[None], kp3d[ids[f]]], axis=0)  # wrist, base, 2, 3, tip
        seg = np.diff(pts, axis=0)
        seg /= np.linalg.norm(seg, axis=1, keepdims=True) + 1e-9
        cos = np.clip((seg[1:] * seg[:-1]).sum(axis=1), -1.0, 1.0)
        out[f] = float(np.degrees(np.arccos(cos)).sum())
    return out


class Dex3Retargeter:
    """Keypoints -> smoothed per-finger curl (0..1) -> Dex3 7-joint targets for both hands."""

    def __init__(self, open_deg=(40.0, 50.0), closed_deg=(110.0, 160.0), alpha=0.5, thumb_mode="fingers",
                 mode="grab", grab_close=0.5, grab_open=0.3):
        # (thumb, other fingers): bend sums measured on an open / a closed hand (D555 at ~2 m, Oct 2026)
        self.open_deg, self.closed_deg, self.alpha, self.thumb_mode = open_deg, closed_deg, alpha, thumb_mode
        self.mode, self.grab_close, self.grab_open = mode, grab_close, grab_open
        self.grab = {"left": False, "right": False}
        self.curl = {s: {"thumb": 0.0, "index": 0.0, "middle": 0.0} for s in ("left", "right")}
        self.last_bend = {"left": {}, "right": {}}

    def _norm(self, bend: float, thumb: bool) -> float:
        lo, hi = (self.open_deg[0], self.closed_deg[0]) if thumb else (self.open_deg[1], self.closed_deg[1])
        return float(np.clip((bend - lo) / (hi - lo), 0.0, 1.0))

    def update(self, kp3d: np.ndarray):
        kp3d = np.asarray(kp3d, dtype=np.float64).reshape(-1, 3)
        for side in ("left", "right"):
            bend = finger_bend_deg(kp3d, side)
            self.last_bend[side] = bend
            target = {"index": self._norm(bend["index"], False),
                      "middle": 0.5 * (self._norm(bend["middle"], False) + self._norm(bend["ring"], False))}
            target["thumb"] = (self._norm(bend["thumb"], True) if self.thumb_mode == "own"
                               else 0.5 * (target["index"] + target["middle"]))
            for f, v in target.items():
                self.curl[side][f] += self.alpha * (v - self.curl[side][f])
            c = 0.5 * (self.curl[side]["index"] + self.curl[side]["middle"])
            if self.grab[side] and c < self.grab_open:
                self.grab[side] = False
            elif not self.grab[side] and c > self.grab_close:
                self.grab[side] = True

    def joints(self, side: str) -> np.ndarray:
        if self.mode == "grab":
            return DEX3_CLOSED[side].copy() if self.grab[side] else np.zeros(7)
        q = np.zeros(7)
        for f, slots in DEX3_SLOT[side].items():
            q[slots] = self.curl[side][f] * DEX3_CLOSED[side][slots]
        return q
