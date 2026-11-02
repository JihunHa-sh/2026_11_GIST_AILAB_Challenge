"""Camera teleop device for the G1 WBC + PINK IK embodiments (`g1_wbc_pink`), e.g. the g1_fast_transport kitchen.

Reads run_fastsam_publisher.py's stream (D555 + Fast SAM 3D Body, tcp://localhost:5556) and emits the same 23-D
action as Arena's keyboard / XR teleop (layout in isaaclab_arena_g1/challenge/API.md, section 8):

  * walking: the operator's pelvis track. The robot follows the operator's displacement and heading change since the
    anchor, expressed in the operator's own frame (forward/left/turn); distances are amplified (walk_scale). It goes
    through a P controller on an open-loop estimate
    of the robot's own motion (integrated commands, 0.02 s per step). This keeps the walked distance right even though
    the simulation runs slower than real time.
  * arms: wrist poses relative to the pelvis (SMPL forward kinematics in the publisher). At calibration the operator
    stands at attention and that pose is mapped to the robot's rest wrist targets; afterwards wrist displacements
    (scaled to the G1's shorter arms) and rotations are applied on top.
  * hands: the publisher's grab state (fist = close, open hand = open).

Keys in the Isaac Sim window:
    C   calibrate: stand at attention facing the camera (arms hanging straight). Starts the teleop and a fresh
        episode (with record_demos.py the frames before it are discarded). Needed once per session.
    P   pause / resume. While paused the robot stands still; on resume the operator's current position becomes the
        new anchor, so you can walk back to the middle of the camera view and continue (like lifting a mouse).
    R   (record_demos.py) reset the episode.
"""

from __future__ import annotations

import collections
import json
import math
import weakref
from collections.abc import Callable

import numpy as np
import torch
import zmq
from scipy.spatial.transform import Rotation as R

from isaaclab.devices.device_base import DeviceBase, DeviceCfg
from isaaclab.utils import configclass

# isaaclab_arena_g1/challenge/api.py
REST_WRIST_POS = {"left": np.array([0.03, 0.20, -0.07]), "right": np.array([0.03, -0.20, -0.07])}
REST_WRIST_QUAT_XYZW = np.array([0.0, math.sin(math.pi / 4.0), 0.0, math.cos(math.pi / 4.0)])
SHOULDER_POS = {"left": np.array([0.0, 0.10, 0.29]), "right": np.array([0.0, -0.10, 0.29])}  # pelvis frame, approx.
# SMPL canonical axes (x = person's left, y = up, z = forward) -> robot pelvis axes (x forward, y left, z up)
SMPL_TO_ROBOT = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
CONTROL_DT = 0.02  # g1_fast_transport control period [s] (one advance() per env step)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def decode(msg: bytes) -> dict:
    header = json.loads(msg[4:1284].rstrip(b"\0"))
    dt = {"i64": "<i8", "f64": "<f8", "f32": "<f4", "i32": "<i4"}
    out, off = {}, 1284
    for f in header["fields"]:
        n = int(np.prod(f["shape"]))
        out[f["name"]] = np.frombuffer(msg, np.dtype(dt[f["dtype"]]), n, off).reshape(f["shape"]).copy()
        off += n * np.dtype(dt[f["dtype"]]).itemsize
    return out


class CameraTeleop(DeviceBase):
    """D555 + Fast SAM 3D Body -> 23-D G1 loco-manipulation action."""

    def __init__(self, cfg: CameraTeleopCfg):
        super().__init__()
        self.cfg = cfg
        self._sim_device = cfg.sim_device
        self._callbacks: dict[str, Callable] = {}
        self._sub = zmq.Context.instance().socket(zmq.SUB)
        self._sub.setsockopt(zmq.CONFLATE, 1)
        self._sub.setsockopt(zmq.SUBSCRIBE, b"pose")
        self._sub.connect(cfg.pose_addr)
        self._pose, self._pose_t = None, 0.0
        self._calibrated = False
        self._arm_ref = None  # per side: (human wrist pos, human wrist rot) at calibration
        self._yaw_off = 0.0  # body_quat heading -> pelvis-track frame
        self._yaw = None  # filtered operator heading (raw body_quat frame)
        self._flip_t = None
        self._track = collections.deque(maxlen=8)  # recent operator pelvis xy (camera frame) for smoothing
        self._last_stamp = 0.0
        self._steps = 0
        self.reset()
        import carb
        import omni

        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._key_press = carb.input.KeyboardEventType.KEY_PRESS
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard, lambda event, *args, obj=weakref.proxy(self): obj._on_key(event, *args))
        print(f"[camera_teleop] reading {cfg.pose_addr}. Stand at attention facing the camera, then press C in the "
              "Isaac Sim window. P = pause/re-anchor, R = reset episode.", flush=True)

    def __del__(self):
        if getattr(self, "_keyboard_sub", None) is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def __str__(self) -> str:
        return ("Camera teleop (D555 + Fast SAM 3D Body): C calibrate/start, P pause/re-anchor, "
                f"pose stream {self.cfg.pose_addr}")

    # ---- DeviceBase ----

    def reset(self):
        """New episode: the robot starts at its own origin; the operator's current pose becomes the anchor."""
        self._robot = np.zeros(3)  # open-loop estimate of the robot since the episode start: x, y, yaw
        self._paused = False
        self._grip = {"left": 0.0, "right": 0.0}
        self._wrist = {s: (REST_WRIST_POS[s].copy(), REST_WRIST_QUAT_XYZW.copy()) for s in ("left", "right")}
        self._anchor_pending = True

    def add_callback(self, key: str, func: Callable):
        self._callbacks[key] = func

    def advance(self) -> torch.Tensor:
        self._read_stream()
        if (not self._calibrated and self.cfg.auto_calibrate_after_steps and self._pose is not None
                and self._steps >= self.cfg.auto_calibrate_after_steps):
            self._calibrate()  # testing without a keyboard
        nav = np.zeros(3)
        fresh = self._pose is not None and self._steps - self._pose_t < 25  # stream seen within ~0.5 s of steps
        if self._calibrated and fresh:
            if self._anchor_pending:
                self._set_anchor()
            if not self._paused:
                nav = self._locomotion()
                self._arms()
                g = self._pose.get("grab")
                if g is not None:
                    self._grip = {"left": float(g.reshape(2)[0] > 0.5), "right": float(g.reshape(2)[1] > 0.5)}
        self._integrate(nav)
        self._steps += 1
        if self._steps % 100 == 0:
            self._log(nav, fresh)
        return torch.tensor(self._compose(nav), dtype=torch.float32, device=self._sim_device)

    # ---- internals ----

    def _on_key(self, event, *args):
        if event.type == self._key_press:
            name = event.input.name
            if name == "C":
                self._calibrate()
            elif name == "P":
                self._paused = not self._paused
                if not self._paused:
                    self._anchor_pending = True
                print(f"[camera_teleop] {'PAUSED (robot stands still)' if self._paused else 'resumed, re-anchored'}",
                      flush=True)
            if name in self._callbacks:
                self._callbacks[name]()
        return True

    def _read_stream(self):
        try:
            while True:
                self._pose = decode(self._sub.recv(zmq.NOBLOCK))
                self._pose_t = self._steps
                stamp = float(self._pose["root_stamp"][0])
                if stamp > self._last_stamp:
                    self._last_stamp = stamp
                    self._track.append(self._pose["root_pos_w"].reshape(3)[:2].copy())
                    self._update_heading(self._pose["body_quat"].reshape(4))
        except zmq.Again:
            pass

    def _update_heading(self, q_wxyz: np.ndarray):
        w, x, y, z = q_wxyz
        fwd = R.from_quat([x, y, z, w]).apply([1.0, 0.0, 0.0])
        yaw = math.atan2(fwd[1], fwd[0])
        if self._yaw is None:
            self._yaw = yaw
        d = wrap(yaw - self._yaw)
        if abs(d) > math.radians(90):  # front/back flips of the pose model: ignore unless they persist
            self._flip_t = self._flip_t or self._steps
            if self._steps - self._flip_t < 50:
                return
        self._flip_t = None
        self._yaw = wrap(self._yaw + self.cfg.yaw_alpha * d)

    def _human_xy(self) -> np.ndarray:
        return np.mean(np.array(self._track), axis=0) if self._track else np.zeros(2)

    def _human_heading(self) -> float:
        return (self._yaw or 0.0) + self._yaw_off

    def _calibrate(self):
        if self._pose is None:
            print("[camera_teleop] no pose stream yet (is run_fastsam_publisher.py running?)", flush=True)
            return
        wp = self._pose["wrist_pose_rel"].reshape(2, 7)
        self._arm_ref = {s: (wp[i, :3].copy(), R.from_quat(wp[i, 3:])) for i, s in enumerate(("left", "right"))}
        p = self._pose["root_pos_w"].reshape(3)
        self._yaw_off = wrap(math.atan2(-p[1], -p[0]) - (self._yaw or 0.0))  # facing the camera = toward it
        self._calibrated, self._anchor_pending, self._paused = True, True, False
        print(f"[camera_teleop] CALIBRATED: attention pose = robot rest pose, heading offset "
              f"{math.degrees(self._yaw_off):+.0f} deg. New episode, teleop running.", flush=True)
        if "R" in self._callbacks:  # record_demos.py: drop the frames before calibration, start a fresh episode
            self._callbacks["R"]()

    def _set_anchor(self):
        self._anchor = {"human_xy": self._human_xy(), "human_yaw": self._human_heading(), "robot": self._robot.copy()}
        self._anchor_pending = False

    def _locomotion(self) -> np.ndarray:
        """P control of the robot's estimated pose toward the operator's displacement since the anchor."""
        a = self._anchor
        c, s = math.cos(-a["human_yaw"]), math.sin(-a["human_yaw"])
        d = self._human_xy() - a["human_xy"]
        d_body = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]]) * self.cfg.walk_scale  # operator frame
        rx, ry, ryaw = a["robot"]
        c, s = math.cos(ryaw), math.sin(ryaw)
        target = np.array([rx + c * d_body[0] - s * d_body[1], ry + s * d_body[0] + c * d_body[1]])
        target_yaw = wrap(ryaw + self.cfg.turn_scale * wrap(self._human_heading() - a["human_yaw"]))
        err = target - self._robot[:2]
        c, s = math.cos(-self._robot[2]), math.sin(-self._robot[2])
        err_b = np.array([c * err[0] - s * err[1], s * err[0] + c * err[1]])  # robot frame
        if np.linalg.norm(err_b) < self.cfg.pos_deadband:
            err_b[:] = 0.0
        yaw_err = wrap(target_yaw - self._robot[2])
        if abs(yaw_err) < math.radians(self.cfg.yaw_deadband_deg):
            yaw_err = 0.0
        return np.array([
            np.clip(self.cfg.kp_pos * err_b[0], -self.cfg.max_forward, self.cfg.max_forward),
            np.clip(self.cfg.kp_pos * err_b[1], -self.cfg.max_left, self.cfg.max_left),
            np.clip(self.cfg.kp_yaw * yaw_err, -self.cfg.max_turn, self.cfg.max_turn),
        ])

    def _integrate(self, nav: np.ndarray):
        c, s = math.cos(self._robot[2]), math.sin(self._robot[2])
        self._robot[0] += (c * nav[0] - s * nav[1]) * CONTROL_DT
        self._robot[1] += (s * nav[0] + c * nav[1]) * CONTROL_DT
        self._robot[2] = wrap(self._robot[2] + nav[2] * CONTROL_DT)

    def _arms(self):
        wp = self._pose["wrist_pose_rel"].reshape(2, 7)
        for i, side in enumerate(("left", "right")):
            p0, r0 = self._arm_ref[side]
            pos = REST_WRIST_POS[side] + self.cfg.arm_scale * (SMPL_TO_ROBOT @ (wp[i, :3] - p0))
            # keep the target within the arm's reach around the shoulder
            v = pos - SHOULDER_POS[side]
            n = np.linalg.norm(v)
            if n > self.cfg.max_reach:
                pos = SHOULDER_POS[side] + v * (self.cfg.max_reach / n)
            pos = self._outside_body(pos, side)
            d_rot = R.from_quat(wp[i, 3:]) * r0.inv()  # rotation since calibration, SMPL axes
            d_rot = R.from_matrix(SMPL_TO_ROBOT @ d_rot.as_matrix() @ SMPL_TO_ROBOT.T)
            quat = (d_rot * R.from_quat(REST_WRIST_QUAT_XYZW)).as_quat()
            p_old, _ = self._wrist[side]
            self._wrist[side] = (p_old + self.cfg.arm_alpha * (pos - p_old), quat)

    def _outside_body(self, pos: np.ndarray, side: str) -> np.ndarray:
        """Keep a wrist target out of the torso/pelvis box (pelvis frame).

        The challenge's G1 has self-collisions disabled (it is the evaluation setup, so it is left as is), so the IK
        would happily move a hand through the body. A target inside the box is pushed to the nearer of the box's front
        face and the face on the arm's own side.
        """
        lo, hi = np.asarray(self.cfg.body_box_min), np.asarray(self.cfg.body_box_max)
        if not np.all((pos > lo) & (pos < hi)):
            return pos
        sign = 1.0 if side == "left" else -1.0
        side_face = hi[1] if sign > 0 else lo[1]
        out = pos.copy()
        if hi[0] - pos[0] < abs(side_face - pos[1]):
            out[0] = hi[0]  # in front of the body
        else:
            out[1] = side_face  # beside the body, on the arm's own side
        return out

    def _compose(self, nav: np.ndarray) -> np.ndarray:
        a = np.zeros(23)
        a[0], a[1] = self._grip["left"], self._grip["right"]
        a[2:5], a[5:9] = self._wrist["left"]
        a[9:12], a[12:16] = self._wrist["right"]
        a[16:19] = nav
        a[19] = self.cfg.pelvis_height
        return a

    def _log(self, nav: np.ndarray, fresh: bool):
        state = "waiting for C" if not self._calibrated else ("PAUSED" if self._paused else "running")
        print(f"[camera_teleop] {state} | {'pose ok' if fresh else 'NO POSE'} | cmd fwd {nav[0]:+.2f} left {nav[1]:+.2f} "
              f"turn {nav[2]:+.2f} | robot est ({self._robot[0]:+.2f},{self._robot[1]:+.2f}) yaw "
              f"{math.degrees(self._robot[2]):+5.0f} | grip L{self._grip['left']:.0f} R{self._grip['right']:.0f}",
              flush=True)


@configclass
class CameraTeleopCfg(DeviceCfg):
    """Configuration of the camera teleop device."""

    pose_addr: str = "tcp://localhost:5556"
    walk_scale: float = 2.5
    """Robot displacement per operator displacement (2.5: 1.3 m in front of the camera = the 3.3 m carry)."""
    turn_scale: float = 1.0
    """Robot turn per operator turn."""
    kp_pos: float = 1.5
    kp_yaw: float = 1.5
    pos_deadband: float = 0.15
    """Ignore position errors below this [m, robot]; pose noise is amplified by walk_scale."""
    yaw_deadband_deg: float = 8.0
    max_forward: float = 0.4
    max_left: float = 0.3
    max_turn: float = 0.5
    """Speed limits [m/s, rad/s], the reference policy's maxima (API.md section 8)."""
    yaw_alpha: float = 0.3
    arm_scale: float = 0.8
    """Wrist displacement scale, operator -> G1 (shorter arms)."""
    arm_alpha: float = 0.5
    max_reach: float = 0.5
    body_box_min: tuple = (-0.12, -0.17, -0.35)
    body_box_max: tuple = (0.16, 0.17, 0.50)
    """Torso + pelvis box in the pelvis frame [m] that wrist targets may not enter (hand size included)."""
    pelvis_height: float = 0.74
    auto_calibrate_after_steps: int = 0
    """Testing only: calibrate automatically after this many steps (0 = press C)."""
    class_type: type[DeviceBase] = CameraTeleop
