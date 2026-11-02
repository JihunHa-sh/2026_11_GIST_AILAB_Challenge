# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Keyboard teleop for G1 loco-manipulation: WASD walking, arrow keys move the wrist targets.

The device emits the 23-D action consumed by ``G1DecoupledWBCPinkAction``, so the WBC policy walks
the legs and the PINK IK moves the arms, exactly as with the XR teleop pipeline.

Key bindings:
    ============================== ================= =================
    Description                    Key (+ve)         Key (-ve)
    ============================== ================= =================
    Walk forward / backward        W                 S
    Turn left / right              A                 D
    Side-step left / right         Q                 E
    Squat down / stand up          Z                 X
    Wrist target forward / back    UP                DOWN
    Wrist target left / right      LEFT              RIGHT
    Both arms: spread / squeeze    LEFT              RIGHT
    Wrist target up / down         T                 G
    Select left / right / both arm 1 / 2 / 3
    Toggle hand close (selected)   K
    Wrist targets back to home     SPACE
    Reset device state             L
    ============================== ================= =================

With both arms selected (3), LEFT/RIGHT move the hands apart/together symmetrically, so one key
pair grasps a box between the palms. The other wrist keys move both hands the same way.

``R`` is left free because ``record_demos.py`` binds it to "reset recording instance".
"""

from __future__ import annotations

import logging
import numpy as np
import torch
import weakref
from collections.abc import Callable
from typing import NamedTuple

from isaaclab.devices.device_base import DeviceBase, DeviceCfg
from isaaclab.utils import configclass

from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.action_constants import (
    BASE_HEIGHT_CMD_START_IDX,
    LEFT_HAND_STATE_IDX,
    LEFT_WRIST_LINK_NAME,
    LEFT_WRIST_POS_END_IDX,
    LEFT_WRIST_POS_START_IDX,
    LEFT_WRIST_QUAT_END_IDX,
    LEFT_WRIST_QUAT_START_IDX,
    NAVIGATE_CMD_END_IDX,
    NAVIGATE_CMD_START_IDX,
    RIGHT_HAND_STATE_IDX,
    RIGHT_WRIST_LINK_NAME,
    RIGHT_WRIST_POS_END_IDX,
    RIGHT_WRIST_POS_START_IDX,
    RIGHT_WRIST_QUAT_END_IDX,
    RIGHT_WRIST_QUAT_START_IDX,
    TORSO_ORIENTATION_RPY_CMD_END_IDX,
)

logger = logging.getLogger(__name__)

G1_LOCOMANIP_ACTION_DIM = TORSO_ORIENTATION_RPY_CMD_END_IDX


class ArmReference(NamedTuple):
    """Per-arm reference points in the pelvis frame, the frame of the WBC Pink wrist targets."""

    wrist_home_pos: np.ndarray
    """Wrist position at the IK model's default configuration (xyz) [m]."""

    wrist_home_quat_xyzw: np.ndarray
    """Wrist orientation at the IK model's default configuration (xyzw)."""

    shoulder_pos: np.ndarray
    """Shoulder pitch joint position (xyz) [m], the center of the reach limit."""


# Values of the Unitree G1 29-DoF rev 1.0 model at its all-zero joint configuration (elbows at
# 90 degrees, forearms pointing forward). Used when the IK robot model cannot be loaded.
_FALLBACK_ARM_REFERENCES = {
    "left": ArmReference(np.array([0.2, 0.149, 0.095]), np.array([0.0, 0.0, 0.0, 1.0]), np.array([0.0, 0.1, 0.292])),
    "right": ArmReference(np.array([0.2, -0.149, 0.095]), np.array([0.0, 0.0, 0.0, 1.0]), np.array([0.0, -0.1, 0.292])),
}

_MOVE_KEYS = {"W", "S", "A", "D", "Q", "E", "Z", "X", "UP", "DOWN", "LEFT", "RIGHT", "T", "G"}
_ARM_SELECT_KEYS = {
    "KEY_1": ("left",),
    "NUMPAD_1": ("left",),
    "KEY_2": ("right",),
    "NUMPAD_2": ("right",),
    "KEY_3": ("left", "right"),
    "NUMPAD_3": ("left", "right"),
}

# Scenes that support this device, as the arguments that select them on the teleop / record command line.
# All ten Lightwheel one_wall kitchen styles share the counter prim and position used by g1_kitchen_pick_and_place.
_ONE_WALL_KITCHEN_STYLES = (
    "farmhouse1",
    "farmhouse2",
    "coastal",
    "industrial",
    "mediterranean",
    "modern1",
    "modern2",
    "rustic",
    "scandinavian",
    "traditional",
)
G1_KEYBOARD_SCENE_ARGS = [("Galileo warehouse (brown box -> bin)", "--task galileo_g1_locomanip_pick_and_place")] + [
    (
        f"Kitchen one_wall {style}",
        f"--task g1_kitchen_pick_and_place --kitchen_background lightwheel_kitchen_one_wall_{style}",
    )
    for style in _ONE_WALL_KITCHEN_STYLES
]


def format_scene_list() -> str:
    """Return the scenes usable with this device and how to switch to one on the next run."""
    lines = ["", "=" * 100, "G1 keyboard teleop: available scenes", "=" * 100]
    for index, (label, scene_args) in enumerate(G1_KEYBOARD_SCENE_ARGS, start=1):
        lines.append(f"  {index:2d}. {label:38s} {scene_args}")
    lines += [
        "",
        "  To switch scene, rerun the same command with its '--task ...' part replaced by one line above.",
        "  Extra options: --embodiment g1_wbc_agile_pink | --object <asset> --destination <asset> (kitchen)",
        "",
        "  Keys: W/S walk, A/D turn, Q/E side-step, Z/X squat/stand, arrows + T/G move wrists,",
        "        1/2/3 select arm, K close/open hand, SPACE wrists home, R reset episode, L reset device.",
        "  Kitchen: raise the wrists with T before walking up, or the hands catch on the cabinet.",
        "=" * 100,
    ]
    return "\n".join(lines)


def compute_arm_references() -> dict[str, ArmReference]:
    """Return the wrist home poses and shoulder positions of the IK robot model at its default configuration.

    Returns:
        Mapping from ``"left"``/``"right"`` to the arm's references in the pelvis frame.
    """
    import pinocchio as pin

    from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.utils.g1 import instantiate_g1_robot_model

    robot_model = instantiate_g1_robot_model()
    robot_model.cache_forward_kinematics(robot_model.q_default)
    references = {}
    for side, wrist_frame in (("left", LEFT_WRIST_LINK_NAME), ("right", RIGHT_WRIST_LINK_NAME)):
        wrist = robot_model.frame_placement(wrist_frame)
        shoulder = robot_model.frame_placement(f"{side}_shoulder_pitch_link")
        references[side] = ArmReference(
            wrist_home_pos=np.array(wrist.translation, dtype=np.float64),
            wrist_home_quat_xyzw=np.array(pin.Quaternion(wrist.rotation).coeffs(), dtype=np.float64),
            shoulder_pos=np.array(shoulder.translation, dtype=np.float64),
        )
    return references


class G1LocomanipKeyboard(DeviceBase):
    """Keyboard device that drives the G1 WBC + PINK IK embodiments with a 23-D loco-manipulation action."""

    def __init__(self, cfg: G1LocomanipKeyboardCfg):
        """Initialize the keyboard layer.

        Args:
            cfg: Configuration object for the keyboard device.
        """
        super().__init__()
        self.cfg = cfg
        self._sim_device = cfg.sim_device
        self._arm_refs = self._resolve_arm_references()
        self._additional_callbacks: dict[str, Callable] = {}
        self._pressed: set[str] = set()
        self._scene_list_printed = not cfg.print_scene_list
        self.reset()

        # Subscribe last so a failed subscription never leaves a half-built device.
        import carb
        import omni

        self._key_press = carb.input.KeyboardEventType.KEY_PRESS
        self._key_release = carb.input.KeyboardEventType.KEY_RELEASE
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        # Use a weakref so the device can be garbage-collected while subscribed.
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

    def __del__(self):
        """Release the keyboard interface."""
        if getattr(self, "_keyboard_sub", None) is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: the key bindings of the device."""
        msg = f"Keyboard Controller for G1 loco-manipulation: {self.__class__.__name__}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tWalk forward/backward: W/S, turn: A/D, side-step: Q/E, squat/stand: Z/X\n"
        msg += "\tWrist target forward/back: UP/DOWN, left/right: LEFT/RIGHT, up/down: T/G\n"
        msg += "\tSelect arm left/right/both: 1/2/3, toggle hand close: K, wrists home: SPACE\n"
        msg += "\tBoth arms selected: LEFT/RIGHT spread/squeeze the hands symmetrically\n"
        msg += "\tReset device state: L"
        return msg

    """
    Operations
    """

    def reset(self):
        """Stop walking, stand at the default height, open both hands and send the wrists home."""
        self._nav_cmd = np.zeros(3)
        self._base_height = self.cfg.default_base_height
        self._hand_offset = {"left": np.zeros(3), "right": np.zeros(3)}
        self._hand_closed = {"left": False, "right": False}
        self._active_arms: tuple[str, ...] = ("left", "right")
        self._pressed.clear()

    def add_callback(self, key: str, func: Callable):
        """Bind an extra function to a key press.

        Args:
            key: The carb keyboard input name, e.g. ``"R"``.
            func: Function called with no arguments when the key is pressed.
        """
        self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        """Advance the held-key state by one control step and return the 23-D action."""
        if not self._scene_list_printed:
            # Printed on the first step, after the startup logs, so it stays visible in the terminal.
            print(format_scene_list(), flush=True)
            self._scene_list_printed = True
        self._step_navigation()
        self._step_wrists()
        return torch.tensor(self._compose_action(), dtype=torch.float32, device=self._sim_device)

    """
    Internal helpers.
    """

    def _resolve_arm_references(self) -> dict[str, ArmReference]:
        if self.cfg.home_from_robot_model:
            try:
                return compute_arm_references()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Falling back to built-in G1 arm references ({e})")
        return dict(_FALLBACK_ARM_REFERENCES)

    def _axis(self, positive: str, negative: str) -> float:
        return float(positive in self._pressed) - float(negative in self._pressed)

    def _step_navigation(self):
        forward = self._axis("W", "S")
        max_vx = self.cfg.max_forward_vel if forward >= 0 else self.cfg.max_backward_vel
        target = np.array([
            forward * max_vx,
            self._axis("Q", "E") * self.cfg.max_lateral_vel,
            self._axis("A", "D") * self.cfg.max_yaw_rate,
        ])
        self._nav_cmd += self.cfg.velocity_smoothing * (target - self._nav_cmd)
        # Snap tiny residuals to zero so the WBC can switch back to its standing mode.
        self._nav_cmd[np.abs(self._nav_cmd) < 1e-3] = 0.0

        height_delta = self._axis("X", "Z") * self.cfg.base_height_step
        self._base_height = float(
            np.clip(self._base_height + height_delta, self.cfg.min_base_height, self.cfg.max_base_height)
        )

    def _step_wrists(self):
        delta = (
            np.array([self._axis("UP", "DOWN"), self._axis("LEFT", "RIGHT"), self._axis("T", "G")])
            * self.cfg.wrist_step
        )
        if not delta.any():
            return
        box_min = np.array(self.cfg.wrist_offset_min)
        box_max = np.array(self.cfg.wrist_offset_max)
        mirror = self.cfg.mirror_lateral_with_both_arms and len(self._active_arms) > 1
        for side in self._active_arms:
            ref = self._arm_refs[side]
            side_delta = delta.copy()
            if mirror and side == "right":
                # LEFT spreads and RIGHT squeezes the hands, mirrored about the robot's sagittal plane.
                side_delta[1] = -side_delta[1]
            offset = np.clip(self._hand_offset[side] + side_delta, box_min, box_max)
            # Keep the target within arm's reach so the IK can track it and the recorded action matches the motion.
            from_shoulder = ref.wrist_home_pos + offset - ref.shoulder_pos
            distance = np.linalg.norm(from_shoulder)
            if distance > self.cfg.max_reach:
                offset = ref.shoulder_pos + from_shoulder * (self.cfg.max_reach / distance) - ref.wrist_home_pos
            self._hand_offset[side] = offset

    def _compose_action(self) -> np.ndarray:
        action = np.zeros(G1_LOCOMANIP_ACTION_DIM)
        action[LEFT_HAND_STATE_IDX] = float(self._hand_closed["left"])
        action[RIGHT_HAND_STATE_IDX] = float(self._hand_closed["right"])
        slices = {
            "left": (
                LEFT_WRIST_POS_START_IDX,
                LEFT_WRIST_POS_END_IDX,
                LEFT_WRIST_QUAT_START_IDX,
                LEFT_WRIST_QUAT_END_IDX,
            ),
            "right": (
                RIGHT_WRIST_POS_START_IDX,
                RIGHT_WRIST_POS_END_IDX,
                RIGHT_WRIST_QUAT_START_IDX,
                RIGHT_WRIST_QUAT_END_IDX,
            ),
        }
        for side, (pos_start, pos_end, quat_start, quat_end) in slices.items():
            ref = self._arm_refs[side]
            action[pos_start:pos_end] = ref.wrist_home_pos + self._hand_offset[side]
            action[quat_start:quat_end] = ref.wrist_home_quat_xyzw
        action[NAVIGATE_CMD_START_IDX:NAVIGATE_CMD_END_IDX] = self._nav_cmd
        action[BASE_HEIGHT_CMD_START_IDX] = self._base_height
        # Torso roll/pitch/yaw commands stay at zero, as in the XR pipeline.
        return action

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Track held keys and handle one-shot keys."""
        # CHAR events carry the typed character as a plain string in ``event.input``.
        if event.type not in (self._key_press, self._key_release):
            return True
        name = event.input.name
        if event.type == self._key_press:
            if name in _MOVE_KEYS:
                self._pressed.add(name)
            elif name in _ARM_SELECT_KEYS:
                self._active_arms = _ARM_SELECT_KEYS[name]
            elif name == "K":
                for side in self._active_arms:
                    self._hand_closed[side] = not self._hand_closed[side]
            elif name == "SPACE":
                self._hand_offset = {"left": np.zeros(3), "right": np.zeros(3)}
            elif name == "L":
                self.reset()
            if name in self._additional_callbacks:
                self._additional_callbacks[name]()
        elif event.type == self._key_release:
            self._pressed.discard(name)
        return True


@configclass
class G1LocomanipKeyboardCfg(DeviceCfg):
    """Configuration for the G1 loco-manipulation keyboard device."""

    max_forward_vel: float = 0.4
    """Walking speed while W is held [m/s]."""

    max_backward_vel: float = 0.3
    """Walking speed while S is held [m/s]."""

    max_lateral_vel: float = 0.3
    """Side-step speed while Q/E is held [m/s]."""

    max_yaw_rate: float = 0.5
    """Turn rate while A/D is held [rad/s]."""

    velocity_smoothing: float = 0.2
    """Fraction of the gap to the target velocity closed each control step (1.0 = no smoothing)."""

    default_base_height: float = 0.74
    """Pelvis height command when standing [m]."""

    min_base_height: float = 0.5
    """Lowest pelvis height reachable with Z [m]."""

    max_base_height: float = 0.75
    """Highest pelvis height reachable with X [m]."""

    base_height_step: float = 0.002
    """Pelvis height change per control step while Z/X is held [m]."""

    wrist_step: float = 0.005
    """Wrist target displacement per control step while an arrow or T/G key is held [m]."""

    wrist_offset_min: tuple[float, float, float] = (-0.1, -0.15, -0.1)
    """Lower bound of the wrist target offset from its home pose, pelvis frame xyz [m]."""

    wrist_offset_max: tuple[float, float, float] = (0.2, 0.15, 0.25)
    """Upper bound of the wrist target offset from its home pose, pelvis frame xyz [m]."""

    max_reach: float = 0.36
    """Largest shoulder-to-wrist distance of a target [m]. Arena's arms-only PINK IK tracks a fixed wrist
    orientation within about 2 cm inside this radius; for lower or further objects, squat or walk closer."""

    mirror_lateral_with_both_arms: bool = True
    """With both arms selected, move the right wrist opposite to the left one on LEFT/RIGHT, so the keys
    spread and squeeze the hands; otherwise both wrists shift sideways together."""

    home_from_robot_model: bool = True
    """Compute the wrist home poses from the IK robot model; otherwise use built-in G1 values."""

    print_scene_list: bool = True
    """Print the scenes usable with this device, and how to switch to one, on the first control step."""

    class_type: type[DeviceBase] = G1LocomanipKeyboard
