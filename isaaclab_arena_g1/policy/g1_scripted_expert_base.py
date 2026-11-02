# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared machinery of the scripted G1 loco-manipulation experts, driven by privileged object poses.

An expert is a state machine over phases. Each phase sets walking, wrist and base-height goals; the expert
moves its commands toward those goals every control step and advances to the next phase once they are
reached and held. It emits the same 23-D action a teleoperator produces for ``G1DecoupledWBCPinkAction``
(hand states, wrist poses in the pelvis frame, walking velocity, base height and torso orientation), so its
successful episodes can be recorded, augmented with Mimic and converted to LeRobot like human demonstrations.

The module only depends on numpy, so the state machines can be exercised without Isaac Sim.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, replace
from enum import Enum
from typing import ClassVar, NamedTuple, Protocol

from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.action_constants import (
    BASE_HEIGHT_CMD_START_IDX,
    LEFT_HAND_STATE_IDX,
    LEFT_WRIST_POS_END_IDX,
    LEFT_WRIST_POS_START_IDX,
    LEFT_WRIST_QUAT_END_IDX,
    LEFT_WRIST_QUAT_START_IDX,
    NAVIGATE_CMD_END_IDX,
    NAVIGATE_CMD_START_IDX,
    RIGHT_HAND_STATE_IDX,
    RIGHT_WRIST_POS_END_IDX,
    RIGHT_WRIST_POS_START_IDX,
    RIGHT_WRIST_QUAT_END_IDX,
    RIGHT_WRIST_QUAT_START_IDX,
    TORSO_ORIENTATION_RPY_CMD_END_IDX,
)

G1_LOCOMANIP_ACTION_DIM = TORSO_ORIENTATION_RPY_CMD_END_IDX

SIDES = ("left", "right")

# Wrist and shoulder positions of the G1 IK model at its default configuration (elbows at 90 degrees,
# forearms pointing forward), in the pelvis frame [m].
HOME_WRIST_POS = {
    "left": np.array([0.2, 0.149, 0.095]),
    "right": np.array([0.2, -0.149, 0.095]),
}
SHOULDER_POS = {
    "left": np.array([0.0, 0.1, 0.292]),
    "right": np.array([0.0, -0.1, 0.292]),
}

# With an identity wrist orientation the Dex3 fingers point forward and both palms face the body
# midline. Distance from the wrist axis to the palm surface [m].
PALM_SURFACE_OFFSET = 0.017
IDENTITY_QUAT_XYZW = np.array([0.0, 0.0, 0.0, 1.0])


def pitch_quat_xyzw(pitch_deg: float) -> np.ndarray:
    """Return the xyzw quaternion of a rotation about the pelvis y axis; positive pitch turns the fingers down."""
    half = 0.5 * math.radians(pitch_deg)
    return np.array([0.0, math.sin(half), 0.0, math.cos(half)])


# Arms hanging at the sides ("attention" pose): wrists beside the hips with the fingers pointing down, in the
# pelvis frame [m]. The IK reaches it with the elbows slightly bent; see G1_ARMS_DOWN_JOINT_POS.
REST_WRIST_POS = {
    "left": np.array([0.03, 0.20, -0.07]),
    "right": np.array([0.03, -0.20, -0.07]),
}
REST_WRIST_QUAT_XYZW = pitch_quat_xyzw(90.0)

# Arm joint positions [rad] of the attention pose, for spawning the G1 with its arms already down.
G1_ARMS_DOWN_JOINT_POS = {
    "left_shoulder_pitch_joint": 0.05,
    "left_shoulder_roll_joint": 0.12,
    "left_shoulder_yaw_joint": 0.05,
    "left_elbow_joint": 1.28,
    "left_wrist_roll_joint": 0.06,
    "left_wrist_pitch_joint": 0.24,
    "left_wrist_yaw_joint": -0.14,
    "right_shoulder_pitch_joint": 0.05,
    "right_shoulder_roll_joint": -0.12,
    "right_shoulder_yaw_joint": -0.05,
    "right_elbow_joint": 1.28,
    "right_wrist_roll_joint": -0.06,
    "right_wrist_pitch_joint": 0.24,
    "right_wrist_yaw_joint": 0.14,
}


class NavigationGoal(NamedTuple):
    """Walking goal in the world frame."""

    xy: np.ndarray
    """Target pelvis position (xy) [m]."""

    heading: float
    """Target pelvis heading [rad]."""

    stop: bool = True
    """Whether the robot must come to a stop at the goal, rather than just pass through it."""


class SceneState(NamedTuple):
    """Privileged scene state the expert plans from, all in the simulation world frame."""

    pelvis_pos: np.ndarray
    """Pelvis position (xyz) [m]."""

    pelvis_quat_xyzw: np.ndarray
    """Pelvis orientation (xyzw)."""

    object_vertices: np.ndarray
    """Bounding-box corners of the object to pick up, shape (N, 3) [m]."""

    destination_vertices: np.ndarray
    """Bounding-box corners of the destination, shape (N, 3) [m]."""

    arm_joint_pos: dict[str, float] | None = None
    """Arm joint positions the IK last commanded, by joint name [rad]; the start of planned arm motions."""

    support_vertices: np.ndarray | None = None
    """Bounding-box corners of the furniture the object stands on, shape (N, 3) [m]; an obstacle when planning."""

    destination_interior_vertices: np.ndarray | None = None
    """Corners of a front-opening container's interior (e.g. a microwave cavity), shape (8, 3) [m], or None for
    containers filled from above."""

    object_pos: np.ndarray | None = None
    """Object root position [m], for grasps planned on the object mesh."""

    object_quat_xyzw: np.ndarray | None = None
    """Object root orientation, for grasps planned on the object mesh."""


WristPose = tuple[np.ndarray, np.ndarray]
"""Wrist position and xyzw quaternion in the pelvis frame."""


class ArmPlanner(Protocol):
    """Collision-aware planner of both arms, see ``g1_curobo_arm_planner.G1SceneArmPlanner``."""

    def plan(
        self, state: SceneState, goals: dict[str, WristPose], ignore: frozenset[str], sides: tuple[str, ...]
    ) -> list[dict[str, WristPose]] | None:
        """Return per-step wrist poses from the commanded arm configuration to ``goals``, or None on failure.

        ``ignore`` names the obstacles ("object", "destination", "support") the arms may touch; only the arms in
        ``sides`` move, the others hold their pose.
        """


def quat_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    """Return the rotation matrix of an xyzw quaternion."""
    x, y, z, w = quat_xyzw
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Return the xyzw quaternion (w >= 0) of a rotation matrix."""
    trace = np.trace(rotation)
    if trace > 0.0:
        k = 2.0 * math.sqrt(trace + 1.0)
        quat = np.array([
            (rotation[2, 1] - rotation[1, 2]) / k,
            (rotation[0, 2] - rotation[2, 0]) / k,
            (rotation[1, 0] - rotation[0, 1]) / k,
            0.25 * k,
        ])
    else:
        i = int(np.argmax(np.diag(rotation)))
        j, k_index = (i + 1) % 3, (i + 2) % 3
        k = 2.0 * math.sqrt(1.0 + rotation[i, i] - rotation[j, j] - rotation[k_index, k_index])
        quat = np.zeros(4)
        quat[i] = 0.25 * k
        quat[j] = (rotation[j, i] + rotation[i, j]) / k
        quat[k_index] = (rotation[k_index, i] + rotation[i, k_index]) / k
        quat[3] = (rotation[k_index, j] - rotation[j, k_index]) / k
    quat /= np.linalg.norm(quat)
    return quat if quat[3] >= 0.0 else -quat


def min_jerk(s: float) -> float:
    """Minimum-jerk blend of a normalized time ``s`` in [0, 1]: zero velocity and acceleration at both ends."""
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation between xyzw quaternions along the shortest arc."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    angle = math.acos(dot)
    return (math.sin((1.0 - t) * angle) * q0 + math.sin(t * angle) * q1) / math.sin(angle)


@dataclass
class ScriptedExpertParams:
    """Walking, arm-speed and timing parameters shared by the scripted experts."""

    ready_wrist_raise: float = 0.08
    """How far above the home pose the empty hands are carried while walking [m]. In simulation the arms sag
    a few centimeters below their IK targets, so at the home height the hands catch on furniture edges."""

    stand_height: float = 0.74
    """Base height command while walking [m]; the WBC default is 0.74 (HOMIE) or 0.72 (AGILE)."""

    max_reach: float = 0.38
    """Maximum wrist distance from the shoulder, so every target stays within the IK's reach [m]."""

    max_forward_vel: float = 0.4
    """Maximum forward walking speed [m/s]."""

    max_backward_vel: float = 0.3
    """Maximum backward walking speed [m/s]."""

    max_lateral_vel: float = 0.3
    """Maximum side-stepping speed [m/s]."""

    max_yaw_rate: float = 0.5
    """Maximum turning rate [rad/s]."""

    min_vel: float = 0.08
    """Smallest nonzero walking command, so the walking policy does not stall short of the goal."""

    kp_position: float = 2.0
    """Proportional gain from position error [m] to walking speed [m/s]."""

    kp_heading: float = 1.5
    """Proportional gain from heading error [rad] to turning rate [rad/s]."""

    velocity_smoothing: float = 0.2
    """Fraction of the gap to the target velocity closed per step."""

    position_tolerance: float = 0.04
    """Position error below which a walking goal counts as reached [m]."""

    heading_tolerance: float = 0.06
    """Heading error below which a walking goal counts as reached [rad]."""

    wrist_speed: float = 0.25
    """Average speed of straight wrist moves [m/s]; they follow a minimum-jerk profile, so they start and stop
    gently."""

    wrist_turn_speed: float = 1.5
    """Average turning speed of the wrist orientation in straight moves [rad/s]."""

    raise_distance: float = 0.5
    """Distance to the walking goal at which the hanging arms start rising into the working pose [m]."""

    wait_for_arms_distance: float = 0.4
    """Distance to the walking goal within which the robot stops and waits while its arms are still moving [m]."""

    height_speed: float = 0.15
    """Speed of the base height command [m/s]."""

    settle_time: float = 0.5
    """Standing time after reset, so the objects and the robot come to rest [s]."""

    hold_time: float = 0.3
    """Time each phase holds its reached targets, so the IK and walking policy catch up [s]."""

    hand_time: float = 0.5
    """Time given to the fingers to close or open [s]."""

    navigation_timeout: float = 15.0
    """Time after which a walking phase gives up [s]."""

    manipulation_timeout: float = 5.0
    """Time after which an arm phase gives up [s]."""


def yaw_from_quat_xyzw(quat_xyzw: np.ndarray) -> float:
    """Return the heading (rotation about z) of an xyzw quaternion [rad]."""
    x, y, z, w = quat_xyzw
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def world_to_heading_frame(points: np.ndarray, origin: np.ndarray, yaw: float) -> np.ndarray:
    """Express world points in a frame at ``origin`` rotated by ``yaw`` about z (x forward, y left)."""
    c, s = math.cos(yaw), math.sin(yaw)
    delta = np.atleast_2d(points) - origin
    local = np.empty_like(delta)
    local[:, 0] = c * delta[:, 0] + s * delta[:, 1]
    local[:, 1] = -s * delta[:, 0] + c * delta[:, 1]
    local[:, 2] = delta[:, 2]
    return local


def heading_to_world_xy(local_xy: np.ndarray, origin_xy: np.ndarray, yaw: float) -> np.ndarray:
    """Map a horizontal point from a frame at ``origin_xy`` rotated by ``yaw`` back to the world frame."""
    c, s = math.cos(yaw), math.sin(yaw)
    return origin_xy + np.array([c * local_xy[0] - s * local_xy[1], s * local_xy[0] + c * local_xy[1]])


def distance_to_rectangle_edge(half_extents: np.ndarray, direction: np.ndarray) -> float:
    """Distance from the center of an axis-aligned rectangle to its boundary along a unit direction."""
    return float(np.min(half_extents / np.maximum(np.abs(direction), 1e-9)))


def unit_xy(vector: np.ndarray) -> np.ndarray:
    """Return the horizontal unit vector along ``vector``."""
    planar = np.array([vector[0], vector[1]], dtype=np.float64)
    norm = np.linalg.norm(planar)
    assert norm > 1e-6, "Cannot take the direction of a zero-length horizontal vector."
    return planar / norm


class ScriptedLocomanipExpertBase:
    """Base of the scripted G1 experts: phase sequencing, walking, wrist and base-height command generation.

    Call :meth:`reset` at the start of each episode and :meth:`compute_action` once per control step.
    Walking goals are planned in the world frame and tracked with a P-controller on the walking command;
    wrist goals are planned in the pelvis frame, where the WBC Pink action expects them, and the
    commanded wrist poses move toward their goals at a fixed speed.

    Subclasses define ``phases`` (an Enum with at least ``SETTLE``, ``DONE`` and whatever members are listed
    in ``hand_phases``, executed in definition order), ``navigation_phases``, :meth:`_plan_phase` and
    :meth:`_ready_wrist_pos`, and may override :meth:`_sample_params`.
    """

    phases: ClassVar[type[Enum]]
    """Phases of the motion, executed in definition order."""

    navigation_phases: ClassVar[tuple[Enum, ...]] = ()
    """Phases that walk, which use the navigation timeout instead of the manipulation timeout."""

    hand_phases: ClassVar[tuple[Enum, ...]] = ()
    """Phases that open or close fingers, which hold for the hand time instead of the hold time."""

    planned_phases: ClassVar[tuple[Enum, ...]] = ()
    """Phases whose arm moves go through the collision-aware planner when one is set; other moves stay straight,
    such as the short moves into contact with the object."""

    def __init__(
        self,
        params: ScriptedExpertParams,
        step_dt: float,
        rng: np.random.Generator | None = None,
    ):
        """Initialize the expert.

        Args:
            params: Geometry and timing of the motion.
            step_dt: Control period of the environment [s].
            rng: Random generator used to vary the motion between episodes, or None to repeat it exactly.
        """
        self.base_params = params
        self.step_dt = step_dt
        self.rng = rng
        self.planner: ArmPlanner | None = None
        """Collision-aware arm planner, set by the policy when motion planning is enabled."""
        self.num_planned_moves = 0
        self.num_failed_plans = 0
        self._phase_order = list(self.phases)
        self.reset()

    @property
    def phase(self) -> Enum:
        """Current phase of the motion."""
        return self._phase

    @property
    def done(self) -> bool:
        """Whether the motion has finished, successfully or not."""
        return self._phase == self.phases.DONE

    @property
    def failed(self) -> bool:
        """Whether the attempt was given up (a phase timed out or the object was lost), so it cannot succeed."""
        return self._failed

    @property
    def failure_reason(self) -> str:
        """Why the attempt was given up, or an empty string."""
        return self._failure_reason

    def _give_up(self, reason: str) -> Enum:
        """Mark the attempt as failed for ``reason`` and return the DONE phase."""
        self._failed = True
        self._failure_reason = f"{reason} (in phase {self._phase.value})"
        return self.phases.DONE

    def reset(self) -> None:
        """Start a new episode: stand at the walking height with the wrists in the ready pose."""
        self.params = self._sample_params()
        self._phase = self.phases.SETTLE
        self._phase_steps = 0
        self._hold_steps = 0
        self._failed = False
        self._failure_reason = ""
        # Start in the attention pose, arms hanging; phases raise them when there is work to do.
        self._wrist_cmd = {side: REST_WRIST_POS[side].copy() for side in SIDES}
        self._wrist_goal = {side: REST_WRIST_POS[side].copy() for side in SIDES}
        self._wrist_quat = {side: REST_WRIST_QUAT_XYZW.copy() for side in SIDES}
        self._wrist_quat_goal = {side: REST_WRIST_QUAT_XYZW.copy() for side in SIDES}
        self._hand_closed = {side: False for side in SIDES}
        self._arms_down = True
        # Set by a phase to move straight even where moves are normally planned, for the next goal change.
        self._force_straight = False
        # Per side: the move under way, as the goal it heads to and the wrist poses still to command.
        self._move_goal: dict[str, WristPose | None] = {side: None for side in SIDES}
        self._move_poses: dict[str, list[WristPose]] = {side: [] for side in SIDES}
        self._height_cmd = self.params.stand_height
        self._height_goal = self.params.stand_height
        self._nav_cmd = np.zeros(3)
        self._nav_goal: NavigationGoal | None = None
        self._nav_speed_scale = 1.0
        """Fraction of the walking speed limits used by the current phase."""
        self._nav_turn_scale = 1.0
        """Fraction of the turning rate limit used by the current phase."""
        self._nav_arrived = False
        self._nav_position_reached = False
        self._nav_heading_reached = False

    def compute_action(self, state: SceneState) -> np.ndarray:
        """Advance the motion by one control step and return the 23-D WBC Pink action."""
        pelvis_xy = np.asarray(state.pelvis_pos[:2], dtype=np.float64)
        yaw = yaw_from_quat_xyzw(state.pelvis_quat_xyzw)
        self._step_navigation(pelvis_xy, yaw)
        self._advance_phase(state, pelvis_xy, yaw)
        self._raise_arms_near_goal(pelvis_xy)
        self._start_wrist_moves(state)
        self._step_wrists()
        self._step_height()
        return self._compose_action()

    """
    Subclass hooks.
    """

    def _plan_phase(self, phase: Enum, state: SceneState, pelvis_xy: np.ndarray, yaw: float) -> None:
        """Set the walking, wrist, hand and height goals of ``phase`` from the current scene state."""
        raise NotImplementedError

    def _ready_wrist_pos(self) -> dict[str, np.ndarray]:
        """Wrist positions in the pelvis frame while walking with empty hands."""
        raise NotImplementedError

    def _working_wrist_poses(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Wrist positions and orientations the hanging arms rise to while walking up to the work, per side."""
        ready = self._ready_wrist_pos()
        return {side: (ready[side], IDENTITY_QUAT_XYZW) for side in SIDES}

    def _next_phase(self, state: SceneState) -> Enum:
        """Phase that follows the current one once its targets are reached; by default the next in definition order."""
        return self._phase_order[self._phase_order.index(self._phase) + 1]

    def _planning_ignores(self) -> frozenset[str]:
        """Obstacles the arms may touch in the current phase ("object", "destination", "support")."""
        return frozenset()

    def _object_lost(self, state: SceneState) -> bool:
        """Whether the object has slipped out of the hands."""
        return False

    def _phase_after_object_lost(self, state: SceneState) -> Enum:
        """Phase to switch to when the object slipped out; giving up unless a subclass retries."""
        return self._give_up("lost the object")

    def _sample_params(self) -> ScriptedExpertParams:
        """Vary the motion between episodes so the demonstrations are not identical."""
        params = replace(self.base_params)
        if self.rng is None:
            return params
        speed_scale = self.rng.uniform(0.75, 1.0)
        params.max_forward_vel *= speed_scale
        params.max_backward_vel *= speed_scale
        params.max_lateral_vel *= speed_scale
        params.max_yaw_rate *= speed_scale
        params.wrist_speed *= self.rng.uniform(0.7, 1.0)
        params.hold_time += self.rng.uniform(0.0, 0.2)
        return params

    """
    Phase sequencing.
    """

    def _advance_phase(self, state: SceneState, pelvis_xy: np.ndarray, yaw: float) -> None:
        if self._phase == self.phases.DONE:
            return
        self._phase_steps += 1
        timeout = (
            self.params.navigation_timeout
            if self._phase in self.navigation_phases
            else self.params.manipulation_timeout
        )
        if self._phase_steps * self.step_dt > timeout:
            self._enter(self._give_up(f"timed out after {timeout:.0f} s"), state, pelvis_xy, yaw)
            return
        if self._object_lost(state):
            self._enter(self._phase_after_object_lost(state), state, pelvis_xy, yaw)
            return

        reached = self._targets_reached()
        self._hold_steps = self._hold_steps + 1 if reached else 0
        if reached and (self._hold_steps - 1) * self.step_dt >= self._required_hold_time():
            self._enter(self._next_phase(state), state, pelvis_xy, yaw)

    def _targets_reached(self) -> bool:
        wrists_reached = all(
            np.array_equal(self._wrist_cmd[side], self._wrist_goal[side])
            and np.array_equal(self._wrist_quat[side], self._wrist_quat_goal[side])
            for side in SIDES
        )
        height_reached = self._height_cmd == self._height_goal
        nav_reached = self._nav_goal is None or self._nav_arrived
        return wrists_reached and height_reached and nav_reached

    def _required_hold_time(self) -> float:
        if self._phase == self.phases.SETTLE:
            return self.params.settle_time
        if self._nav_goal is not None and not self._nav_goal.stop:
            return 0.0
        if self._phase in self.hand_phases:
            return self.params.hand_time
        return self.params.hold_time

    def _enter(self, phase: Enum, state: SceneState, pelvis_xy: np.ndarray, yaw: float) -> None:
        """Switch to ``phase`` and plan its goals from the current scene state."""
        self._phase = phase
        self._phase_steps = 0
        self._hold_steps = 0
        self._nav_goal = None
        self._nav_arrived = False
        self._nav_speed_scale = 1.0
        self._nav_turn_scale = 1.0
        self._nav_position_reached = False
        self._nav_heading_reached = False
        if phase != self.phases.DONE:
            self._plan_phase(phase, state, pelvis_xy, yaw)

    """
    Command generation.
    """

    def _step_navigation(self, pelvis_xy: np.ndarray, yaw: float) -> None:
        p = self.params
        target = np.zeros(3)
        self._nav_arrived = False
        if self._nav_goal is not None:
            error_xy = self._nav_goal.xy - pelvis_xy
            heading_error = wrap_to_pi(self._nav_goal.heading - yaw)
            # Hysteresis: once inside a tolerance, stay reached until the error doubles, so the walking command
            # does not chatter on and off while the robot sways at the tolerance boundary.
            position_scale = 2.0 if self._nav_position_reached else 1.0
            heading_scale = 2.0 if self._nav_heading_reached else 1.0
            position_reached = np.linalg.norm(error_xy) < position_scale * p.position_tolerance
            heading_reached = abs(heading_error) < heading_scale * p.heading_tolerance
            self._nav_position_reached = position_reached
            self._nav_heading_reached = heading_reached
            if not position_reached:
                c, s = math.cos(yaw), math.sin(yaw)
                scale = self._nav_speed_scale
                target[0] = np.clip(
                    p.kp_position * (c * error_xy[0] + s * error_xy[1]),
                    -p.max_backward_vel * scale,
                    p.max_forward_vel * scale,
                )
                target[1] = np.clip(
                    p.kp_position * (-s * error_xy[0] + c * error_xy[1]),
                    -p.max_lateral_vel * scale,
                    p.max_lateral_vel * scale,
                )
            if not heading_reached:
                scale = self._nav_turn_scale
                target[2] = np.clip(p.kp_heading * heading_error, -p.max_yaw_rate * scale, p.max_yaw_rate * scale)
            # Close to the goal, wait for the arms to finish rising before walking on.
            arms_moving = any(self._move_poses[side] for side in SIDES)
            if arms_moving and np.linalg.norm(error_xy) < p.wait_for_arms_distance:
                target[:] = 0.0
            # Lift small commands to the minimum speed so the walking policy does not stall near the goal.
            small = (np.abs(target) > 1e-6) & (np.abs(target) < p.min_vel)
            target[small] = np.sign(target[small]) * p.min_vel
        self._nav_cmd += p.velocity_smoothing * (target - self._nav_cmd)
        self._nav_cmd[np.abs(self._nav_cmd) < 1e-3] = 0.0
        if self._nav_goal is not None:
            # Arrived once inside the tolerances and, for goals to stop at, the walking command has wound down.
            wound_down = not self._nav_goal.stop or not self._nav_cmd.any()
            self._nav_arrived = position_reached and heading_reached and wound_down

    def _raise_arms_near_goal(self, pelvis_xy: np.ndarray) -> None:
        """While walking with hanging arms, raise them into the working pose once the walking goal is near."""
        if self._nav_goal is None or not self._arms_down:
            return
        if np.linalg.norm(self._nav_goal.xy - pelvis_xy) > self.params.raise_distance:
            return
        for side, (pos, quat) in self._working_wrist_poses().items():
            self._set_wrist_goal(side, pos, quat)
        self._arms_down = False

    def _lower_arms(self) -> None:
        """Lower both arms back into the attention pose."""
        for side in SIDES:
            self._set_wrist_goal(side, REST_WRIST_POS[side], REST_WRIST_QUAT_XYZW)
        self._arms_down = True

    def _start_wrist_moves(self, state: SceneState) -> None:
        """Start a move for every wrist whose goal changed: planned around obstacles, or straight."""
        changed = [
            side
            for side in SIDES
            if self._move_goal[side] is None
            or not np.array_equal(self._move_goal[side][0], self._wrist_goal[side])
            or not np.array_equal(self._move_goal[side][1], self._wrist_quat_goal[side])
        ]
        if not changed:
            return
        goals = {side: (self._wrist_goal[side].copy(), self._wrist_quat_goal[side].copy()) for side in SIDES}
        use_planner = self.planner is not None and self._phase in self.planned_phases and not self._force_straight
        self._force_straight = False
        if use_planner and state.arm_joint_pos is not None:
            # Move only the arms whose goal changed; the others stay locked where they are.
            trajectory = self.planner.plan(state, goals, self._planning_ignores(), tuple(changed))
            if trajectory is not None:
                self.num_planned_moves += 1
                for side in SIDES:
                    self._move_goal[side] = goals[side]
                    self._move_poses[side] = [poses[side] for poses in trajectory[1:]]
                return
            self.num_failed_plans += 1
        for side in changed:
            self._move_goal[side] = goals[side]
            self._move_poses[side] = self._straight_move(side, goals[side])

    def _install_planned_move(self, goals: dict[str, WristPose], trajectory: list[dict[str, WristPose]]) -> None:
        """Set ``goals`` as the wrist goals and follow an already planned trajectory to them."""
        self.num_planned_moves += 1
        for side in SIDES:
            self._wrist_goal[side] = np.asarray(goals[side][0], dtype=np.float64).copy()
            self._wrist_quat_goal[side] = np.asarray(goals[side][1], dtype=np.float64).copy()
            self._move_goal[side] = (self._wrist_goal[side].copy(), self._wrist_quat_goal[side].copy())
            self._move_poses[side] = [poses[side] for poses in trajectory[1:]]

    def _straight_move(self, side: str, goal: WristPose) -> list[WristPose]:
        """Wrist poses of a straight move from the commanded pose to ``goal`` with a minimum-jerk profile."""
        start_pos, start_quat = self._wrist_cmd[side].copy(), self._wrist_quat[side].copy()
        distance = float(np.linalg.norm(goal[0] - start_pos))
        angle = 2.0 * math.acos(min(abs(float(np.dot(start_quat, goal[1]))), 1.0))
        duration = max(distance / self.params.wrist_speed, angle / self.params.wrist_turn_speed)
        num_steps = int(math.ceil(duration / self.step_dt))
        poses = []
        for step in range(1, num_steps + 1):
            blend = min_jerk(step / num_steps)
            poses.append((start_pos + blend * (goal[0] - start_pos), slerp(start_quat, goal[1], blend)))
        return poses

    def _step_wrists(self) -> None:
        for side in SIDES:
            if self._move_poses[side]:
                position, quat = self._move_poses[side].pop(0)
                self._wrist_cmd[side] = np.asarray(position, dtype=np.float64).copy()
                self._wrist_quat[side] = np.asarray(quat, dtype=np.float64).copy()
            if not self._move_poses[side]:
                # The move is over: sit exactly on the goal, so the phase sees its targets reached.
                self._wrist_cmd[side] = self._wrist_goal[side].copy()
                self._wrist_quat[side] = self._wrist_quat_goal[side].copy()

    def _step_height(self) -> None:
        max_step = self.params.height_speed * self.step_dt
        delta = self._height_goal - self._height_cmd
        self._height_cmd = (
            self._height_goal if abs(delta) <= max_step else self._height_cmd + math.copysign(max_step, delta)
        )

    def _compose_action(self) -> np.ndarray:
        action = np.zeros(G1_LOCOMANIP_ACTION_DIM)
        action[LEFT_HAND_STATE_IDX] = float(self._hand_closed["left"])
        action[RIGHT_HAND_STATE_IDX] = float(self._hand_closed["right"])
        action[LEFT_WRIST_POS_START_IDX:LEFT_WRIST_POS_END_IDX] = self._wrist_cmd["left"]
        action[LEFT_WRIST_QUAT_START_IDX:LEFT_WRIST_QUAT_END_IDX] = self._wrist_quat["left"]
        action[RIGHT_WRIST_POS_START_IDX:RIGHT_WRIST_POS_END_IDX] = self._wrist_cmd["right"]
        action[RIGHT_WRIST_QUAT_START_IDX:RIGHT_WRIST_QUAT_END_IDX] = self._wrist_quat["right"]
        action[NAVIGATE_CMD_START_IDX:NAVIGATE_CMD_END_IDX] = self._nav_cmd
        action[BASE_HEIGHT_CMD_START_IDX] = self._height_cmd
        # Torso orientation stays upright (zeros).
        return action

    """
    Helpers.
    """

    def _set_wrist_goal(self, side: str, goal: np.ndarray, quat_xyzw: np.ndarray | None = None) -> None:
        """Set a wrist goal, pulled back onto the reach sphere around the shoulder when it is too far.

        Args:
            side: ``"left"`` or ``"right"``.
            goal: Wrist position in the pelvis frame [m].
            quat_xyzw: Wrist orientation in the pelvis frame, or None to keep the current orientation goal.
        """
        if quat_xyzw is not None:
            self._wrist_quat_goal[side] = np.asarray(quat_xyzw, dtype=np.float64).copy()
        shoulder = SHOULDER_POS[side]
        from_shoulder = goal - shoulder
        distance = np.linalg.norm(from_shoulder)
        if distance > self.params.max_reach:
            goal = shoulder + from_shoulder * (self.params.max_reach / distance)
        self._wrist_goal[side] = np.asarray(goal, dtype=np.float64).copy()

    def _shift_wrist_goals(self, delta: np.ndarray, sides: tuple[str, ...] = SIDES) -> None:
        for side in sides:
            self._set_wrist_goal(side, self._wrist_goal[side] + delta)
