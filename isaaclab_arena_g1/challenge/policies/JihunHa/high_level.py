# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""JihunHa high level: a fast closed-loop state machine on the privileged object, goal and pelvis poses.

Phases, each re-planned from the live poses every step (no stored commands or trajectories):

- approach: walk up to the counter while both hands already rise to beside the bottle.
- reach: slide the open hands in against the bottle's sides.
- close: squeeze the bottle lightly between the palms and close the fingers.
- carry: lift until the bottle clears the counter, then turn around and walk to the desk; the body turns as far
  as needed to bring the held bottle to the goal yaw (the grasp twists it in the hands).
- place: servo the wrists so the bottle sits on the goal, stepping or turning the body only when the arms cannot
  make up the error, and keep holding it (no release needed).

``two_hands: false`` switches to the one-hand (right) grasp, which is quicker to close but slips when the body turns.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, fields
from enum import Enum

from isaaclab_arena_g1.challenge import api
from isaaclab_arena_g1.challenge.interfaces import Command, HighLevelPolicy

SHOULDER_POS = {"left": np.array([0.0, 0.1, 0.292]), "right": np.array([0.0, -0.1, 0.292])}
"""Shoulder positions in the pelvis frame [m]."""

PALM_SURFACE_OFFSET = 0.017
"""Distance from the wrist axis to the palm surface [m]."""

REST_WRIST_POS = {"left": np.array([0.03, 0.20, -0.07]), "right": np.array([0.03, -0.20, -0.07])}
"""Wrists hanging at the sides, pelvis frame [m]."""

SIDE_SIGN = {"left": 1.0, "right": -1.0}
"""+1 for the left hand, -1 for the right one (the pelvis y axis points left)."""


class Phase(Enum):
    SETTLE = "settle"
    APPROACH = "approach"
    REACH = "reach"
    CLOSE = "close"
    CARRY = "carry"
    PLACE = "place"
    RETRY = "retry"


@dataclass
class JihunHaParams:
    """Tunable parameters; any of them can be set from the team file options."""

    settle_time: float = 0.1
    """Standing time after reset [s]."""

    approach_speed: float = 0.7
    """Walking speed limit while walking up to the counter [m/s]."""

    carry_speed: float = 1.0
    """Forward walking speed limit while carrying [m/s]."""

    max_backward_speed: float = 0.5
    """Backward walking speed limit [m/s]."""

    max_lateral_speed: float = 0.5
    """Sideways walking speed limit [m/s]."""

    max_yaw_rate: float = 1.0
    """Turning rate limit [rad/s]; faster turns swing the bottle out of the one-hand grasp."""

    turn_first_heading_error: float = 1.0
    """Heading error beyond which the walk mostly turns in place first [rad]; turning while side-stepping fast shakes
    the bottle loose."""

    turn_first_speed: float = 0.2
    """Walking speed limit while turning in place first [m/s]."""

    linear_accel: float = 1.5
    """Rate limit of the linear walking command [m/s^2]."""

    yaw_accel: float = 1.5
    """Rate limit of the turning command [rad/s^2]."""

    decel: float = 0.8
    """Deceleration the speed profile plans with when stopping at a walking goal [m/s^2]."""

    kp_position: float = 2.0
    """Gain from position error [m] to walking speed [m/s] close to a goal."""

    kp_heading: float = 2.0
    """Gain from heading error [rad] to turning rate [rad/s]."""

    min_walk_speed: float = 0.06
    """Smallest nonzero linear walking command, so the walking policy does not stall short of the goal [m/s]."""

    stop_tolerance: float = 0.02
    """Distance to a walking goal within which the linear walking command is zero [m]."""

    heading_stop_tolerance: float = 0.02
    """Heading error within which the turning command is zero [rad]."""

    approach_position_tolerance: float = 0.04
    """Pelvis distance to the grasp standoff at which the hand moves in [m]."""

    approach_heading_tolerance: float = 0.08
    """Heading error at the grasp standoff at which the hand moves in [rad]."""

    approach_stop_speed: float = 0.08
    """Pelvis speed below which the robot counts as stopped at the counter [m/s]; the hand moving in while the body
    still drifts knocks the bottle over."""

    wrist_settle_speed: float = 0.03
    """Speed of the actual wrist below which the raised hand counts as arrived [m/s]; it settles a few centimeters
    below its target, by how much depends on the pose."""

    wrist_settle_tolerance: float = 0.07
    """Largest distance of the settled wrist from its command [m]."""

    grasp_forward: float = 0.30
    """Forward distance from the pelvis to the bottle when grasping [m]."""

    grasp_lateral: float = 0.17
    """Rightward distance from the pelvis to the bottle when grasping with one hand [m]."""

    two_hands: bool = True
    """Squeeze the bottle between both palms in front of the body instead of holding it in the right hand; the
    one-hand grasp slips when the body turns."""

    two_hand_reach_depth: float = 0.01
    """How far each palm target goes into the bottle's side when the hands move in for the two-hand grasp [m];
    negative stops just short. Equal to two_hand_contact_depth, the palms squeeze as they arrive."""

    two_hand_contact_depth: float = 0.01
    """How far each palm target goes into the bottle's side once the fingers close in the two-hand grasp, so the
    arms squeeze it [m]; a softer squeeze (0.003) twists the bottle less but makes placing slower."""

    squeeze_wrist_speed: float = 0.1
    """Speed of the wrist commands while the palms squeeze in [m/s]."""

    two_hand_sequential: bool = False
    """In the two-hand grasp, take the bottle with the right hand first, as in the one-hand grasp, and only then
    bring the left hand in from its other side; both hands moving in at once knock the bottle about."""

    sequential_right_close_time: float = 0.5
    """Time the right fingers get to close before the left hand moves in [s]."""

    two_hand_close_fingers: bool = True
    """Close the fingers in the two-hand grasp too; the fingers of both hands meet in front of the bottle and can
    twist it."""

    thumb_clearance: float = 0.10
    """How far behind the bottle's near face the wrist sits when grasping [m]."""

    contact_depth: float = -0.005
    """How far the palm is pushed into the bottle's side in the one-hand grasp [m]; negative stops just short."""

    side_clearance: float = 0.09
    """Sideways gap between the open hand and the bottle before moving in [m]."""

    grasp_height_fraction: float = 0.4
    """Grasp height as a fraction of the bottle height."""

    min_grasp_height: float = 0.125
    """Lowest grasp height above the bottle's bottom, for the wrist command [m]; the arm sags a few centimeters below
    it."""

    compensate_wrist_offset: bool = False
    """Aim the grasp by where the wrist actually sits relative to its command once settled at the pre-grasp pose;
    the arm sags a few centimeters and reaches further forward than commanded, which puts the thumb into the
    bottle."""

    max_wrist_offset: float = 0.04
    """Largest wrist offset per axis the grasp compensates [m]."""

    grasp_pitch_deg: float = -20.0
    """Wrist pitch while grasping; negative tilts the fingers up, leveling them once the arm sags [deg]."""

    free_wrist_speed: float = 1.0
    """Speed of the wrist command for moves in free space [m/s]."""

    reach_wrist_speed: float = 0.4
    """Speed of the wrist command when moving in against the bottle [m/s]."""

    wrist_turn_speed: float = 3.0
    """Turning speed of the wrist orientation command [rad/s]."""

    reach_settle_time: float = 0.15
    """Time the hand rests against the bottle before closing [s]."""

    close_time: float = 1.0
    """Time given to the fingers to close before lifting with one hand [s]; lifted earlier, the grasp slips."""

    two_hand_close_time: float = 0.5
    """Time given to the fingers to close before lifting with both hands [s]; the palms already squeeze."""

    lift_height: float = 0.08
    """Wrist lift right after closing [m]."""

    lift_wrist_speed: float = 0.5
    """Speed of the wrist command while lifting [m/s]."""

    lift_timeout: float = 1.2
    """Time after closing within which the bottle must clear the counter, or the grasp is retried [s]."""

    retry_open_time: float = 0.3
    """Time the fingers are given to open before the hand backs off for another grasp [s]."""

    max_grasp_attempts: int = 3
    """Most grasp attempts before the policy stops retrying."""

    counter_clearance: float = 0.025
    """Height of the bottle's bottom above the counter top at which walking starts [m]."""

    counter_clear_distance: float = 0.35
    """Distance walked away from the grasp spot before the bottle is lowered toward the desk height [m]."""

    carry_clearance: float = 0.10
    """Height of the bottle above its goal height while carrying [m]; high enough that the forearms clear the desk
    edge while the body turns in front of it."""

    stuck_time: float = 1.0
    """Time the body may barely move while walking is commanded before it backs off [s]; turning at the desk can
    catch the arms on its edge."""

    stuck_backoff_time: float = 0.8
    """Time spent backing away from the desk when stuck [s]."""

    stuck_backoff_distance: float = 0.25
    """How far the body backs away from the desk when stuck [m]."""

    carry_pull_in: float = 0.0
    """How much closer to the body than where it hangs the placing standoff plans the bottle [m]; the arm reaches
    out by that much when placing."""

    place_forward_min: float = 0.30
    """Smallest forward distance from the pelvis to the goal when standing at the desk [m]."""

    min_desk_gap: float = 0.10
    """Smallest distance from the pelvis to the desk's near edge, measured square to the desk [m]; standing turned
    to fix the bottle's yaw otherwise brings the body into the desk."""

    max_body_yaw_correction_deg: float = 60.0
    """Largest deviation from facing the desk square-on that the body takes to fix the bottle's yaw [deg]."""

    turn_distance: float = 1.6
    """Distance to the placing standoff within which the body turns to its placing heading instead of the walking
    direction [m]; without pursuit."""

    pursuit: bool = False
    """Walk to the desk like a car, facing the way to go, through a waypoint in front of the placing standoff, so the
    body arrives turned to its placing heading; the walking policy side-steps slowly."""

    pursuit_offset: float = 0.8
    """Distance of the waypoint in front of the placing standoff, along the placing heading [m]."""

    pursuit_switch_distance: float = 0.45
    """Distance to the waypoint at which the walk heads on to the standoff [m]."""

    pursuit_final_distance: float = 0.25
    """Distance to the placing standoff within which the walk lines up precisely instead of steering [m]."""

    pursuit_lateral_gain: float = 0.3
    """Fraction of the sideways part of the way to go walked sideways while steering."""

    place_distance: float = 0.08
    """Distance to the placing standoff at which the placing servo takes over [m]."""

    held_filter_time: float = 0.3
    """Time constant of the filter on the held bottle's offset from the pelvis, which sets the placing standoff [s]."""

    place_speed: float = 0.25
    """Speed limit of the corrective steps while placing [m/s]."""

    place_servo_gain: float = 5.0
    """Gain of the bottle position servo on the wrist while placing [1/s]."""

    servo_deadband: float = 0.003
    """Bottle position error below which the servo leaves the wrist alone [m]."""

    rotation_servo_gain: float = 1.5
    """Gain of the bottle orientation servo on the wrist [1/s]."""

    rotation_servo_start_deg: float = 9.0
    """Bottle orientation error above which the wrist turns it [deg]; the walk's heading fixes the yaw, so the
    bottle usually arrives well inside the 15 deg tolerance."""

    rotation_servo_stop_deg: float = 5.0
    """Bottle orientation error at which the wrist stops turning it [deg]."""

    servo_max_speed: float = 0.4
    """Speed limit of the servo's wrist corrections [m/s]."""

    align_height: float = 0.015
    """Height above the goal the bottle is held at while it is moved over the goal [m]."""

    place_height_margin: float = 0.003
    """Height above the goal the bottle is lowered to [m]."""

    place_lower_xy_tolerance: float = 0.012
    """Horizontal bottle error below which the bottle is lowered onto the desk [m]."""

    freeze_position_tolerance: float = 0.015
    """Bottle position error below which the wrist goal is frozen, so the bottle stays still for the success hold
    [m]."""

    freeze_rotation_tolerance_deg: float = 10.0
    """Bottle orientation error below which the wrist goal may freeze [deg]."""

    unfreeze_position_tolerance: float = 0.024
    """Bottle position error above which a frozen wrist starts correcting again [m]."""

    unfreeze_rotation_tolerance_deg: float = 13.0
    """Bottle orientation error above which a frozen wrist starts correcting again [deg]."""

    step_position_error: float = 0.06
    """Horizontal bottle error while placing beyond which the body steps toward the goal [m]."""

    place_stall_time: float = 1.0
    """Time without the bottle's error shrinking after which the body steps toward the goal while placing [s]; the
    arms may be at the end of their reach."""

    place_stall_progress: float = 0.003
    """Error decrease that counts as progress while placing [m]."""

    step_heading_error_deg: float = 8.0
    """Bottle yaw error while placing beyond which the body turns to fix it [deg]."""

    lower_yaw_tolerance_deg: float = 8.0
    """Bottle yaw error below which the bottle may be lowered onto the desk [deg]."""

    still_speed: float = 0.08
    """Pelvis speed below which the body counts as standing still while placing [m/s]."""

    max_reach: float = 0.40
    """Largest wrist distance from the shoulder [m]."""

    pelvis_height: float = 0.74
    """Pelvis height command [m]."""

    debug: bool = False
    """Print the state twice a second."""


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _matrix_to_quat(R: np.ndarray) -> np.ndarray:
    trace = np.trace(R)
    if trace > 0.0:
        k = 2.0 * math.sqrt(trace + 1.0)
        q = np.array([(R[2, 1] - R[1, 2]) / k, (R[0, 2] - R[2, 0]) / k, (R[1, 0] - R[0, 1]) / k, 0.25 * k])
    else:
        i = int(np.argmax(np.diag(R)))
        j, m = (i + 1) % 3, (i + 2) % 3
        k = 2.0 * math.sqrt(1.0 + R[i, i] - R[j, j] - R[m, m])
        q = np.zeros(4)
        q[i] = 0.25 * k
        q[j] = (R[j, i] + R[i, j]) / k
        q[m] = (R[m, i] + R[i, m]) / k
        q[3] = (R[m, j] - R[j, m]) / k
    q /= np.linalg.norm(q)
    return q if q[3] >= 0.0 else -q


def _rotvec_to_matrix(v: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(v))
    if angle < 1e-9:
        return np.eye(3)
    a = v / angle
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * K @ K


def _matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    cos_angle = min(max((np.trace(R) - 1.0) / 2.0, -1.0), 1.0)
    angle = math.acos(cos_angle)
    if angle < 1e-6:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        # Half turn: take the axis from the symmetric part.
        axis = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        return axis * angle
    return axis / norm * angle


def _yaw(q: np.ndarray) -> float:
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _pitch_quat(pitch_deg: float) -> np.ndarray:
    half = 0.5 * math.radians(pitch_deg)
    return np.array([0.0, math.sin(half), 0.0, math.cos(half)])


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    angle = math.acos(dot)
    return (math.sin((1.0 - t) * angle) * q0 + math.sin(t * angle) * q1) / math.sin(angle)


def _np(tensor) -> np.ndarray:
    return tensor[0].detach().cpu().numpy().astype(np.float64)


class JihunHaHighLevel(HighLevelPolicy):
    """Fast closed-loop pick, carry and place of the bottle with the right hand."""

    def __init__(self, **options):
        super().__init__(**options)
        names = {field.name for field in fields(JihunHaParams)}
        unknown = set(options) - names
        assert not unknown, f"Unknown JihunHa options: {sorted(unknown)}"
        self.p = JihunHaParams(**options)
        self.reset()

    def reset(self) -> None:
        self.phase = Phase.SETTLE
        self._phase_start = 0.0
        self._initialized = False
        self._nav_cmd = np.zeros(3)
        self._sides = ("left", "right") if self.p.two_hands else ("right",)
        self._wrist_pos = {side: REST_WRIST_POS[side].copy() for side in SIDE_SIGN}
        self._wrist_quat = {side: _pitch_quat(90.0) for side in SIDE_SIGN}
        self._wrist_goal = {side: self._wrist_pos[side].copy() for side in SIDE_SIGN}
        self._wrist_goal_quat = {side: self._wrist_quat[side].copy() for side in SIDE_SIGN}
        self._grip = False
        self._place_best_error = float("inf")
        self._place_progress_time = 0.0
        self._stuck_since: float | None = None
        self._backoff_until: float | None = None
        self._backoff_xy = np.zeros(2)
        self._wrist_offset = {side: np.zeros(3) for side in SIDE_SIGN}
        self._left_grip = False
        self._close_stage = 0
        self._squeeze_start = 0.0
        self._arrived = False
        self._passed_waypoint = False
        self._wrist_actual: dict[str, np.ndarray] | None = None
        self._wrist_actual_speed = {side: 1.0 for side in SIDE_SIGN}
        self._reached_time: float | None = None
        self._moving = False
        self._lowered = False
        self._held_xy = np.zeros(2)
        self._held_yaw = 0.0
        self._place_heading = 0.0
        self._frozen = False
        self._stepping = False
        self._lowering = False
        self._grasp_attempts = 0
        self._rotating = False
        self._last_debug = -1.0

    # -----------------------------------------------------------------------------------------------------------------
    # Main step.
    # -----------------------------------------------------------------------------------------------------------------

    def compute_command(self, env, observation) -> Command:
        p = self.p
        self._dt = api.control_dt(env)
        self._t = float(api.elapsed_time(env)[0])
        pelvis = _np(api.pelvis_pose(env))
        self._pelvis_pos = pelvis[:3]
        self._yaw = _yaw(pelvis[3:7])
        pelvis_velocity = _np(api.pelvis_velocity(env))
        self._pelvis_speed = float(np.linalg.norm(pelvis_velocity[:2]))
        self._pelvis_yaw_rate = float(pelvis_velocity[5])
        object_pose = _np(api.object_pose(env))
        self._obj_pos = object_pose[:3]
        self._obj_quat = object_pose[3:7]
        self._obj_corners = _np(api.object_corners(env))
        wrist_actual = {side: _np(api.wrist_pose_in_pelvis(env, side))[:3] for side in self._sides}
        if self._wrist_actual is not None:
            for side in self._sides:
                speed = float(np.linalg.norm(wrist_actual[side] - self._wrist_actual[side])) / self._dt
                self._wrist_actual_speed[side] += 0.2 * (speed - self._wrist_actual_speed[side])
        self._wrist_actual = wrist_actual
        goal = _np(api.goal_pose(env))
        self._goal_pos = goal[:3]
        self._goal_quat = goal[3:7]
        if not self._initialized:
            self._initialize(env)

        walk_target = None
        if self.phase == Phase.SETTLE:
            if self._t >= p.settle_time:
                self._enter(Phase.APPROACH)
        if self.phase == Phase.APPROACH:
            walk_target = self._step_approach()
        elif self.phase == Phase.REACH:
            self._step_reach()
        elif self.phase == Phase.CLOSE:
            self._step_close()
        elif self.phase == Phase.CARRY:
            walk_target = self._step_carry()
        elif self.phase == Phase.PLACE:
            walk_target = self._step_place()
        elif self.phase == Phase.RETRY:
            self._step_retry()

        self._step_walk(walk_target)
        self._step_wrist()
        if p.debug and self._t - self._last_debug >= 0.5:
            self._last_debug = self._t
            self._print_debug(env)
        return Command.from_tensor(
            api.make_action(
                env,
                left_wrist_pos=self._wrist_pos["left"],
                left_wrist_quat=self._wrist_quat["left"],
                right_wrist_pos=self._wrist_pos["right"],
                right_wrist_quat=self._wrist_quat["right"],
                left_hand_closed=self._grip and self._left_grip and p.two_hands and p.two_hand_close_fingers,
                right_hand_closed=self._grip and (not p.two_hands or p.two_hand_close_fingers),
                walk_forward=self._nav_cmd[0],
                walk_left=self._nav_cmd[1],
                turn_rate=self._nav_cmd[2],
                pelvis_height=p.pelvis_height,
            )
        )

    def _initialize(self, env) -> None:
        p = self.p
        self._initialized = True
        self._start_heading = self._yaw
        self._counter_top = float(_np(api.counter_top_corners(env))[:, 2].max())
        self._desk_corners = _np(api.desk_corners(env))
        bottom = self._obj_corners[:, 2].min()
        self._root_above_bottom = float(self._obj_pos[2] - bottom)
        self._object_height = float(self._obj_corners[:, 2].max() - bottom)
        # Stand so that the bottle sits grasp_forward ahead (and, for one hand, grasp_lateral to the right).
        center = self._obj_corners.mean(axis=0)
        lateral = 0.0 if p.two_hands and not p.two_hand_sequential else p.grasp_lateral
        self._grasp_stand = center[:2] + _rot_z(self._start_heading)[:2, :2] @ np.array([-p.grasp_forward, lateral])
        # Face the desk square-on: turned around from the start heading.
        self._desk_heading = _wrap(self._start_heading + math.pi)

    def _enter(self, phase: Phase) -> None:
        self.phase = phase
        self._phase_start = self._t
        self._reached_time = None
        if phase == Phase.APPROACH:
            # Raise the open hands to beside the bottle as it will be seen from the grasp stand.
            corners = self._to_frame(self._obj_corners, self._grasp_stand, self._start_heading)
            for side in self._sides:
                self._wrist_goal[side] = self._clamp_reach(self._pregrasp_wrist(corners, side), side)
                self._wrist_goal_quat[side] = _pitch_quat(self.p.grasp_pitch_deg)
        elif phase == Phase.RETRY:
            self._grip = False
            self._left_grip = False
        elif phase == Phase.REACH:
            self._grasp_attempts += 1
            if self.p.compensate_wrist_offset:
                limit = self.p.max_wrist_offset
                for side in self._sides:
                    offset = self._wrist_actual[side] - self._wrist_pos[side]
                    self._wrist_offset[side] = np.clip(offset, -limit, limit)
            # Re-plan the grasp from the live bottle, absorbing where the robot actually stopped.
            corners = self._to_heading(self._obj_corners)
            sides = ("right",) if self._sequential() else self._sides
            for side in sides:
                self._wrist_goal[side] = self._aim(side, self._grasp_wrist(corners, side))
        elif phase == Phase.CLOSE:
            self._grip = True
            self._close_stage = 0
            if self.p.two_hands and not self._sequential():
                self._squeeze()
        elif phase == Phase.PLACE:
            self._place_best_error = float("inf")
            self._place_progress_time = self._t
        elif phase == Phase.CARRY:
            self._shift_wrist_goals(np.array([0.0, 0.0, self.p.lift_height]))
            self._moving = False
            self._lowered = False

    # -----------------------------------------------------------------------------------------------------------------
    # Phases.
    # -----------------------------------------------------------------------------------------------------------------

    def _step_approach(self):
        p = self.p
        self._wrist_speed = p.free_wrist_speed
        distance = float(np.linalg.norm(self._grasp_stand - self._pelvis_pos[:2]))
        heading_error = abs(_wrap(self._start_heading - self._yaw))
        # Once inside the tolerances, stop walking (with hysteresis): creeping at the smallest walking command, the
        # body never comes to rest.
        if self._arrived:
            self._arrived = (
                distance < 2.0 * p.approach_position_tolerance and heading_error < 2.0 * p.approach_heading_tolerance
            )
        else:
            self._arrived = distance < p.approach_position_tolerance and heading_error < p.approach_heading_tolerance
        if not self._arrived:
            return (self._grasp_stand, self._start_heading, p.approach_speed, False, False)
        if (
            self._pelvis_speed < p.approach_stop_speed
            and self._wrist_reached()
            and all(
                self._wrist_actual_speed[side] < p.wrist_settle_speed
                and float(np.linalg.norm(self._wrist_actual[side] - self._wrist_pos[side])) < p.wrist_settle_tolerance
                for side in self._sides
            )
        ):
            self._enter(Phase.REACH)
        return None

    def _step_reach(self) -> None:
        self._wrist_speed = self.p.reach_wrist_speed
        if self._wrist_reached():
            if self._reached_time is None:
                self._reached_time = self._t
            if self._t - self._reached_time >= self.p.reach_settle_time:
                self._enter(Phase.CLOSE)

    def _step_retry(self) -> None:
        """The bottle did not come up: open the hand, back it off beside the bottle and reach again."""
        p = self.p
        if self._t - self._phase_start < p.retry_open_time:
            return
        if self._reached_time is None:
            self._reached_time = self._t
            corners = self._to_heading(self._obj_corners)
            for side in self._sides:
                self._wrist_goal[side] = self._clamp_reach(self._pregrasp_wrist(corners, side), side)
            self._wrist_speed = p.reach_wrist_speed
        elif self._wrist_reached():
            self._enter(Phase.REACH)

    def _step_close(self) -> None:
        p = self.p
        if not self._sequential():
            close_time = p.two_hand_close_time if p.two_hands else p.close_time
            if self._t - self._phase_start >= close_time:
                self._enter(Phase.CARRY)
            return
        # Right fingers close; then the left hand moves in from the bottle's other side and both palms squeeze.
        if self._close_stage == 0 and self._t - self._phase_start >= p.sequential_right_close_time:
            corners = self._to_heading(self._obj_corners)
            self._wrist_goal["left"] = self._aim("left", self._grasp_wrist(corners, "left"))
            self._wrist_speed = p.reach_wrist_speed
            self._close_stage = 1
        elif self._close_stage == 1 and self._wrist_reached():
            self._squeeze()
            self._squeeze_start = self._t
            self._close_stage = 2
        elif self._close_stage == 2 and self._t - self._squeeze_start >= p.two_hand_close_time:
            self._enter(Phase.CARRY)

    def _sequential(self) -> bool:
        return self.p.two_hands and self.p.two_hand_sequential

    def _squeeze(self) -> None:
        """Push both palm targets into the bottle so the arms squeeze it, and close the fingers."""
        squeeze = self.p.two_hand_contact_depth - self.p.two_hand_reach_depth
        for side in self._sides:
            goal = self._wrist_goal[side].copy()
            goal[1] -= SIDE_SIGN[side] * squeeze
            self._wrist_goal[side] = self._clamp_reach(goal, side)
        self._wrist_speed = self.p.squeeze_wrist_speed
        self._left_grip = True

    def _step_carry(self):
        p = self.p
        bottom = self._obj_corners[:, 2].min()
        if not self._moving:
            self._wrist_speed = p.lift_wrist_speed
            if bottom < self._counter_top + p.counter_clearance:
                if self._t - self._phase_start > p.lift_timeout and self._grasp_attempts < p.max_grasp_attempts:
                    self._enter(Phase.RETRY)
                return None
            self._start_moving()
        self._wrist_speed = p.lift_wrist_speed
        # Track where the bottle actually hangs from the pelvis (it can shift in the hand), filtered against swing.
        held = self._to_heading(self._obj_pos[None, :])[0][:2]
        blend = min(1.0, self._dt / p.held_filter_time)
        self._held_xy += blend * (held - self._held_xy)
        self._held_yaw = _wrap(self._held_yaw + blend * _wrap(_yaw(self._obj_quat) - self._yaw - self._held_yaw))
        self._place_heading = self._placing_heading()
        if not self._lowered and np.linalg.norm(self._pelvis_pos[:2] - self._grasp_stand) > p.counter_clear_distance:
            # Clear of the counter: lower the bottle to just above the goal height in one move.
            self._lowered = True
            drop = self._goal_pos[2] + p.carry_clearance - self._obj_pos[2]
            self._shift_wrist_goals(np.array([0.0, 0.0, drop]))
        stand, heading = self._place_stand()
        distance = float(np.linalg.norm(stand - self._pelvis_pos[:2]))
        if distance < p.place_distance:
            self._enter(Phase.PLACE)
            return None
        backoff = self._stuck_backoff()
        if backoff is not None:
            return backoff
        if p.pursuit:
            if distance > p.pursuit_final_distance:
                toward_desk = np.array([math.cos(heading), math.sin(heading)])
                waypoint = stand - p.pursuit_offset * toward_desk
                to_waypoint = waypoint - self._pelvis_pos[:2]
                # Past the waypoint once close to it or beyond its line across the way in.
                if not self._passed_waypoint and (
                    np.linalg.norm(to_waypoint) < p.pursuit_switch_distance or float(to_waypoint @ toward_desk) < 0.0
                ):
                    self._passed_waypoint = True
                if self._passed_waypoint:
                    return (stand, None, p.carry_speed, True, True, distance)
                remaining = float(np.linalg.norm(to_waypoint)) + p.pursuit_offset
                return (waypoint, None, p.carry_speed, True, True, remaining)
            return (stand, heading, p.carry_speed, True, False)
        if distance > p.turn_distance:
            # Far from the desk, face the way to go; turn in place first so the turn does not shake the bottle.
            travel = stand - self._pelvis_pos[:2]
            return (stand, math.atan2(travel[1], travel[0]), p.carry_speed, True, True)
        return (stand, heading, p.carry_speed, True, False)

    def _step_place(self):
        p = self.p
        self._wrist_speed = p.free_wrist_speed
        offset = self._goal_pos - self._obj_pos
        position_error = float(np.linalg.norm(offset))
        horizontal_error = float(np.linalg.norm(offset[:2]))
        rotation_error = math.degrees(self._rotation_error())
        # Inside the tolerances, keep the wrist goal still so the bottle comes to rest for the success hold.
        still = self._pelvis_speed < p.still_speed and not self._nav_cmd.any()
        if self._frozen:
            if position_error > p.unfreeze_position_tolerance or rotation_error > p.unfreeze_rotation_tolerance_deg:
                self._frozen = False
        elif (
            still
            and position_error < p.freeze_position_tolerance
            and rotation_error < p.freeze_rotation_tolerance_deg
        ):
            self._frozen = True

        # Track progress, so the body can step in when the arms are stuck at the end of their reach.
        if self._frozen or position_error < self._place_best_error - p.place_stall_progress:
            self._place_best_error = position_error
            self._place_progress_time = self._t
        stalled = not self._frozen and self._t - self._place_progress_time > p.place_stall_time
        # The body stands still unless the arm cannot make up the error.
        stalled_off_goal = stalled and horizontal_error > 0.5 * p.place_lower_xy_tolerance
        if horizontal_error > p.step_position_error or stalled_off_goal:
            self._stepping = True
        elif horizontal_error < 0.5 * p.step_position_error and not stalled:
            self._stepping = False
        elif horizontal_error < 0.5 * p.place_lower_xy_tolerance:
            self._stepping = False
            self._place_best_error = position_error
            self._place_progress_time = self._t
        yaw_error = _wrap(_yaw(self._goal_quat) - _yaw(self._obj_quat))
        walk = None
        if self._stepping or abs(math.degrees(yaw_error)) > p.step_heading_error_deg:
            target_xy = self._pelvis_pos[:2] + (offset[:2] if self._stepping else 0.0)
            walk = (target_xy, self._yaw + yaw_error, p.place_speed, False, False)

        if not self._frozen:
            # Turn and move the bottle while it hangs above the desk; lower it only once it is over the goal at the
            # goal yaw and the body stands still, since dragging it on the desk tips it.
            if (
                still
                and walk is None
                and horizontal_error < p.place_lower_xy_tolerance
                and abs(math.degrees(yaw_error)) < p.lower_yaw_tolerance_deg
            ):
                self._lowering = True
            target = self._goal_pos.copy()
            target[2] += p.place_height_margin if self._lowering else p.align_height
            # Once down, never raise the wrist: a bottle resting on the desk does not follow, the fingers slide up it.
            self._servo_position(target, p.place_servo_gain, raise_wrist=not self._lowering)
            if rotation_error > p.rotation_servo_start_deg:
                self._rotating = True
            elif rotation_error < p.rotation_servo_stop_deg:
                self._rotating = False
            if self._rotating and not self._lowering and not p.two_hands:
                self._servo_orientation(_quat_to_matrix(self._goal_quat))
        return walk

    def _rotation_error(self) -> float:
        """Angle between the bottle's orientation and the goal orientation [rad]."""
        R = _quat_to_matrix(self._goal_quat) @ _quat_to_matrix(self._obj_quat).T
        return float(np.linalg.norm(_matrix_to_rotvec(R)))

    def _stuck_backoff(self):
        """Walking target that backs away from the desk while the body is stuck, or None."""
        p = self.p
        if self._backoff_until is not None:
            if self._t < self._backoff_until:
                return (self._backoff_xy, self._yaw, p.place_speed, False, False)
            self._backoff_until = None
            self._stuck_since = None
        commanded = float(np.linalg.norm(self._nav_cmd[:2])) > 0.15 or abs(self._nav_cmd[2]) > 0.3
        moving = self._pelvis_speed > 0.06 or abs(self._pelvis_yaw_rate) > 0.15
        if not commanded or moving:
            self._stuck_since = None
            return None
        if self._stuck_since is None:
            self._stuck_since = self._t
            return None
        if self._t - self._stuck_since < p.stuck_time:
            return None
        toward_desk = np.array([math.cos(self._desk_heading), math.sin(self._desk_heading)])
        self._backoff_xy = self._pelvis_pos[:2] - p.stuck_backoff_distance * toward_desk
        self._backoff_until = self._t + p.stuck_backoff_time
        return (self._backoff_xy, self._yaw, p.place_speed, False, False)

    def _start_moving(self) -> None:
        """The bottle has cleared the counter: start tracking where it is held and the heading to place it with."""
        self._moving = True
        self._held_xy = self._to_heading(self._obj_pos[None, :])[0][:2]
        self._held_yaw = _wrap(_yaw(self._obj_quat) - self._yaw)
        self._place_heading = self._placing_heading()

    def _placing_heading(self) -> float:
        """Heading at the desk that turns the held bottle to the goal yaw, as far from square-on as allowed; the
        grasp twists the bottle in the hand, and the walk can turn it further."""
        needed = _wrap(_yaw(self._goal_quat) - self._held_yaw)
        limit = math.radians(self.p.max_body_yaw_correction_deg)
        correction = float(np.clip(_wrap(needed - self._desk_heading), -limit, limit))
        return _wrap(self._desk_heading + correction)

    def _place_stand(self) -> tuple[np.ndarray, float]:
        """Pelvis position and heading at the desk that put the held bottle over the goal."""
        held = np.array([max(self._held_xy[0] - self.p.carry_pull_in, self.p.place_forward_min), self._held_xy[1]])
        stand = self._goal_pos[:2] - _rot_z(self._place_heading)[:2, :2] @ held
        # Keep clear of the desk's near edge; the arm reaches out the rest.
        toward_desk = np.array([math.cos(self._desk_heading), math.sin(self._desk_heading)])
        gap = float(np.min((self._desk_corners[:, :2] - stand) @ toward_desk))
        if gap < self.p.min_desk_gap:
            stand = stand - toward_desk * (self.p.min_desk_gap - gap)
        return stand, self._place_heading

    # -----------------------------------------------------------------------------------------------------------------
    # Wrist.
    # -----------------------------------------------------------------------------------------------------------------

    def _contact_depth(self) -> float:
        """Palm depth into the bottle's side when the hands move in."""
        return self.p.two_hand_reach_depth if self.p.two_hands else self.p.contact_depth

    def _grasp_wrist(self, corners: np.ndarray, side: str) -> np.ndarray:
        """Wrist position with the palm against the bottle's ``side`` side, from its corners in a heading frame."""
        p = self.p
        sign = SIDE_SIGN[side]
        bottom, top = corners[:, 2].min(), corners[:, 2].max()
        side_y = corners[:, 1].max() if sign > 0 else corners[:, 1].min()
        return np.array([
            corners[:, 0].min() - p.thumb_clearance,
            side_y + sign * (PALM_SURFACE_OFFSET - self._contact_depth()),
            bottom + max(p.min_grasp_height, p.grasp_height_fraction * (top - bottom)),
        ])

    def _aim(self, side: str, wrist: np.ndarray) -> np.ndarray:
        """Wrist command that puts the actual wrist at ``wrist``, by the offset measured at the pre-grasp pose."""
        return self._clamp_reach(wrist - self._wrist_offset[side], side)

    def _pregrasp_wrist(self, corners: np.ndarray, side: str) -> np.ndarray:
        """Grasp wrist position backed off sideways, so the open hand clears the bottle."""
        pregrasp = self._grasp_wrist(corners, side)
        pregrasp[1] += SIDE_SIGN[side] * (self.p.side_clearance + self._contact_depth())
        return pregrasp

    def _shift_wrist_goals(self, delta: np.ndarray) -> None:
        for side in self._sides:
            self._wrist_goal[side] = self._clamp_reach(self._wrist_goal[side] + delta, side)

    def _servo_position(self, target_w: np.ndarray, gain: float, raise_wrist: bool = True) -> None:
        """Move the wrist goal so the bottle's root approaches ``target_w`` (world); with ``raise_wrist`` False the
        wrist goal only moves sideways or down."""
        error = _rot_z(self._yaw).T @ (target_w - self._obj_pos)
        if not raise_wrist:
            error[2] = min(error[2], 0.0)
        if np.linalg.norm(error) < self.p.servo_deadband:
            return
        step = gain * self._dt * error
        limit = self.p.servo_max_speed * self._dt
        norm = float(np.linalg.norm(step))
        if norm > limit:
            step *= limit / norm
        self._shift_wrist_goals(step)

    def _servo_orientation(self, desired_w: np.ndarray) -> None:
        """Turn the wrist goal so the bottle's orientation approaches ``desired_w`` (world rotation matrix)."""
        error_w = _matrix_to_rotvec(desired_w @ _quat_to_matrix(self._obj_quat).T)
        error_p = _rot_z(self._yaw).T @ error_w
        step = self.p.rotation_servo_gain * self._dt * error_p
        limit = self.p.wrist_turn_speed * self._dt
        norm = float(np.linalg.norm(step))
        if norm > limit:
            step *= limit / norm
        R = _rotvec_to_matrix(step) @ _quat_to_matrix(self._wrist_goal_quat["right"])
        self._wrist_goal_quat["right"] = _matrix_to_quat(R)

    def _clamp_reach(self, goal: np.ndarray, side: str) -> np.ndarray:
        shoulder = SHOULDER_POS[side]
        from_shoulder = goal - shoulder
        distance = float(np.linalg.norm(from_shoulder))
        if distance > self.p.max_reach:
            goal = shoulder + from_shoulder * (self.p.max_reach / distance)
        return goal

    def _wrist_reached(self) -> bool:
        return all(
            np.linalg.norm(self._wrist_pos[side] - self._wrist_goal[side]) < 1e-3
            and abs(float(np.dot(self._wrist_quat[side], self._wrist_goal_quat[side]))) > 1.0 - 1e-6
            for side in self._sides
        )

    def _step_wrist(self) -> None:
        """Move the wrist commands toward their goals at the phase's speed."""
        speed = getattr(self, "_wrist_speed", self.p.free_wrist_speed)
        max_step = speed * self._dt
        max_turn = self.p.wrist_turn_speed * self._dt
        for side in self._sides:
            delta = self._wrist_goal[side] - self._wrist_pos[side]
            distance = float(np.linalg.norm(delta))
            if distance <= max_step:
                self._wrist_pos[side] = self._wrist_goal[side].copy()
            else:
                self._wrist_pos[side] = self._wrist_pos[side] + delta * (max_step / distance)
            dot = min(abs(float(np.dot(self._wrist_quat[side], self._wrist_goal_quat[side]))), 1.0)
            angle = 2.0 * math.acos(dot)
            if angle <= max_turn:
                self._wrist_quat[side] = self._wrist_goal_quat[side].copy()
            else:
                self._wrist_quat[side] = _slerp(self._wrist_quat[side], self._wrist_goal_quat[side], max_turn / angle)

    # -----------------------------------------------------------------------------------------------------------------
    # Walking.
    # -----------------------------------------------------------------------------------------------------------------

    def _step_walk(self, target) -> None:
        """Rate-limited walking command toward ``target`` = (xy, heading, speed limit, turn counterclockwise, turn in
        place first when the heading is far off), or standing still when None."""
        p = self.p
        desired = np.zeros(3)
        if target is not None and target[1] is None:
            self._steer(target, desired)
        elif target is not None:
            xy, heading, speed_limit, ccw, turn_first = target
            error = xy - self._pelvis_pos[:2]
            distance = float(np.linalg.norm(error))
            speed = min(speed_limit, math.sqrt(2.0 * p.decel * distance), p.kp_position * distance)
            if distance > p.stop_tolerance:
                speed = max(speed, p.min_walk_speed)
                velocity = _rot_z(self._yaw)[:2, :2].T @ (error / distance * speed)
                scale = 1.0
                if velocity[0] > speed_limit:
                    scale = min(scale, speed_limit / velocity[0])
                if velocity[0] < -p.max_backward_speed:
                    scale = min(scale, p.max_backward_speed / -velocity[0])
                if abs(velocity[1]) > p.max_lateral_speed:
                    scale = min(scale, p.max_lateral_speed / abs(velocity[1]))
                desired[:2] = velocity * scale
            heading_error = _wrap(heading - self._yaw)
            if ccw and heading_error < -(math.pi - 0.6):
                # Keep turning counterclockwise through the half turn, away from the counter.
                heading_error += 2.0 * math.pi
            if turn_first and abs(heading_error) > p.turn_first_heading_error:
                linear = float(np.linalg.norm(desired[:2]))
                if linear > p.turn_first_speed:
                    desired[:2] *= p.turn_first_speed / linear
            if abs(heading_error) > p.heading_stop_tolerance:
                desired[2] = float(np.clip(p.kp_heading * heading_error, -p.max_yaw_rate, p.max_yaw_rate))
        self._limit_walk_command(desired)

    def _steer(self, target, desired: np.ndarray) -> None:
        """Car-like walk toward ``target`` = (aim xy, None, speed limit, turn counterclockwise, turn in place first,
        distance left to stop): face the aim point and walk mostly forward."""
        p = self.p
        aim, _, speed_limit, ccw, turn_first, remaining = target
        error = aim - self._pelvis_pos[:2]
        heading_error = _wrap(math.atan2(error[1], error[0]) - self._yaw)
        if ccw and heading_error < -(math.pi - 0.6):
            heading_error += 2.0 * math.pi
        speed = min(speed_limit, math.sqrt(2.0 * p.decel * remaining))
        if turn_first and abs(heading_error) > p.turn_first_heading_error:
            speed = min(speed, p.turn_first_speed)
        desired[0] = speed * max(math.cos(heading_error), 0.0) if abs(heading_error) < math.pi / 2 else 0.0
        desired[1] = float(
            np.clip(p.pursuit_lateral_gain * speed * math.sin(heading_error), -p.max_lateral_speed, p.max_lateral_speed)
        )
        if abs(heading_error) > math.pi / 2:
            # Aim point behind: turn in place, drifting toward it.
            desired[0] = 0.0
            drift = math.sin(heading_error) * p.turn_first_speed
            desired[1] = float(np.clip(drift, -p.turn_first_speed, p.turn_first_speed))
        desired[2] = float(np.clip(p.kp_heading * heading_error, -p.max_yaw_rate, p.max_yaw_rate))

    def _limit_walk_command(self, desired: np.ndarray) -> None:
        """Move the walking command toward ``desired`` within the acceleration limits."""
        p = self.p
        max_linear = p.linear_accel * self._dt
        max_angular = p.yaw_accel * self._dt
        delta = desired - self._nav_cmd
        delta[:2] = np.clip(delta[:2], -max_linear, max_linear)
        delta[2] = float(np.clip(delta[2], -max_angular, max_angular))
        self._nav_cmd = self._nav_cmd + delta
        self._nav_cmd[np.abs(self._nav_cmd) < 1e-3] = 0.0

    # -----------------------------------------------------------------------------------------------------------------
    # Frames.
    # -----------------------------------------------------------------------------------------------------------------

    def _to_frame(self, points_w: np.ndarray, origin_xy: np.ndarray, heading: float) -> np.ndarray:
        """World points in a frame at ``origin_xy`` (height of the pelvis) turned by ``heading``."""
        origin = np.array([origin_xy[0], origin_xy[1], self._pelvis_pos[2]])
        return (np.atleast_2d(points_w) - origin) @ _rot_z(heading)

    def _to_heading(self, points_w: np.ndarray) -> np.ndarray:
        """World points in the pelvis heading frame."""
        return self._to_frame(points_w, self._pelvis_pos[:2], self._yaw)

    def _from_heading(self, point_h: np.ndarray) -> np.ndarray:
        """A pelvis heading frame point in the world."""
        return self._pelvis_pos + _rot_z(self._yaw) @ point_h

    def _print_debug(self, env) -> None:
        wrist = _np(api.wrist_pose_in_pelvis(env, "right"))[:3]
        obj_h = self._to_heading(self._obj_pos[None, :])[0]
        position_error, rotation_error = api.goal_error(env)
        print(
            f"[JihunHa] t={self._t:5.2f} {self.phase.value:<8} nav=({self._nav_cmd[0]:+.2f},{self._nav_cmd[1]:+.2f},"
            f"{self._nav_cmd[2]:+.2f}) v={self._pelvis_speed:.2f} wrist_cmd={np.round(self._wrist_pos['right'], 3)}"
            f" wrist={np.round(wrist, 3)} err={float(position_error[0]) * 100:.1f}cm"
            f" {math.degrees(float(rotation_error[0])):.1f}deg obj_h={np.round(obj_h, 3)}"
            f" obj-wrist={np.round(obj_h - wrist, 3)} held={np.round(self._held_xy, 3)}"
            f" hyaw={math.degrees(self._held_yaw):.0f} low={int(self._lowering)} frz={int(self._frozen)}"
            f" stp={int(self._stepping)} rot={int(self._rotating)}",
            flush=True,
        )
