# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""State-machine expert that moves a small object into a container with one G1 hand, driven by privileged poses.

The expert starts standing with its arms at its sides, raises its hands above counter height while walking
up to the object, slides one open hand alongside it, closes the fingers around it, lifts it, side-steps
until the object is above the container, lowers it in, lets go and drops its arms back to its sides. The
other hand stays raised and tucked in out of the way.

The module only depends on numpy, so the state machine can be exercised without Isaac Sim.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from enum import Enum

from isaaclab_arena_g1.policy.g1_scripted_expert_base import (
    HOME_WRIST_POS,
    IDENTITY_QUAT_XYZW,
    PALM_SURFACE_OFFSET,
    REST_WRIST_POS,
    REST_WRIST_QUAT_XYZW,
    SIDES,
    NavigationGoal,
    SceneState,
    ScriptedExpertParams,
    ScriptedLocomanipExpertBase,
    WristPose,
    heading_to_world_xy,
    matrix_to_quat_xyzw,
    pitch_quat_xyzw,
    quat_xyzw_to_matrix,
    unit_xy,
    world_to_heading_frame,
)


class OneHandPhase(Enum):
    """Phases of the scripted one-hand pick-and-place, executed in definition order up to LOWER.

    A destination on the same counter is reached by side-stepping; for one elsewhere (e.g. on the floor) the
    expert also backs off the counter (BACK_OFF), walks over facing it (CARRY), squats to its height (SQUAT) and
    stands up again after letting go (STAND). After a failed grasp the expert opens its hand in place (RETRY_OPEN), lifts it straight up above the object
    (RETRY_UP), moves it back to the raised pose around the object (RETRY_BACK) and starts over at APPROACH.
    """

    SETTLE = "settle"
    APPROACH = "approach"
    REACH_SIDE = "reach_side"
    REACH_FORWARD = "reach_forward"
    CONTACT = "contact"
    CLOSE = "close"
    LIFT = "lift"
    BACK_OFF = "back_off"
    CARRY = "carry"
    SQUAT = "squat"
    PLACE_OVER = "place_over"
    PLACE_DOWN = "place_down"
    RELEASE = "release"
    OPEN_AWAY = "open_away"
    RETRACT = "retract"
    STAND = "stand"
    LOWER = "lower"
    RETRY_OPEN = "retry_open"
    RETRY_UP = "retry_up"
    RETRY_BACK = "retry_back"
    DONE = "done"


@dataclass
class OneHandPickAndPlaceParams(ScriptedExpertParams):
    """Tunable geometry and timing of the scripted one-hand pick-and-place."""

    hand: str = "right"
    """Hand that grasps the object, ``"left"`` or ``"right"``."""

    raised_wrist_height: float = 0.30
    """Wrist height above the pelvis while walking up to the counter, so the hands pass over its edge [m]."""

    idle_arm_down: bool = True
    """Keep the arm that does not grasp hanging at the side; otherwise it is held raised and tucked in."""

    idle_wrist_forward: float = 0.12
    """Forward wrist position of the hand that does not grasp, held raised and tucked in [m]."""

    idle_wrist_lateral: float = 0.20
    """Sideways wrist position of the hand that does not grasp, away from the body midline [m]."""

    grasp_wrist_pitch_deg: float = -20.0
    """Pitch of the grasping wrist about the pelvis y axis [deg]; negative tilts the fingers up, which offsets
    the arm sagging below its IK target so the fingertips do not drag on the counter."""

    raise_distance: float = 1.0
    """Distance to the standoff at which the hanging arms start rising above counter height [m]."""

    retry_clearance: float = 0.06
    """Height of the wrist above the object top when the open hand pulls out after a failed grasp [m]."""

    bbox_grasp_min_height: float = 0.12
    """Objects at least this tall and at most ``bbox_grasp_max_width`` wide (bottles, cans) are wrapped from their
    bounding box even when a mesh grasp planner is set [m]."""

    bbox_grasp_max_width: float = 0.08
    """See ``bbox_grasp_min_height`` [m]."""

    stand_for_grasp: bool = False
    """Choose where to stand from the best mesh grasp, putting its wrist at the comfortable spot below, instead of
    standing at ``grasp_forward`` and ``grasp_lateral`` from the object. Off by default: the grasp finally picked
    at the counter can differ from the one stood for, which then leaves the object too far out."""

    comfortable_wrist_forward: float = 0.18
    """Forward distance from the pelvis to the wrist at a mesh grasp that the robot stands for [m]."""

    comfortable_wrist_lateral: float = 0.22
    """Sideways distance from the pelvis to the wrist at a mesh grasp, toward the grasping hand [m]."""

    min_support_gap: float = 0.22
    """Smallest distance between the pelvis and the near edge of the support the robot stands at [m]."""

    pregrasp_distance: float = 0.08
    """How far out along the approach direction the open hand waits beside the object before moving in [m]."""

    pregrasp_behind: float = 0.06
    """How far back along the fingers the hand first arrives, before sliding forward beside the object [m]."""

    grasp_sag_offset: float = 0.04
    """Height added to mesh-grasp wrist targets, since the arm sags below its IK target in simulation [m]."""

    grasp_finger_lift_deg: float = 15.0
    """Tilt of mesh-grasp wrist targets lifting the fingers, since the wrist droops in simulation [deg]."""

    max_grasp_candidates: int = 10
    """How many of the best mesh grasps to try planning a pre-grasp move to."""

    insert_clearance: float = 0.03
    """Gap kept between the object and a front-opening container's floor and opening while inserting [m]."""

    insert_open_away: float = 0.03
    """How far the open hand moves sideways off the object inside a front-opening container [m]."""

    insert_retract: float = 0.3
    """How far the open hand backs straight out of a front-opening container [m]."""

    rim_release_depth: float = 0.03
    """How far below a container's rim the object bottom may go before the hand lets go [m]."""

    same_support_tolerance: float = 0.1
    """Height difference between the object's and the destination's supports up to which they count as the same
    counter, reached by side-stepping [m]."""

    max_side_step: float = 0.8
    """Farthest destination, from the object, that is reached by side-stepping along the counter [m]."""

    max_hang_below_wrist: float = 0.2
    """How far below the commanded wrist the object center may hang while carried before it counts as lost [m]."""

    carry_speed_scale: float = 0.6
    """Fraction of the walking speed limits used while carrying the object, so it does not shake loose."""

    carry_turn_scale: float = 0.5
    """Fraction of the turning rate limit used while walking to a destination elsewhere with the object."""

    carry_pull_in: float = 0.08
    """How far the held object is pulled toward the body for walking to a destination elsewhere [m]."""

    backoff_distance: float = 0.35
    """How far the robot steps back from the counter before walking to a destination elsewhere [m]."""

    min_squat_height: float = 0.45
    """Lowest base height command when squatting to a low destination; the WBC stays stable down to ~0.43 [m]."""

    min_place_wrist_height: float = 0.05
    """Lowest wrist height above the pelvis used to lower the object before squatting further [m]."""

    squat_margin: float = 0.03
    """Extra squat depth beyond the computed one, as the arm sags less when low [m]."""

    max_grasp_attempts: int = 3
    """How many times to walk up and grasp the object, after it slipped out while still standing upright."""

    face_object: bool = False
    """Whether to turn toward the object before walking up to it; otherwise keep the start heading, which
    suits a robot spawned facing the counter."""

    grasp_forward: float = 0.30
    """Forward distance from the pelvis to the object center when grasping [m]."""

    grasp_lateral: float = 0.17
    """Sideways distance from the pelvis to the object center, toward the grasping hand [m]."""

    thumb_clearance: float = 0.10
    """Forward distance from the wrist to the object's near face at the grasp, so the thumb sits behind it [m]."""

    grasp_height_fraction: float = 0.4
    """Height of the wrist above the object bottom when grasping, as a fraction of the object height."""

    min_grasp_height: float = 0.10
    """Minimum height of the wrist above the object bottom, so the fingers clear the support [m]."""

    side_clearance: float = 0.07
    """Gap between the palm and the object side while sliding the hand forward [m]."""

    contact_depth: float = 0.01
    """How far the palm target goes inside the object side before closing the fingers [m]."""

    lift_height: float = 0.20
    """How high the object is lifted off its support before carrying it [m]."""

    place_clearance: float = 0.03
    """Height of the object bottom above the container bottom when the hand opens [m]."""

    open_away_distance: float = 0.06
    """How far the open hand moves away from the object sideways after releasing it [m]."""

    hand_time: float = 1.0
    """Time given to the fingers to close or open [s]; a one-hand grasp slips if lifted before they wrap around."""


class G1OneHandPickAndPlaceExpert(ScriptedLocomanipExpertBase):
    """Scripted expert that moves an object into a container on the same counter with one hand."""

    phases = OneHandPhase
    navigation_phases = (OneHandPhase.APPROACH, OneHandPhase.BACK_OFF, OneHandPhase.CARRY)
    hand_phases = (OneHandPhase.CLOSE, OneHandPhase.RELEASE, OneHandPhase.RETRY_OPEN)
    planned_phases = (
        OneHandPhase.APPROACH,
        OneHandPhase.REACH_SIDE,
        OneHandPhase.PLACE_OVER,
        OneHandPhase.RETRACT,
        OneHandPhase.LOWER,
        OneHandPhase.RETRY_BACK,
    )

    params: OneHandPickAndPlaceParams

    grasp_planner = None
    """Planner of grasps on the object mesh (``g1_grasp_planner.G1SceneGraspPlanner``), set by the policy; without
    one, the hand wraps the object from the side, planned from its bounding box."""

    def reset(self) -> None:
        """Start a new episode: stand at the walking height with the wrists in the ready pose."""
        super().reset()
        assert self.params.hand in SIDES, f"hand must be one of {SIDES}, got '{self.params.hand}'."
        self._heading = 0.0
        self._grasp_wrist = np.zeros(3)
        self._support_height: float | None = None
        self._raised: dict[str, np.ndarray] = {}
        self._use_mesh_grasp = False
        self._far_destination = False
        self._chosen_grasp: WristPose | None = None
        self._chosen_beside: np.ndarray | None = None
        self.chosen_grasp_kind = ""
        self.last_plan_failure = ""
        self._heading_set = False
        self._object_height: float | None = None
        self._grasp_attempts = 0

    @property
    def _side_sign(self) -> float:
        """+1 when the grasping hand is the left one, -1 for the right one (pelvis y axis points left)."""
        return 1.0 if self.params.hand == "left" else -1.0

    def _plan_phase(self, phase: OneHandPhase, state: SceneState, pelvis_xy: np.ndarray, yaw: float) -> None:
        """Set the goals of ``phase``; each group of phases is planned by its own method."""
        self._plan_grasping_phase(phase, state, pelvis_xy, yaw)
        self._plan_transport_phase(phase, state, pelvis_xy, yaw)
        self._plan_placing_phase(phase, state, pelvis_xy, yaw)
        self._plan_retry_phase(phase, state, pelvis_xy, yaw)

    def _plan_grasping_phase(self, phase: OneHandPhase, state: SceneState, pelvis_xy: np.ndarray, yaw: float) -> None:
        """Approach, grasp and lift."""
        p = self.params
        hand = p.hand
        sign = self._side_sign
        obj = world_to_heading_frame(state.object_vertices, state.pelvis_pos, yaw)

        if phase == OneHandPhase.APPROACH:
            self._grasp_attempts += 1
            if self._object_height is None:
                self._object_height = state.object_vertices[:, 2].max() - state.object_vertices[:, 2].min()
                # Upright bottles and cans keep the bounding-box wrap, which is the most reliable on them; other
                # shapes are grasped from their mesh.
                horizontal = obj[:, :2].max(axis=0) - obj[:, :2].min(axis=0)
                bottle_like = (
                    self._object_height >= p.bbox_grasp_min_height and horizontal.max() <= p.bbox_grasp_max_width
                )
                self._use_mesh_grasp = self.grasp_planner is not None and not bottle_like
            # Hold the grasping hand wide enough that, after walking up, its open fingers pass beside the object.
            half_width = 0.5 * (obj[:, 1].max() - obj[:, 1].min())
            self._raised = self._raised_wrist_pos()
            self._raised[hand][1] = sign * (p.grasp_lateral + half_width + PALM_SURFACE_OFFSET + p.side_clearance)
            # Tuck the other hand in toward the shoulder, so its fingers stay clear of the container.
            other = SIDES[1 - SIDES.index(hand)]
            self._raised[other][0] = p.idle_wrist_forward
            self._raised[other][1] = -sign * p.idle_wrist_lateral
            # Keep the start heading (square to the counter), or face the object, and stand so that the object
            # sits in front of the grasping hand. The arms rise on the way, see _working_wrist_poses.
            object_center = state.object_vertices.mean(axis=0)
            if self._heading_set:
                pass  # A retry keeps the heading of the first approach.
            elif p.face_object:
                direction = unit_xy(object_center - state.pelvis_pos)
                self._heading = math.atan2(direction[1], direction[0])
            else:
                self._heading = yaw
            self._heading_set = True
            standoff = heading_to_world_xy(
                np.array([-p.grasp_forward, -sign * p.grasp_lateral]), object_center[:2], self._heading
            )
            if self.grasp_planner is not None and p.stand_for_grasp:
                standoff = self._grasp_standoff(state, standoff)
            # Where the object will sit beside the robot once it stands at the standoff: keep the raised grasping
            # hand outside it by the side clearance.
            object_at_standoff = world_to_heading_frame(object_center, np.append(standoff, 0.0), self._heading)[0]
            object_outer_side = object_at_standoff[1] + sign * half_width
            self._raised[hand][1] = object_outer_side + sign * (PALM_SURFACE_OFFSET + p.side_clearance)
            self._nav_goal = NavigationGoal(standoff, self._heading)

        elif self._use_mesh_grasp and phase in (
            OneHandPhase.REACH_SIDE,
            OneHandPhase.REACH_FORWARD,
            OneHandPhase.CONTACT,
        ):
            self._plan_mesh_grasp_phase(phase, state, pelvis_xy, yaw)

        elif phase in (OneHandPhase.REACH_SIDE, OneHandPhase.REACH_FORWARD, OneHandPhase.CONTACT):
            # Re-plan from the live object at each step, so base drift since arriving is absorbed.
            self._plan_grasp(obj)
            goal = self._grasp_wrist.copy()
            if phase == OneHandPhase.REACH_SIDE:
                # Lower the open hand beside the object, without moving it forward yet.
                goal[0] = min(goal[0], self._wrist_cmd[hand][0])
            if phase != OneHandPhase.CONTACT:
                goal[1] += sign * (p.side_clearance + p.contact_depth)
            self._set_wrist_goal(hand, goal)

        elif phase == OneHandPhase.CLOSE:
            self._hand_closed[hand] = True
            self._support_height = state.object_vertices[:, 2].min()

        elif phase == OneHandPhase.LIFT:
            self._shift_wrist_goals(np.array([0.0, 0.0, p.lift_height]), sides=(hand,))

    def _plan_transport_phase(self, phase: OneHandPhase, state: SceneState, pelvis_xy: np.ndarray, yaw: float) -> None:
        """Carry the object to the destination: back off, walk or side-step, squat, stand."""
        p = self.params
        hand = p.hand
        obj = world_to_heading_frame(state.object_vertices, state.pelvis_pos, yaw)

        if phase == OneHandPhase.BACK_OFF:
            # Step back from the counter before turning toward a destination elsewhere; no need to stop.
            backward = heading_to_world_xy(np.array([-p.backoff_distance, 0.0]), pelvis_xy, self._heading)
            self._nav_goal = NavigationGoal(backward, self._heading, stop=False)
            self._nav_speed_scale = p.carry_speed_scale
            # Hold the object close to the body for the walk, where it swings least.
            self._shift_wrist_goals(np.array([-p.carry_pull_in, 0.0, 0.0]), sides=(hand,))

        elif phase == OneHandPhase.CARRY and state.destination_interior_vertices is not None:
            # Front-opening container: keep facing the counter and walk to where the held object waits just in
            # front of the opening, level with the cavity center.
            interior = world_to_heading_frame(state.destination_interior_vertices, state.pelvis_pos, self._heading)
            held = world_to_heading_frame(state.object_vertices, state.pelvis_pos, self._heading)
            half_depth = 0.5 * (held[:, 0].max() - held[:, 0].min())
            target = np.array([interior[:, 0].min() - p.insert_clearance - half_depth, interior.mean(axis=0)[1]])
            shift = target - held.mean(axis=0)[:2]
            self._nav_goal = NavigationGoal(heading_to_world_xy(shift, pelvis_xy, self._heading), self._heading)
            self._nav_speed_scale = p.carry_speed_scale
            self._nav_turn_scale = p.carry_turn_scale

        elif phase == OneHandPhase.CARRY and self._far_destination:
            # Turn toward the destination while walking, and stop where the held object hangs above its center.
            destination_center = state.destination_vertices.mean(axis=0)
            direction = unit_xy(destination_center - state.pelvis_pos)
            self._heading = math.atan2(direction[1], direction[0])
            held = obj.mean(axis=0)
            standoff = heading_to_world_xy(-held[:2], destination_center[:2], self._heading)
            self._nav_goal = NavigationGoal(standoff, self._heading)
            self._nav_speed_scale = p.carry_speed_scale
            self._nav_turn_scale = p.carry_turn_scale

        elif phase == OneHandPhase.SQUAT:
            # Squat as far as the arm alone cannot lower the object to its release height.
            drop = self._release_bottom(state) - state.object_vertices[:, 2].min()
            arm_drop = max(0.0, self._wrist_goal[hand][2] - p.min_place_wrist_height)
            squat = max(0.0, -drop - arm_drop + p.squat_margin)
            self._height_goal = max(p.min_squat_height, p.stand_height - squat)

        elif phase == OneHandPhase.STAND:
            self._height_goal = p.stand_height

        elif phase == OneHandPhase.CARRY:
            # Side-step along the approach heading (parallel to the counter edge) until the held object is level
            # with the container; the arm covers the forward offset when placing. Planning in the approach
            # heading rather than the current yaw keeps the goal from creeping into the counter.
            object_center = world_to_heading_frame(state.object_vertices, state.pelvis_pos, self._heading).mean(axis=0)
            container = world_to_heading_frame(self._target_vertices(state), state.pelvis_pos, self._heading)
            lateral_shift = container.mean(axis=0)[1] - object_center[1]
            target = heading_to_world_xy(np.array([0.0, lateral_shift]), pelvis_xy, self._heading)
            self._nav_goal = NavigationGoal(target, self._heading)
            self._nav_speed_scale = p.carry_speed_scale

    def _plan_placing_phase(self, phase: OneHandPhase, state: SceneState, pelvis_xy: np.ndarray, yaw: float) -> None:
        """Put the object in or on the destination, let go and pull the hand away."""
        p = self.params
        hand = p.hand
        sign = self._side_sign
        obj = world_to_heading_frame(state.object_vertices, state.pelvis_pos, yaw)

        if phase == OneHandPhase.PLACE_OVER and state.destination_interior_vertices is not None:
            # Front-opening container: bring the object level with the cavity floor, still outside its opening.
            interior = world_to_heading_frame(state.destination_interior_vertices, state.pelvis_pos, yaw)
            object_center = obj.mean(axis=0)
            dz = interior[:, 2].min() + p.insert_clearance - obj[:, 2].min()
            dy = interior.mean(axis=0)[1] - object_center[1]
            dx = min(0.0, interior[:, 0].min() - p.insert_clearance - obj[:, 0].max())
            self._shift_wrist_goals(np.array([dx, dy, dz]), sides=(hand,))

        elif phase == OneHandPhase.PLACE_DOWN and state.destination_interior_vertices is not None:
            # Slide the object in through the opening, as deep as the cavity center or the arm's reach.
            interior = world_to_heading_frame(state.destination_interior_vertices, state.pelvis_pos, yaw)
            dx = interior.mean(axis=0)[0] - obj.mean(axis=0)[0]
            self._shift_wrist_goals(np.array([max(dx, 0.0), 0.0, 0.0]), sides=(hand,))

        elif phase == OneHandPhase.OPEN_AWAY and state.destination_interior_vertices is not None:
            self._shift_wrist_goals(np.array([0.0, sign * p.insert_open_away, 0.01]), sides=(hand,))

        elif phase == OneHandPhase.RETRACT and state.destination_interior_vertices is not None:
            # Back the open hand straight out of the cavity before lowering the arm.
            self._shift_wrist_goals(np.array([-p.insert_retract, 0.0, 0.0]), sides=(hand,))
            self._force_straight = True

        elif phase == OneHandPhase.PLACE_OVER:
            object_center = obj.mean(axis=0)
            container_center = world_to_heading_frame(state.destination_vertices, state.pelvis_pos, yaw).mean(axis=0)
            delta = container_center - object_center
            self._shift_wrist_goals(np.array([delta[0], delta[1], 0.0]), sides=(hand,))

        elif phase == OneHandPhase.PLACE_DOWN:
            drop = self._release_bottom(state) - state.object_vertices[:, 2].min()
            self._shift_wrist_goals(np.array([0.0, 0.0, min(drop, 0.0)]), sides=(hand,))

        elif phase == OneHandPhase.RELEASE:
            self._hand_closed[hand] = False

        elif phase == OneHandPhase.OPEN_AWAY:
            self._shift_wrist_goals(np.array([0.0, sign * p.open_away_distance, 0.0]), sides=(hand,))

        elif phase == OneHandPhase.RETRACT:
            # Lift the open hand straight up out of the container, then it is back in the raised pose.
            raised = self._raised_wrist_pos()[hand]
            goal = self._wrist_goal[hand].copy()
            goal[2] = max(goal[2], raised[2])
            self._set_wrist_goal(hand, goal)

        elif phase == OneHandPhase.LOWER:
            self._lower_arms()

    def _plan_retry_phase(self, phase: OneHandPhase, state: SceneState, pelvis_xy: np.ndarray, yaw: float) -> None:
        """Pull the open hand off the object after a failed grasp."""
        p = self.params
        hand = p.hand
        obj = world_to_heading_frame(state.object_vertices, state.pelvis_pos, yaw)

        if phase == OneHandPhase.RETRY_OPEN:
            # Let go without moving, so the opening fingers do not push the object over.
            self._hand_closed[hand] = False

        elif phase == OneHandPhase.RETRY_UP:
            # Slide the open hand straight up until the fingers clear the top of the object.
            object_top = obj[:, 2].max()
            goal = self._wrist_goal[hand].copy()
            goal[2] = max(goal[2], object_top + p.retry_clearance)
            self._set_wrist_goal(hand, goal)

        elif phase == OneHandPhase.RETRY_BACK:
            self._set_wrist_goal(hand, self._raised[hand])

    def _plan_mesh_grasp_phase(self, phase: OneHandPhase, state: SceneState, pelvis_xy: np.ndarray, yaw: float) -> None:
        """Grasp planned on the object mesh: move to a pre-grasp pose, then straight in along the approach.

        REACH_SIDE picks the best grasp whose first pre-grasp pose the arm planner reaches without collisions,
        REACH_FORWARD slides the hand forward beside the object, and CONTACT moves in to the grasp (see
        :meth:`_pregrasp_poses`).
        """
        p = self.params
        hand = p.hand
        other = SIDES[1 - SIDES.index(hand)]
        if phase == OneHandPhase.REACH_SIDE:
            grasps = self.grasp_planner.plan(state, hand, heading=self._heading)
            R_W_P = quat_xyzw_to_matrix(state.pelvis_quat_xyzw)
            for grasp in grasps[: p.max_grasp_candidates]:
                grasp_pos, grasp_quat = self._commanded_grasp_pose(grasp, state, R_W_P)
                approach = R_W_P.T @ grasp.approach
                behind, beside = self._pregrasp_poses(grasp_pos, grasp_quat, approach)
                pregrasp = (behind, grasp_quat)
                goals = {hand: pregrasp, other: (self._wrist_goal[other].copy(), self._wrist_quat_goal[other].copy())}
                if self.planner is not None and state.arm_joint_pos is not None:
                    # The hand comes down beside the object from above; only the counter is in the way.
                    trajectory = self.planner.plan(state, goals, frozenset({"object"}), (hand,))
                    if trajectory is None:
                        self.num_failed_plans += 1
                        self.last_plan_failure = getattr(getattr(self.planner, "planner", None), "last_failure", "")
                        continue
                    self._install_planned_move(goals, trajectory)
                else:
                    self._set_wrist_goal(hand, *pregrasp)
                self._chosen_grasp = (grasp_pos, grasp_quat)
                self._chosen_beside = beside
                self.chosen_grasp_kind = grasp.kind
                return
            stats = getattr(getattr(self.grasp_planner, "planner", None), "last_stats", {})
            if grasps:
                # No collision-free plan: go to the best grasp's pre-grasp pose in a straight move instead, which
                # is how the bounding-box grasps reach beside the object.
                grasp_pos, grasp_quat = self._commanded_grasp_pose(grasps[0], state, R_W_P)
                behind, beside = self._pregrasp_poses(grasp_pos, grasp_quat, R_W_P.T @ grasps[0].approach)
                self._set_wrist_goal(hand, behind, grasp_quat)
                self._chosen_grasp = (grasp_pos, grasp_quat)
                self._chosen_beside = beside
                self.chosen_grasp_kind = grasps[0].kind
                self._force_straight = True
                return
            self._enter(self._give_up(f"found no grasp on the object mesh {stats}"), state, pelvis_xy, yaw)
        elif phase == OneHandPhase.REACH_FORWARD:
            # Slide forward along the fingers, still beside the object.
            self._set_wrist_goal(hand, self._chosen_beside)
        elif phase == OneHandPhase.CONTACT:
            self._set_wrist_goal(hand, *self._chosen_grasp)

    def _pregrasp_poses(self, grasp_pos: np.ndarray, grasp_quat: np.ndarray, approach: np.ndarray) -> tuple:
        """Return the two pre-grasp positions of a mesh grasp, in the pelvis frame.

        The hand first arrives beside the object, out along the approach and back along its fingers, then slides
        forward along the fingers, and finally moves in along the approach: the path that keeps the thumb, which
        sticks out on the palm side, behind the object.
        """
        p = self.params
        fingers = quat_xyzw_to_matrix(grasp_quat)[:, 0]
        beside = grasp_pos - p.pregrasp_distance * approach
        behind = beside - p.pregrasp_behind * fingers
        return behind, beside

    def _grasp_standoff(self, state: SceneState, default: np.ndarray) -> np.ndarray:
        """Where to stand so that the best mesh grasp puts the wrist at a comfortable spot for the arm.

        The spot is ``comfortable_wrist`` in the heading frame; the robot keeps ``min_support_gap`` between its
        pelvis and the near edge of the support, which it cannot walk into.
        """
        p = self.params
        grasps = self.grasp_planner.plan(state, p.hand, heading=self._heading)
        if not grasps:
            return default
        sign = self._side_sign
        comfortable = np.array([p.comfortable_wrist_forward, sign * p.comfortable_wrist_lateral])
        standoff = heading_to_world_xy(-comfortable, grasps[0].wrist_pos[:2], self._heading)
        if state.support_vertices is not None:
            # Distance ahead of the standoff, along the heading, to the support's near edge.
            support = world_to_heading_frame(
                np.asarray(state.support_vertices), np.append(standoff, 0.0), self._heading
            )
            overshoot = p.min_support_gap - support[:, 0].min()
            if overshoot > 0.0:
                standoff = heading_to_world_xy(np.array([-overshoot, 0.0]), standoff, self._heading)
        return standoff

    def _commanded_grasp_pose(self, grasp, state: SceneState, R_W_P: np.ndarray) -> WristPose:
        """Wrist target in the pelvis frame for a world-frame grasp, raised and tilted fingers-up to offset the arm
        sagging below its IK target in simulation."""
        p = self.params
        position = R_W_P.T @ (grasp.wrist_pos - state.pelvis_pos)
        rotation = R_W_P.T @ quat_xyzw_to_matrix(grasp.wrist_quat_xyzw)
        position[2] += p.grasp_sag_offset
        lift_axis = np.cross(rotation[:, 0], np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(lift_axis) > 1e-3:
            lift_axis /= np.linalg.norm(lift_axis)
            angle = math.radians(p.grasp_finger_lift_deg)
            K = np.array(
                [[0, -lift_axis[2], lift_axis[1]], [lift_axis[2], 0, -lift_axis[0]], [-lift_axis[1], lift_axis[0], 0]]
            )
            rotation = (np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K) @ rotation
        return position, matrix_to_quat_xyzw(rotation)

    def _planning_ignores(self) -> frozenset[str]:
        if self._phase == OneHandPhase.PLACE_OVER:
            # The object is in the hand.
            return frozenset({"object"})
        if self._phase in (OneHandPhase.RETRACT, OneHandPhase.LOWER):
            # The hand starts next to the object, inside the container.
            return frozenset({"object", "destination"})
        return frozenset()

    @staticmethod
    def _target_vertices(state: SceneState) -> np.ndarray:
        """Corners of where the object goes: a front-opening container's interior, or the destination itself."""
        if state.destination_interior_vertices is not None:
            return state.destination_interior_vertices
        return state.destination_vertices

    def _release_bottom(self, state: SceneState) -> float:
        """World height the object bottom is lowered to before letting go: just above the container bottom, but no
        deeper than a little below its rim, so deep bins are dropped into from the rim."""
        p = self.params
        destination_bottom = state.destination_vertices[:, 2].min()
        destination_top = state.destination_vertices[:, 2].max()
        return max(destination_bottom + p.place_clearance, destination_top - p.rim_release_depth)

    def _is_far_destination(self, state: SceneState) -> bool:
        """Whether the destination is off the object's support (another height) or too far to side-step to."""
        p = self.params
        height_gap = abs(state.destination_vertices[:, 2].min() - self._support_height)
        horizontal_gap = np.linalg.norm(
            state.destination_vertices.mean(axis=0)[:2] - state.object_vertices.mean(axis=0)[:2]
        )
        return height_gap > p.same_support_tolerance or horizontal_gap > p.max_side_step

    def _next_phase(self, state: SceneState) -> OneHandPhase:
        if self._phase == OneHandPhase.LIFT and not self._object_lifted(state):
            return self._phase_after_object_lost(state)
        if self._phase == OneHandPhase.LIFT:
            # Front-opening containers are walked up to from the front, like destinations elsewhere.
            inserting = state.destination_interior_vertices is not None
            self._far_destination = inserting or self._is_far_destination(state)
            return OneHandPhase.BACK_OFF if self._far_destination else OneHandPhase.CARRY
        if self._phase == OneHandPhase.CARRY and (
            not self._far_destination or state.destination_interior_vertices is not None
        ):
            return OneHandPhase.PLACE_OVER
        if self._phase == OneHandPhase.RETRACT and not self._far_destination:
            return OneHandPhase.LOWER
        if self._phase == OneHandPhase.LOWER:
            return OneHandPhase.DONE
        if self._phase == OneHandPhase.RETRY_BACK:
            # Letting go can knock the object over; only walk up again if it still stands.
            if self._object_upright(state):
                return OneHandPhase.APPROACH
            return self._give_up("the object tipped over while letting go")
        return super()._next_phase(state)

    def _phase_after_object_lost(self, state: SceneState) -> OneHandPhase:
        """Retry while attempts remain and the object still stands upright, otherwise give up."""
        if self._phase in (OneHandPhase.BACK_OFF, OneHandPhase.CARRY, OneHandPhase.SQUAT) and self._far_destination:
            return self._give_up("dropped the object on the way to the destination")
        if not self._object_upright(state):
            return self._give_up("the object tipped over")
        if self._grasp_attempts >= self.params.max_grasp_attempts:
            return self._give_up(f"the grasp failed {self._grasp_attempts} times")
        return OneHandPhase.RETRY_OPEN

    def _object_upright(self, state: SceneState) -> bool:
        """Whether the object is still about as tall as when the episode started, i.e. has not tipped over."""
        extent = state.object_vertices[:, 2].max() - state.object_vertices[:, 2].min()
        return extent > 0.8 * self._object_height

    def _object_lifted(self, state: SceneState) -> bool:
        """Whether the object hangs at least half the lift height above where it was grasped."""
        lifted_by = state.object_vertices[:, 2].min() - self._support_height
        return lifted_by >= 0.5 * self.params.lift_height

    def _working_wrist_poses(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Raised hands for walking up to the counter; the grasping hand's fingers tilt up a little, which levels
        them once the arm sags in simulation."""
        hand = self.params.hand
        other = SIDES[1 - SIDES.index(hand)]
        poses = {hand: (self._raised[hand], pitch_quat_xyzw(self.params.grasp_wrist_pitch_deg))}
        if self.params.idle_arm_down:
            # The other arm keeps hanging at the side, where it catches on nothing (counter edges, open doors).
            poses[other] = (REST_WRIST_POS[other], REST_WRIST_QUAT_XYZW)
        else:
            poses[other] = (self._raised[other], IDENTITY_QUAT_XYZW)
        return poses

    def _object_lost(self, state: SceneState) -> bool:
        """Whether the hand knocked the object over before grasping it, or the object slipped out while carried."""
        if self._phase in (
            OneHandPhase.REACH_SIDE,
            OneHandPhase.REACH_FORWARD,
            OneHandPhase.CONTACT,
            OneHandPhase.CLOSE,
        ):
            return not self._object_upright(state)
        if self._phase in (OneHandPhase.BACK_OFF, OneHandPhase.CARRY, OneHandPhase.SQUAT, OneHandPhase.PLACE_OVER):
            # Held objects hang next to the hand; one well below it has slipped out. Measured against the hand,
            # not the counter, since squatting legitimately lowers the object below the counter.
            pelvis_quat = state.pelvis_quat_xyzw
            object_center = quat_xyzw_to_matrix(pelvis_quat).T @ (state.object_vertices.mean(axis=0) - state.pelvis_pos)
            return object_center[2] < self._wrist_cmd[self.params.hand][2] - self.params.max_hang_below_wrist
        return False

    def _plan_grasp(self, obj: np.ndarray) -> None:
        """Plan the wrist position that wraps the hand around the object, from its bounds in the heading frame."""
        p = self.params
        sign = self._side_sign
        bottom, top = obj[:, 2].min(), obj[:, 2].max()
        side_y = obj[:, 1].max() if sign > 0 else obj[:, 1].min()
        self._grasp_wrist = np.array([
            obj[:, 0].min() - p.thumb_clearance,
            side_y + sign * (PALM_SURFACE_OFFSET - p.contact_depth),
            bottom + max(p.min_grasp_height, p.grasp_height_fraction * (top - bottom)),
        ])

    def _raised_wrist_pos(self) -> dict[str, np.ndarray]:
        """Wrist positions with the hands held above counter height, pulled in so they stay clear of the body."""
        raised = {}
        for side in SIDES:
            pos = HOME_WRIST_POS[side].copy()
            pos[0] -= 0.05
            pos[2] = self.params.raised_wrist_height
            raised[side] = pos
        return raised

    def _ready_wrist_pos(self) -> dict[str, np.ndarray]:
        """Wrist positions at reset: the home pose, pulled back and raised slightly."""
        offset = np.array([-0.05, 0.0, self.params.ready_wrist_raise])
        return {side: HOME_WRIST_POS[side] + offset for side in SIDES}

    def _sample_params(self) -> OneHandPickAndPlaceParams:
        """Vary distances, speeds and pauses between episodes so the demonstrations are not identical."""
        params = super()._sample_params()
        if self.rng is None:
            return params
        rng = self.rng
        params.grasp_forward += rng.uniform(-0.02, 0.02)
        params.grasp_lateral += rng.uniform(-0.02, 0.02)
        params.grasp_height_fraction += rng.uniform(-0.05, 0.05)
        params.lift_height += rng.uniform(-0.02, 0.02)
        params.raised_wrist_height += rng.uniform(-0.02, 0.02)
        return params
