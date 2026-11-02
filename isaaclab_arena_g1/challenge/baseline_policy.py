# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Baseline for the g1_fast_transport challenge: the scripted one-hand expert, aimed at the goal pose.

It reads privileged state (object and goal poses) from the simulator, like the challenge observation does. Run it
with::

    python isaaclab_arena_g1/challenge/challenge_runner.py \\
        --policy_type isaaclab_arena_g1.challenge.baseline_policy.G1FastTransportBaselinePolicy \\
        --team_name baseline --num_seeds 10 g1_fast_transport
"""

from __future__ import annotations

import gymnasium as gym
import itertools
import numpy as np
import torch
from dataclasses import dataclass
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_g1.policy.g1_one_hand_pick_and_place_expert import (
    G1OneHandPickAndPlaceExpert,
    OneHandPhase,
    OneHandPickAndPlaceParams,
)
from isaaclab_arena_g1.policy.g1_scripted_expert_base import (
    NavigationGoal,
    SceneState,
    heading_to_world_xy,
    matrix_to_quat_xyzw,
    quat_xyzw_to_matrix,
    world_to_heading_frame,
    wrap_to_pi,
    yaw_from_quat_xyzw,
)
from isaaclab_arena_g1.policy.g1_scripted_expert_policy import G1ScriptedExpertPolicyMixin
from isaaclab_arena_g1.policy.g1_scripted_one_hand_pick_and_place_policy import G1ScriptedOneHandPickAndPlacePolicyCfg

GOAL_DESTINATION_NAME = "fast_transport_goal"
"""Destination name the baseline uses for the goal footprint; it is not a scene entity."""

GOAL_FOOTPRINT_HEIGHT = 0.005
"""Height of the box standing for the goal footprint on the table top [m]."""


class G1FastTransportExpert(G1OneHandPickAndPlaceExpert):
    """One-hand expert that, while carrying, turns so the held object reaches the goal yaw."""

    goal_yaw: float | None = None
    """Goal yaw of the object [rad], set by the policy every step; None keeps the plain expert's heading."""

    object_yaw: float | None = None
    """Current yaw of the object [rad], set by the policy every step."""

    object_quat_w: np.ndarray | None = None
    """Current orientation (x, y, z, w) of the object in the world, set by the policy every step."""

    goal_quat_w: np.ndarray | None = None
    """Goal orientation (x, y, z, w) of the object in the world, set by the policy every step."""

    max_carry_yaw_correction: float = np.radians(15.0)
    """Largest deviation from facing the table square-on that the walk takes to turn the object toward the goal
    yaw [rad]."""

    _approach_heading: float = 0.0

    def _plan_grasping_phase(self, phase: OneHandPhase, state: SceneState, pelvis_xy: np.ndarray, yaw: float) -> None:
        super()._plan_grasping_phase(phase, state, pelvis_xy, yaw)
        if phase == OneHandPhase.APPROACH:
            self._approach_heading = self._heading

    place_forward: float = 0.30
    """Minimum forward distance from the pelvis to the goal when the robot stops at the table [m]; the arm reaches
    out over the table to place. The object is carried pulled in, and stopping at that closer distance would put the
    torso against the table edge."""

    def _plan_transport_phase(self, phase: OneHandPhase, state: SceneState, pelvis_xy: np.ndarray, yaw: float) -> None:
        aligning = (
            phase == OneHandPhase.CARRY
            and self._far_destination
            and state.destination_interior_vertices is None
            and self.goal_yaw is not None
            and self.object_yaw is not None
        )
        if not aligning:
            super()._plan_transport_phase(phase, state, pelvis_xy, yaw)
            return
        p = self.params
        # The grasp holds the object's yaw fixed relative to the pelvis, so turning the pelvis by the object's yaw
        # error brings the object to the goal yaw. Re-planned every step, so it closes the loop while turning.
        # Face the table square-on (turned around from the approach), letting the turn take at most
        # max_carry_yaw_correction of the yaw error; the wrist removes the rest when placing.
        facing_table = wrap_to_pi(self._approach_heading + np.pi)
        wanted = wrap_to_pi(yaw + wrap_to_pi(self.goal_yaw - self.object_yaw))
        correction = np.clip(
            wrap_to_pi(wanted - facing_table), -self.max_carry_yaw_correction, self.max_carry_yaw_correction
        )
        self._heading = wrap_to_pi(facing_table + correction)
        held = world_to_heading_frame(state.object_vertices, state.pelvis_pos, yaw).mean(axis=0)
        held[0] = max(held[0], self.place_forward)
        destination_center = state.destination_vertices.mean(axis=0)
        # Stand where the held object, at its current offset from the pelvis, hangs above the goal.
        standoff = heading_to_world_xy(-held[:2], destination_center[:2], self._heading)
        # Far from the table, first walk forward to a waypoint square in front of it, facing the way to go; the
        # last stretch is then a short straight step toward the table.
        approach_dir = np.array([np.cos(facing_table), np.sin(facing_table)])
        waypoint = standoff - self.transit_distance * approach_dir
        if not self._transit_done and np.linalg.norm(waypoint - pelvis_xy) > self.transit_min_distance:
            travel = waypoint - pelvis_xy
            self._in_transit = True
            self._nav_goal = NavigationGoal(waypoint, float(np.arctan2(travel[1], travel[0])), stop=False)
            self._nav_speed_scale = p.carry_speed_scale
            self._nav_turn_scale = p.carry_turn_scale
            return
        self._in_transit = False
        self._nav_goal = NavigationGoal(standoff, self._heading)
        self._nav_speed_scale = p.carry_speed_scale
        self._nav_turn_scale = p.carry_turn_scale

    transit_distance: float = 0.6
    """Distance in front of the standing spot at the table of the waypoint walked to first [m]."""

    transit_min_distance: float = 0.4
    """Skip the waypoint when the robot is already this close to it [m]."""

    _in_transit: bool = False
    _transit_done: bool = False

    recenter_tolerance: float = 0.012
    """Horizontal object-to-goal distance above which the object is moved again before letting go [m]."""

    recenter_yaw_tolerance: float = np.radians(4.0)
    """Object yaw error above which the object is turned again before letting go [rad]."""

    max_recenters: int = 3
    """Most extra re-centering moves before letting go."""

    _recenters: int = 0

    def reset(self) -> None:
        super().reset()
        self._recenters = 0
        self._in_transit = False
        self._transit_done = False
        self._place_xy = None

    def _next_phase(self, state: SceneState) -> OneHandPhase:
        if self._phase == OneHandPhase.CARRY and self._in_transit:
            # At the waypoint: plan the last stretch to the table.
            self._transit_done = True
            return OneHandPhase.CARRY
        if (
            self._phase == OneHandPhase.PLACE_DOWN
            and state.destination_interior_vertices is None
            and self._recenters < self.max_recenters
        ):
            offset = state.object_vertices.mean(axis=0)[:2] - state.destination_vertices.mean(axis=0)[:2]
            yaw_error = abs(wrap_to_pi(self.goal_yaw - self.object_yaw)) if self.goal_yaw is not None else 0.0
            if np.linalg.norm(offset) > self.recenter_tolerance or yaw_error > self.recenter_yaw_tolerance:
                self._recenters += 1
                return OneHandPhase.PLACE_DOWN
        if self._phase == OneHandPhase.RETRACT and self._far_destination:
            # Done once the hand is clear: lowering the arm past the table could sweep the object over.
            return OneHandPhase.DONE
        return super()._next_phase(state)

    def _correct_object_orientation(self, hand: str, yaw: float) -> None:
        """Turn the wrist by the object's remaining orientation error (its tilt in the hand and the yaw the walk
        did not remove), so the object comes down upright at the goal yaw."""
        R_W_goal = quat_xyzw_to_matrix(self.goal_quat_w)
        R_W_obj = quat_xyzw_to_matrix(self.object_quat_w)
        R_err_W = R_W_goal @ R_W_obj.T
        c, s = np.cos(yaw), np.sin(yaw)
        R_W_P = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        R_err_P = R_W_P.T @ R_err_W @ R_W_P
        R_P_wrist = quat_xyzw_to_matrix(self._wrist_quat_goal[hand])
        self._wrist_quat_goal[hand] = matrix_to_quat_xyzw(R_err_P @ R_P_wrist)

    retract_lift: float = 0.12
    """How far above the object's top the open hand is lifted after letting go [m]."""

    _place_xy: np.ndarray | None = None

    def _targets_reached(self) -> bool:
        if self._phase in (OneHandPhase.PLACE_OVER, OneHandPhase.PLACE_DOWN):
            # The heading hold keeps correcting in the background; the arm moves need not wait for it.
            nav_goal, self._nav_goal = self._nav_goal, None
            try:
                return super()._targets_reached()
            finally:
                self._nav_goal = nav_goal
        return super()._targets_reached()

    def _plan_placing_phase(self, phase: OneHandPhase, state: SceneState, pelvis_xy: np.ndarray, yaw: float) -> None:
        if state.destination_interior_vertices is not None or self.goal_quat_w is None or self.object_quat_w is None:
            super()._plan_placing_phase(phase, state, pelvis_xy, yaw)
            return
        hand = self.params.hand
        if phase == OneHandPhase.PLACE_OVER:
            self._place_xy = pelvis_xy.copy()
        if phase in (OneHandPhase.PLACE_OVER, OneHandPhase.PLACE_DOWN) and self._place_xy is not None:
            # Hold the spot and heading the robot arrived at: the reaching arm otherwise slowly turns and pushes
            # the body back. See _targets_reached: the arm does not wait for this hold.
            self._nav_goal = NavigationGoal(self._place_xy.copy(), self._heading)
        if phase == OneHandPhase.PLACE_OVER:
            super()._plan_placing_phase(phase, state, pelvis_xy, yaw)
            self._correct_object_orientation(hand, yaw)
        elif phase == OneHandPhase.PLACE_DOWN:
            # Entered again by _next_phase while the object is off center: re-center and re-level each time.
            super()._plan_placing_phase(phase, state, pelvis_xy, yaw)
            # Re-center over the goal: turning the wrist and the arm sag move the object off it.
            obj = world_to_heading_frame(state.object_vertices, state.pelvis_pos, yaw).mean(axis=0)
            goal = world_to_heading_frame(state.destination_vertices, state.pelvis_pos, yaw).mean(axis=0)
            self._shift_wrist_goals(np.array([goal[0] - obj[0], goal[1] - obj[1], 0.0]), sides=(hand,))
            self._correct_object_orientation(hand, yaw)
        elif phase == OneHandPhase.RETRACT:
            # Lift the open hand straight up, clear of the object's top; the arm then stays up.
            goal = self._wrist_goal[hand].copy()
            object_top = world_to_heading_frame(state.object_vertices, state.pelvis_pos, yaw)[:, 2].max()
            goal[2] = max(goal[2] + self.retract_lift, object_top + self.retract_lift)
            self._set_wrist_goal(hand, goal)
            self._force_straight = True
        else:
            super()._plan_placing_phase(phase, state, pelvis_xy, yaw)


@dataclass
class G1FastTransportBaselinePolicyCfg(G1ScriptedOneHandPickAndPlacePolicyCfg):
    """Configure the fast-transport baseline."""

    pick_up_object_name: str = "bbq_sauce_bottle_hope_robolab"
    """Scene name of the object to carry; match the environment's ``--object``."""

    destination_name: str = GOAL_DESTINATION_NAME
    """Leave as is: the goal footprint on the table stands in for the destination."""

    done_timeout_s: float = 2.0
    """Give up this long after the motion finished without the episode succeeding, instead of idling [s]."""

    place_clearance: float = 0.035
    """Height above the table top at which the object is let go [m]."""

    contact_depth: float = -0.005
    """How far the palm is pushed into the object's side before closing [m]; slightly negative stops it just short,
    so it does not tip tall bottles over, and the closing fingers pull the object in."""

    side_clearance: float = 0.09
    """Sideways gap between the open hand and the object while reaching, so the fingers do not bump it [m]."""

    max_forward_vel: float = 0.4
    """Maximum forward walking speed [m/s].

    These speed limits are conservative. The built-in walking policy follows commands up to about 1.4 m/s; with
    max_forward_vel 1.0, max_yaw_rate 1.0, carry_turn_scale 0.8 the baseline finished seeds 0 and 1 in about 24 s
    instead of 35 s (turning while carrying was the slowest part), but tilts the bottle more in the hand."""

    carry_speed_scale: float = 0.6
    """Fraction of the walking speed limits used while carrying the object."""

    carry_turn_scale: float = 0.5
    """Fraction of the turning rate limit used while carrying the object."""

    max_lateral_vel: float = 0.3
    """Maximum sideways walking speed [m/s]."""

    max_yaw_rate: float = 0.5
    """Maximum turning rate [rad/s]."""

    navigation_timeout: float = 30.0
    """Longest a walking phase may take before the expert gives up [s]; the carry across the room is long."""

    randomize_motion: bool = False
    """Vary distances and speeds between episodes; off for a reproducible baseline."""

    use_motion_planner: bool = False
    """Plan the free-space arm moves with cuRobo; off, every arm move is a straight minimum-jerk move."""

    use_grasp_planner: bool = False
    """Plan grasps on the object mesh; off, upright bottles are wrapped by their bounding box."""


@register_policy
class G1FastTransportBaselinePolicy(G1ScriptedExpertPolicyMixin, PolicyBase[G1FastTransportBaselinePolicyCfg]):
    """Scripted baseline: walk to the counter, grasp, back off, turn to the goal yaw, walk to the table, place."""

    name = "g1_fast_transport_baseline"
    expert_cls = G1FastTransportExpert
    params_cls = OneHandPickAndPlaceParams

    def __init__(self, config: G1FastTransportBaselinePolicyCfg):
        """Initialize the policy.

        Args:
            config: Typed policy configuration.
        """
        super().__init__(config)
        self._object_local_box: tuple[np.ndarray, np.ndarray] | None = None
        self._done_steps = 0

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        """Compute the next 23-D action from the privileged scene state and the goal."""
        if self.expert is not None:
            unwrapped = env.unwrapped
            goal_quat = self._goal_pose(unwrapped)[1]
            object_quat = unwrapped.scene[self.config.pick_up_object_name].data.root_pose_w.torch[0, 3:7]
            object_quat = object_quat.cpu().numpy().astype(np.float64)
            self.expert.goal_yaw = yaw_from_quat_xyzw(goal_quat)
            self.expert.object_yaw = yaw_from_quat_xyzw(object_quat)
            self.expert.goal_quat_w = goal_quat
            self.expert.object_quat_w = object_quat
            self._give_up_when_idle(unwrapped)
        return super().get_action(env, observation)

    def _give_up_when_idle(self, unwrapped) -> None:
        """End the episode as a failure once the motion has been done for ``done_timeout_s`` without success."""
        if self.expert.phase != OneHandPhase.DONE:
            self._done_steps = 0
            return
        self._done_steps += 1
        if self._done_steps * unwrapped.step_dt < self.config.done_timeout_s or self._ended_episode:
            return
        print(f"[{self.name}] Giving up: the object did not settle at the goal.", flush=True)
        flags = getattr(unwrapped, "policy_gave_up_buf", None)
        if flags is None:
            flags = torch.zeros(unwrapped.num_envs, dtype=torch.bool, device=unwrapped.device)
            unwrapped.policy_gave_up_buf = flags
        flags[:] = True
        self._gave_up_flags = flags
        self._ended_episode = True

    def _goal_pose(self, unwrapped) -> tuple[np.ndarray, np.ndarray]:
        """Goal position (world) and orientation (x, y, z, w) of the object."""
        from isaaclab_arena_g1.challenge.fast_transport_task import FAST_TRANSPORT_STATE_ATTR

        state = getattr(unwrapped, FAST_TRANSPORT_STATE_ATTR)
        position = (state.goal_pos_e[0] + unwrapped.scene.env_origins[0]).cpu().numpy().astype(np.float64)
        quat = state.goal_quat[0].cpu().numpy().astype(np.float64)
        return position, quat

    def _vertices_w(self, unwrapped, name: str) -> np.ndarray:
        """Corners of a scene object; for the goal, a thin box under the object's footprint at the goal pose."""
        if name != GOAL_DESTINATION_NAME:
            return super()._vertices_w(unwrapped, name)
        if self._object_local_box is None:
            from isaaclab_arena.assets.registries import AssetRegistry

            box = AssetRegistry().get_asset_by_name(self.config.pick_up_object_name)().get_bounding_box()
            self._object_local_box = (box.min_point[0].cpu().numpy(), box.max_point[0].cpu().numpy())
        low, high = self._object_local_box
        position, quat = self._goal_pose(unwrapped)
        corners = np.array(list(itertools.product(*zip(low, high))), dtype=np.float64)
        corners = corners @ quat_xyzw_to_matrix(quat).T + position
        # Flatten to the table top, where the object's bottom rests at the goal.
        table_top = corners[:, 2].min()
        corners[:, 2] = np.where(corners[:, 2] > table_top + 1e-6, table_top + GOAL_FOOTPRINT_HEIGHT, table_top)
        return corners

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Restart the scripted motion for a new episode."""
        super().reset(env_ids)
        self._done_steps = 0
        if self.expert is not None:
            self.expert.goal_yaw = None
            self.expert.object_yaw = None
            self.expert.goal_quat_w = None
            self.expert.object_quat_w = None
