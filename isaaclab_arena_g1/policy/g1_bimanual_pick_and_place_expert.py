# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""State-machine expert that carries a box with both G1 hands, driven by privileged object poses.

The expert starts standing with its arms at its sides, raises them while walking up to the box, squeezes it
between the palms, lifts it, backs off, turns toward the destination while walking up to it, squats, lowers
the box onto it and drops its arms back to its sides.

The module only depends on numpy, so the state machine can be exercised without Isaac Sim.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from enum import Enum

from isaaclab_arena_g1.policy.g1_scripted_expert_base import (
    HOME_WRIST_POS,
    PALM_SURFACE_OFFSET,
    SIDES,
    NavigationGoal,
    SceneState,
    ScriptedExpertParams,
    ScriptedLocomanipExpertBase,
    distance_to_rectangle_edge,
    unit_xy,
    world_to_heading_frame,
)


class Phase(Enum):
    """Phases of the scripted pick-and-place, executed in definition order."""

    SETTLE = "settle"
    APPROACH = "approach"
    REACH_SIDE = "reach_side"
    REACH_FORWARD = "reach_forward"
    SQUEEZE = "squeeze"
    CLOSE = "close"
    LIFT = "lift"
    BACK_OFF = "back_off"
    CARRY = "carry"
    SQUAT = "squat"
    PLACE_FORWARD = "place_forward"
    PLACE_DOWN = "place_down"
    RELEASE = "release"
    SPREAD = "spread"
    RETRACT = "retract"
    LOWER = "lower"
    DONE = "done"


@dataclass
class BimanualPickAndPlaceParams(ScriptedExpertParams):
    """Tunable geometry and timing of the scripted two-handed pick-and-place."""

    grasp_distance: float = 0.40
    """Horizontal distance from the pelvis to the box center when grasping [m]."""

    wrist_behind_near_face: float = 0.10
    """How far the wrists stay behind the box face nearest to the robot, so the thumbs clear it [m]."""

    grasp_clearance: float = 0.06
    """Gap between each palm and the box side before squeezing [m]."""

    squeeze_depth: float = 0.02
    """How far each palm target goes inside the box side, which sets the squeeze force [m]."""

    close_hands: bool = True
    """Whether to close the fingers on the box after squeezing it."""

    lift_height: float = 0.08
    """How high the box is lifted off its support before walking [m]."""

    lift_pull_back: float = 0.02
    """How far the box is pulled toward the body while lifting it [m]."""

    backoff_distance: float = 0.3
    """How far the robot walks backward after lifting, to clear the shelf before turning [m]."""

    place_wrist_x: float = 0.27
    """Forward wrist position, in the pelvis frame, at which the box is lowered onto the destination [m]."""

    place_margin: float = 0.05
    """Distance kept between the box and the edges of the destination when placing [m]."""

    place_lateral_offset: float = 0.0
    """Sideways shift of the release point, positive to the robot's left [m]."""

    place_clearance: float = 0.02
    """Height of the box bottom above the destination top when the hands open [m]."""

    release_spread: float = 0.06
    """How far each wrist moves outward after the hands open [m]."""

    place_height: float = 0.62
    """Base height command while placing the box, squatting to bring the shoulders down [m]."""


class G1BimanualPickAndPlaceExpert(ScriptedLocomanipExpertBase):
    """Scripted expert that picks a box up with both hands and puts it on a destination."""

    phases = Phase
    navigation_phases = (Phase.APPROACH, Phase.BACK_OFF, Phase.CARRY)
    hand_phases = (Phase.CLOSE, Phase.RELEASE)

    params: BimanualPickAndPlaceParams

    def reset(self) -> None:
        """Start a new episode: stand at the walking height with the wrists in the ready pose."""
        super().reset()
        self._grasp: dict[str, float] = {}
        self._release_point = np.zeros(2)

    def _plan_phase(self, phase: Phase, state: SceneState, pelvis_xy: np.ndarray, yaw: float) -> None:
        p = self.params
        object_center = state.object_vertices.mean(axis=0)
        destination_center = state.destination_vertices.mean(axis=0)

        if phase == Phase.APPROACH:
            # Face the box and stop at the grasp distance in front of it.
            direction = unit_xy(object_center - state.pelvis_pos)
            heading = math.atan2(direction[1], direction[0])
            self._grasp["heading"] = heading
            self._nav_goal = NavigationGoal(object_center[:2] - p.grasp_distance * direction, heading)

        elif phase in (Phase.REACH_SIDE, Phase.REACH_FORWARD, Phase.SQUEEZE):
            # Re-plan the grasp from the live box at each step, so base drift since arriving is absorbed.
            self._plan_grasp(state, yaw)
            for side, sign in (("left", 1.0), ("right", -1.0)):
                if phase == Phase.REACH_SIDE:
                    # Raise the open hands beside the box, still behind its near face.
                    x = self._wrist_cmd[side][0]
                else:
                    x = self._grasp["wrist_x"]
                if phase == Phase.SQUEEZE:
                    y = self._grasp[f"{side}_side_y"] - sign * p.squeeze_depth
                else:
                    y = self._grasp[f"{side}_side_y"] + sign * p.grasp_clearance
                self._set_wrist_goal(side, np.array([x, y, self._grasp["wrist_z"]]))

        elif phase == Phase.CLOSE:
            self._hand_closed = {side: p.close_hands for side in SIDES}

        elif phase == Phase.LIFT:
            self._shift_wrist_goals(np.array([-p.lift_pull_back, 0.0, p.lift_height]))

        elif phase == Phase.BACK_OFF:
            # Back straight out of the shelf before turning; no need to stop before carrying on.
            heading = self._grasp["heading"]
            backward = -p.backoff_distance * np.array([math.cos(heading), math.sin(heading)])
            self._nav_goal = NavigationGoal(pelvis_xy + backward, heading, stop=False)

        elif phase == Phase.CARRY:
            # Turn toward the destination while walking up to it.
            self._nav_goal = self._plan_place_standoff(state, yaw, destination_center)

        elif phase == Phase.SQUAT:
            self._height_goal = p.place_height

        elif phase == Phase.PLACE_FORWARD:
            # Move the box horizontally over the release point, measured on the box itself.
            box_center = world_to_heading_frame(object_center, state.pelvis_pos, yaw)[0]
            release = world_to_heading_frame(
                np.array([*self._release_point, object_center[2]]),
                state.pelvis_pos,
                yaw,
            )[0]
            self._shift_wrist_goals(np.array([release[0] - box_center[0], release[1] - box_center[1], 0.0]))

        elif phase == Phase.PLACE_DOWN:
            box_bottom = state.object_vertices[:, 2].min()
            destination_top = state.destination_vertices[:, 2].max()
            drop = destination_top + p.place_clearance - box_bottom
            self._shift_wrist_goals(np.array([0.0, 0.0, min(drop, 0.0)]))

        elif phase == Phase.RELEASE:
            self._hand_closed = {side: False for side in SIDES}

        elif phase == Phase.SPREAD:
            for side, sign in (("left", 1.0), ("right", -1.0)):
                self._set_wrist_goal(
                    side,
                    self._wrist_goal[side] + np.array([0.0, sign * p.release_spread, 0.0]),
                )

        elif phase == Phase.RETRACT:
            # Pull the open hands back and up without sweeping them over the placed box, and stand up.
            ready = self._ready_wrist_pos()
            for side in SIDES:
                goal = self._wrist_goal[side].copy()
                goal[0] = ready[side][0]
                goal[2] += p.release_spread
                self._set_wrist_goal(side, goal)
            self._height_goal = p.stand_height

        elif phase == Phase.LOWER:
            self._lower_arms()

    def _plan_grasp(self, state: SceneState, yaw: float) -> None:
        """Plan the wrist positions of the squeeze grasp from the box bounds in the pelvis heading frame."""
        box = world_to_heading_frame(state.object_vertices, state.pelvis_pos, yaw)
        self._grasp.update(
            wrist_x=box[:, 0].min() - self.params.wrist_behind_near_face,
            wrist_z=0.5 * (box[:, 2].min() + box[:, 2].max()),
            left_side_y=box[:, 1].max() + PALM_SURFACE_OFFSET,
            right_side_y=box[:, 1].min() - PALM_SURFACE_OFFSET,
        )

    def _plan_place_standoff(
        self,
        state: SceneState,
        yaw: float,
        destination_center: np.ndarray,
    ) -> NavigationGoal:
        """Pick the release point on the destination and the pelvis pose that holds the box above it."""
        p = self.params
        direction = unit_xy(destination_center - state.pelvis_pos)
        left = np.array([-direction[1], direction[0]])
        heading = math.atan2(direction[1], direction[0])

        # Release point: as close to the robot as the destination footprint allows with the box fully on it,
        # so the reach stays short. The footprint is the destination's horizontal bounding rectangle.
        footprint = state.destination_vertices[:, :2]
        footprint_half = 0.5 * (footprint.max(axis=0) - footprint.min(axis=0))
        box = world_to_heading_frame(state.object_vertices, state.pelvis_pos, yaw)
        box_half_depth = 0.5 * (box[:, 0].max() - box[:, 0].min())
        box_half_width = 0.5 * (box[:, 1].max() - box[:, 1].min())
        back_room = max(
            0.0,
            distance_to_rectangle_edge(footprint_half, direction) - box_half_depth - p.place_margin,
        )
        side_room = max(
            0.0,
            distance_to_rectangle_edge(footprint_half, left) - box_half_width - p.place_margin,
        )
        lateral_offset = float(np.clip(p.place_lateral_offset, -side_room, side_room))
        self._release_point = destination_center[:2] - back_room * direction + lateral_offset * left

        # Stand so that, with the wrists at place_wrist_x, the carried box sits above the release point.
        box_center = box.mean(axis=0)
        wrist_x = 0.5 * (self._wrist_cmd["left"][0] + self._wrist_cmd["right"][0])
        hold_distance = p.place_wrist_x + (box_center[0] - wrist_x)
        standoff = self._release_point - hold_distance * direction - box_center[1] * left
        return NavigationGoal(standoff, heading)

    def _ready_wrist_pos(self) -> dict[str, np.ndarray]:
        """Wrist positions while walking with empty hands: the home pose, pulled back to clear the box and raised."""
        offset = np.array([-0.05, 0.0, self.params.ready_wrist_raise])
        return {side: HOME_WRIST_POS[side] + offset for side in SIDES}

    def _sample_params(self) -> BimanualPickAndPlaceParams:
        """Vary distances, speeds and pauses between episodes so the demonstrations are not identical."""
        params = super()._sample_params()
        if self.rng is None:
            return params
        rng = self.rng
        params.grasp_distance += rng.uniform(-0.02, 0.02)
        params.lift_height += rng.uniform(-0.02, 0.02)
        params.backoff_distance += rng.uniform(-0.05, 0.05)
        params.place_height += rng.uniform(-0.02, 0.02)
        params.place_lateral_offset += rng.uniform(-0.04, 0.04)
        return params
