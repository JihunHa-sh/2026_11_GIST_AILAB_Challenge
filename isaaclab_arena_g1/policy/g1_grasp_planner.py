# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Grasp planning on object meshes for the G1 Dex3 hand: antipodal pinches, with the hand treated as a two-finger
gripper, and wrap grasps with the palm on the object.

The Dex3 closes its thumb against its index and middle fingers, so it is modeled as a parallel gripper: a closing
axis between the thumb and the finger pads, a grasp center between them, and an approach direction along the palm
normal. Candidates are sampled around the object's vertical axis: closing axes in every horizontal direction,
approached either from the side (palm vertical) or from above (palm down). Each candidate casts rays along the
closing axis to find the two contact points; it is kept when the surface normals there lie inside the friction
cone, the object fits between the fingers and the open hand clears the support, and it is scored by contact
quality, distance from the object center and how naturally the hand is oriented for the robot.

The module only depends on numpy and trimesh, so it can be tested without Isaac Sim.
"""

from __future__ import annotations

import math
import numpy as np
import trimesh
from dataclasses import dataclass

# Two-finger model of the right Dex3 hand in the wrist (right_wrist_yaw_link) frame, measured from the URDF with
# the joint targets the simulation reaches when the hand closes. The left hand mirrors it in y.
RIGHT_GRASP_CENTER = np.array([0.14, 0.055, 0.0])
"""Point between the thumb and the finger pads that the object center goes to [m]."""

RIGHT_CLOSING_AXIS = np.array([0.87, -0.5, 0.0]) / np.linalg.norm([0.87, -0.5, 0.0])
"""Unit direction from the thumb pad to the finger pads."""

# Points bounding the open hand in the wrist frame (palm, fingers, thumb), used for the support clearance check.
RIGHT_OPEN_HAND_POINTS = np.array(
    [[x, y, z] for x in (0.041, 0.128) for y in (-0.024, 0.017) for z in (-0.044, 0.044)]
    + [[x, y, z] for x in (0.112, 0.217) for y in (-0.017, 0.01) for z in (-0.042, 0.042)]
    + [[x, y, z] for x in (0.056, 0.073) for y in (0.016, 0.114) for z in (-0.013, 0.013)]
)


def hand_model(side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the grasp center, closing axis and open-hand points of one hand in its wrist frame."""
    mirror = np.array([1.0, -1.0, 1.0]) if side == "left" else np.ones(3)
    return RIGHT_GRASP_CENTER * mirror, RIGHT_CLOSING_AXIS * mirror, RIGHT_OPEN_HAND_POINTS * mirror


def palm_normal(side: str) -> np.ndarray:
    """Unit direction, in the wrist frame, in which the hand moves onto the object: the palm normal, orthogonal to
    the closing axis."""
    _, axis, _ = hand_model(side)
    toward_palm = np.array([0.0, 1.0 if side == "right" else -1.0, 0.0])
    normal = toward_palm - np.dot(toward_palm, axis) * axis
    return normal / np.linalg.norm(normal)


def matrix_to_quat_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to an xyzw quaternion."""
    w, x, y, z = trimesh.transformations.quaternion_from_matrix(np.block([[rotation, np.zeros((3, 1))], [0, 0, 0, 1]]))
    quat = np.array([x, y, z, w])
    return quat if quat[3] >= 0.0 else -quat


@dataclass
class Grasp:
    """A grasp of the object, in the world frame."""

    wrist_pos: np.ndarray
    """Wrist position at the grasp [m]."""

    wrist_quat_xyzw: np.ndarray
    """Wrist orientation at the grasp."""

    approach: np.ndarray
    """Unit direction the hand moves in onto the object; the pre-grasp pose sits back along it."""

    width: float
    """Object width between the contacts [m]."""

    score: float
    """Higher is better."""

    kind: str
    """"side", "tilted" (pinches approaching from the side or from above at an angle), "top" (pinch with the palm
    down) or "power" (palm on the object, fingers wrapped around it)."""


@dataclass
class GraspPlannerParams:
    """Tunable limits and weights of the grasp search."""

    friction_coefficient: float = 0.8
    """Friction coefficient of the finger contacts; sets the friction cone of an antipodal contact."""

    max_width: float = 0.10
    """Widest object section that fits between the open thumb and fingers [m]."""

    min_width: float = 0.012
    """Narrowest object section the closed hand still pinches [m]."""

    support_clearance: float = 0.015
    """Height the open hand keeps above the support surface in pinches [m]."""

    power_support_clearance: float = 0.0
    """Height the open hand keeps above the support surface in wrap grasps, which may graze it [m]."""

    yaw_step_deg: float = 15.0
    """Angular step of the closing-axis directions [deg]."""

    height_step: float = 0.015
    """Vertical step of the side-grasp heights [m]."""

    top_grasp_depths: tuple[float, ...] = (0.01, 0.02, 0.03)
    """How far below the object top the closing axis of a top grasp sits, tried in turn [m]."""

    approach_tilts_deg: tuple[float, ...] = (0.0, 30.0, 60.0)
    """Downward tilts of the palm normal tried for grasps from the side; 0 is a horizontal approach [deg]."""

    min_power_depth: float = 0.03
    """Shallowest object section along the approach that a wrap grasp holds [m]."""

    max_power_depth: float = 0.085
    """Deepest object section along the approach the fingers still curl around [m]."""

    max_power_length: float = 0.12
    """Longest object section along the fingers that fits in a wrap grasp [m]."""

    power_thumb_clearance: float = 0.10
    """Distance along the fingers from the wrist to the near edge of a wrapped section, which clears the thumb
    that sticks out toward the palm side [m]."""

    palm_surface_offset: float = 0.017
    """Distance from the wrist axis to the palm surface [m]."""

    power_squeeze: float = 0.01
    """How far the palm target goes into the object when wrapping it [m]."""

    min_power_inward: float = 0.97
    """Smallest cosine between a wrap grasp's horizontal approach and the direction toward the body midline, and
    between its fingers and the robot heading; diagonal wraps push the object over while moving in."""

    power_bonus: float = 0.6
    """Score bonus of wrap grasps on round sections, which hold better than fingertip pinches in simulation."""

    preferred_height_fraction: float = 0.45
    """Side grasps near this fraction of the object height score best."""


class G1GraspPlanner:
    """Finds antipodal Dex3 grasps on one object mesh."""

    def __init__(self, mesh: trimesh.Trimesh, params: GraspPlannerParams | None = None):
        """Prepare the object mesh for ray casting.

        Args:
            mesh: Object mesh in the object's root frame.
            params: Limits and weights of the search.
        """
        # Plan on the convex hull: simulated colliders are convex approximations at least as large as the visual
        # mesh, so contacts planned on the visual surface would push into the collider.
        self.mesh = mesh.convex_hull
        self.params = params or GraspPlannerParams()

    def plan(
        self,
        object_pos: np.ndarray,
        object_quat_xyzw: np.ndarray,
        side: str,
        support_height: float,
        robot_xy: np.ndarray,
        robot_yaw: float | None = None,
        max_grasps: int = 8,
    ) -> list[Grasp]:
        """Return the best grasps for one hand, best first.

        Args:
            object_pos: Object root position in the world [m].
            object_quat_xyzw: Object root orientation in the world.
            side: Hand that grasps, "left" or "right".
            support_height: World height of the surface the object stands on [m].
            robot_xy: Robot pelvis position in the world (xy) [m], to prefer grasps the robot faces naturally.
            robot_yaw: Robot heading [rad]; the preferred directions follow it. None uses the direction from the
                robot to the object instead, which suits a robot still far away.
            max_grasps: How many grasps to return at most.
        """
        p = self.params
        x, y, z, w = object_quat_xyzw
        T_W_O = trimesh.transformations.quaternion_matrix([w, x, y, z])
        T_W_O[:3, 3] = object_pos
        mesh = self.mesh.copy()
        mesh.apply_transform(T_W_O)
        low, high = mesh.bounds
        center_xy = 0.5 * (low[:2] + high[:2])
        if robot_yaw is None:
            to_object = np.append(center_xy - robot_xy, 0.0)
            to_object /= max(np.linalg.norm(to_object), 1e-6)
        else:
            to_object = np.array([math.cos(robot_yaw), math.sin(robot_yaw), 0.0])
        # Horizontal direction toward the body midline from the grasping hand's side: a right hand comes in from
        # the robot's right and moves left onto the object, the way it reaches without crossing the body.
        robot_left = np.array([-to_object[1], to_object[0], 0.0])
        inward = robot_left if side == "right" else -robot_left

        grasps = []
        for yaw in np.arange(0.0, 360.0, p.yaw_step_deg):
            axis = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.0])
            for horizontal in (np.array([-axis[1], axis[0], 0.0]), np.array([axis[1], -axis[0], 0.0])):
                # Palm normal tilted from horizontal (side grasp) down toward vertical; low objects need the
                # steeper approaches, which keep the fingers off the support.
                for tilt_deg in p.approach_tilts_deg:
                    tilt = math.radians(tilt_deg)
                    approach = math.cos(tilt) * horizontal + math.sin(tilt) * np.array([0.0, 0.0, -1.0])
                    kind = "side" if tilt_deg == 0.0 else "tilted"
                    for height in np.arange(low[2] + p.height_step, high[2] - 0.005, p.height_step):
                        center = np.array([*center_xy, height])
                        fraction = (height - low[2]) / max(high[2] - low[2], 1e-6)
                        for grasp in (
                            self._evaluate(mesh, center, axis, approach, side, kind),
                            self._evaluate_power(mesh, center, axis, approach, side),
                        ):
                            if grasp is not None:
                                grasp.score -= 1.0 * abs(fraction - p.preferred_height_fraction)
                                grasps.append(grasp)
            # Top grasps: palm facing down, closing near the top of the object.
            for depth in p.top_grasp_depths:
                top = np.array([*center_xy, high[2] - depth])
                grasp = self._evaluate(mesh, top, axis, np.array([0.0, 0.0, -1.0]), side, "top")
                if grasp is not None:
                    grasps.append(grasp)

        valid = []
        self.last_stats = {"candidates": len(grasps), "power_orientation": 0, "support": 0}
        for grasp in grasps:
            if grasp.kind == "power":
                # Wrap only the way that works on the G1 in simulation: palm toward the body midline, fingers
                # pointing away from the robot. Other wraps push the object over while moving in.
                R = trimesh.transformations.quaternion_matrix([grasp.wrist_quat_xyzw[3], *grasp.wrist_quat_xyzw[:3]])
                horizontal_approach = np.array([*grasp.approach[:2], 0.0])
                horizontal_approach /= max(np.linalg.norm(horizontal_approach), 1e-6)
                if (
                    np.dot(horizontal_approach, inward) < p.min_power_inward
                    or np.dot(R[:3, 0], to_object) < p.min_power_inward
                ):
                    self.last_stats["power_orientation"] += 1
                    continue
            if not self._clears_support(grasp, side, support_height):
                self.last_stats["support"] += 1
            else:
                # Prefer the fingers pointing away from the robot, and side grasps coming in from the hand's side
                # (or from the front), never from beyond the object toward the robot.
                R = trimesh.transformations.quaternion_matrix([grasp.wrist_quat_xyzw[3], *grasp.wrist_quat_xyzw[:3]])[
                    :3, :3
                ]
                grasp.score += 0.5 * float(np.dot(R[:, 0], to_object))
                if grasp.kind != "top":
                    grasp.score += 0.6 * float(np.dot(grasp.approach, inward))
                    grasp.score += 0.3 * float(np.dot(grasp.approach, to_object))
                valid.append(grasp)
        valid.sort(key=lambda grasp: grasp.score, reverse=True)
        return valid[:max_grasps]

    def _evaluate(
        self,
        mesh: trimesh.Trimesh,
        center: np.ndarray,
        axis: np.ndarray,
        approach: np.ndarray,
        side: str,
        kind: str,
    ) -> Grasp | None:
        """Find the two contacts along ``axis`` through ``center`` and build the grasp, or None if it is invalid."""
        p = self.params
        reach = 0.5
        origins = np.array([center - reach * axis, center + reach * axis])
        directions = np.array([axis, -axis])
        locations, ray_ids, triangle_ids = mesh.ray.intersects_location(origins, directions, multiple_hits=False)
        if len(set(ray_ids.tolist())) < 2:
            return None
        hit = {
            int(ray): (location, mesh.face_normals[tri]) for location, ray, tri in zip(locations, ray_ids, triangle_ids)
        }
        (p_thumb, n_thumb), (p_fingers, n_fingers) = hit[0], hit[1]
        width = float(np.dot(p_fingers - p_thumb, axis))
        if not p.min_width <= width <= p.max_width:
            return None
        # Antipodal: the outward normals must oppose the squeezing directions within the friction cone.
        cos_cone = math.cos(math.atan(p.friction_coefficient))
        quality = min(float(np.dot(n_thumb, -axis)), float(np.dot(n_fingers, axis)))
        if quality < cos_cone:
            return None
        grasp_center = 0.5 * (p_thumb + p_fingers)

        # Wrist orientation: map the hand's closing axis and palm normal onto the chosen world directions.
        hand_center, hand_axis, _ = hand_model(side)
        hand_normal = palm_normal(side)
        hand_frame = np.column_stack([hand_axis, hand_normal, np.cross(hand_axis, hand_normal)])
        world_frame = np.column_stack([axis, approach, np.cross(axis, approach)])
        rotation = world_frame @ hand_frame.T
        wrist_pos = grasp_center - rotation @ hand_center
        return Grasp(
            wrist_pos=wrist_pos,
            wrist_quat_xyzw=matrix_to_quat_xyzw(rotation),
            approach=approach.copy(),
            width=width,
            score=quality,
            kind=kind,
        )

    def _evaluate_power(
        self,
        mesh: trimesh.Trimesh,
        center: np.ndarray,
        axis: np.ndarray,
        approach: np.ndarray,
        side: str,
    ) -> Grasp | None:
        """Wrap grasp: palm pressed on the object's near side, fingers along ``axis`` curling around its far side.

        The object section through ``center`` must be shallow enough along the approach for the fingers to curl
        around it, and the hand sits so that the section's near edge along the fingers stays ahead of the thumb.
        Returns None when the object does not fit.
        """
        p = self.params
        reach = 0.5
        origins = np.array(
            [center - reach * approach, center + reach * approach, center - reach * axis, center + reach * axis]
        )
        directions = np.array([approach, -approach, axis, -axis])
        locations, ray_ids, _ = mesh.ray.intersects_location(origins, directions, multiple_hits=False)
        if len(set(ray_ids.tolist())) < 4:
            return None
        hit = {int(ray): location for location, ray in zip(locations, ray_ids)}
        depth = float(np.dot(hit[1] - hit[0], approach))
        length = float(np.dot(hit[3] - hit[2], axis))
        if not p.min_power_depth <= depth <= p.max_power_depth or length > p.max_power_length:
            return None
        section_center = 0.5 * (hit[0] + hit[1])
        section_center += (0.5 * (hit[2] + hit[3]) - center) @ axis * axis
        # Wrist frame: fingers along the axis, palm normal along the approach (the right palm faces +y, the left -y).
        palm_sign = 1.0 if side == "right" else -1.0
        x_axis = axis
        y_axis = palm_sign * approach
        rotation = np.column_stack([x_axis, y_axis, np.cross(x_axis, y_axis)])
        # Object center relative to the wrist: past the thumb along the fingers, against the palm along its normal.
        local_center = np.array([
            p.power_thumb_clearance + 0.5 * length,
            palm_sign * (p.palm_surface_offset + 0.5 * depth - p.power_squeeze),
            0.0,
        ])
        wrist_pos = section_center - rotation @ local_center
        # Wrapping suits sections about as deep as they are long, such as bottles and fruit.
        roundness = min(depth, length) / max(depth, length)
        return Grasp(
            wrist_pos=wrist_pos,
            wrist_quat_xyzw=matrix_to_quat_xyzw(rotation),
            approach=approach.copy(),
            width=depth,
            score=0.8 + p.power_bonus * roundness,
            kind="power",
        )

    def _clears_support(self, grasp: Grasp, side: str, support_height: float) -> bool:
        """Whether the open hand at the grasp stays above the support surface."""
        _, _, hand_points = hand_model(side)
        R = trimesh.transformations.quaternion_matrix([grasp.wrist_quat_xyzw[3], *grasp.wrist_quat_xyzw[:3]])[:3, :3]
        lowest = (hand_points @ R.T + grasp.wrist_pos)[:, 2].min()
        clearance = self.params.power_support_clearance if grasp.kind == "power" else self.params.support_clearance
        return lowest >= support_height + clearance


class G1SceneGraspPlanner:
    """Plans grasps for a scripted expert from its scene state; implements the expert's ``grasp_planner``."""

    def __init__(self, mesh: trimesh.Trimesh, params: GraspPlannerParams | None = None):
        """Prepare the planner for one object.

        Args:
            mesh: Object mesh in the object's root frame, scaled like the spawned object.
            params: Limits and weights of the search.
        """
        self.planner = G1GraspPlanner(mesh, params)

    def plan(self, state, side: str, heading: float | None = None) -> list[Grasp]:
        """Return the best grasps of the object in ``state`` for one hand, best first, in the world frame.

        Args:
            state: Scene state of the expert.
            side: Hand that grasps, "left" or "right".
            heading: Robot heading the grasp directions are judged against [rad], or None to use the direction
                toward the object.
        """
        assert state.object_pos is not None, "Mesh grasps need the object pose in the scene state."
        if state.support_vertices is not None:
            support_height = float(np.asarray(state.support_vertices)[:, 2].max())
        else:
            support_height = float(np.asarray(state.object_vertices)[:, 2].min())
        return self.planner.plan(
            np.asarray(state.object_pos, dtype=np.float64),
            np.asarray(state.object_quat_xyzw, dtype=np.float64),
            side,
            support_height,
            np.asarray(state.pelvis_pos[:2], dtype=np.float64),
            robot_yaw=heading,
        )
