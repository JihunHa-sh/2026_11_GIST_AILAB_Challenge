# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

r"""Import Objaverse objects as Arena rigid objects.

Downloads objects of an Objaverse LVIS category (or given uids), turns them Z-up, scales each to a target size,
and writes a physics-ready USD per object (rigid body, mass, convex-decomposition collision, textured preview
material, high-friction contact material). The objects are recorded in the local object registry, so they can
be spawned by name without Isaac Sim needing to know about them in advance:

    python -m isaaclab_arena_g1.tools.objaverse_import --category mug --count 3 --size 0.11
    python isaaclab_arena/evaluation/policy_runner.py ... g1_kitchen_pick_and_place --object objaverse_mug_0

List categories with ``--list_categories <substring>``. Isaac Sim is not needed to import.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import random
import re
import trimesh
from pathlib import Path

from isaaclab_arena_g1.tools.local_objects import load_local_object_entries, local_object_dir

MAX_TRIANGLES = 60000
"""Larger models are decimated to about this many triangles; their textures are dropped for the base color."""

MAX_DECOMPOSITION_TRIANGLES = 30000
"""Above this many triangles the collision shape falls back from a convex decomposition to a convex hull."""


def _sanitize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _load_meshes(glb_path: str) -> list[trimesh.Trimesh]:
    """Load the meshes of a glTF file with their scene transforms applied, rotated from glTF Y-up to Z-up."""
    loaded = trimesh.load(glb_path, force="scene")
    meshes = [geometry for geometry in loaded.dump(concatenate=False) if isinstance(geometry, trimesh.Trimesh)]
    y_up_to_z_up = trimesh.transformations.rotation_matrix(np.pi / 2.0, [1.0, 0.0, 0.0])
    for mesh in meshes:
        mesh.apply_transform(y_up_to_z_up)
    return [mesh for mesh in meshes if len(mesh.faces) > 0]


def _normalize(meshes: list[trimesh.Trimesh], size: float) -> float:
    """Center the meshes on their joint bounding box and scale its longest side to ``size``; return the scale."""
    bounds = np.array([mesh.bounds for mesh in meshes])
    low, high = bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)
    scale = size / float((high - low).max())
    center = 0.5 * (low + high)
    for mesh in meshes:
        mesh.apply_translation(-center)
        mesh.apply_scale(scale)
    return scale


def _decimate(meshes: list[trimesh.Trimesh], max_triangles: int) -> list[trimesh.Trimesh]:
    """Reduce the meshes to about ``max_triangles`` in total, keeping each mesh's share; small models pass as is."""
    total = sum(len(mesh.faces) for mesh in meshes)
    if total <= max_triangles:
        return meshes
    reduced = []
    for mesh in meshes:
        target = max(4, int(len(mesh.faces) * max_triangles / total))
        try:
            color = mesh.visual.main_color
        except (AttributeError, ValueError):
            color = None
        simplified = mesh.simplify_quadric_decimation(face_count=target)
        if color is not None:
            simplified.visual = trimesh.visual.ColorVisuals(
                simplified, face_colors=np.tile(color, (len(simplified.faces), 1))
            )
        reduced.append(simplified)
    print(f"[objaverse_import] Decimated {total} triangles to {sum(len(m.faces) for m in reduced)}.")
    return reduced


def _write_material(stage, path: str, mesh: trimesh.Trimesh, texture_dir: Path, index: int):
    """Create a UsdPreviewSurface material from the mesh's texture or color; return it, or None."""
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    visual = mesh.visual
    image = None
    if isinstance(visual, trimesh.visual.TextureVisuals) and visual.uv is not None and visual.material is not None:
        image = getattr(visual.material, "baseColorTexture", None) or getattr(visual.material, "image", None)
    if image is not None:
        texture_file = texture_dir / f"texture_{index}.png"
        image.convert("RGB").save(texture_file)
        reader = UsdShade.Shader.Define(stage, f"{path}/uv_reader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        texture = UsdShade.Shader.Define(stage, f"{path}/texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./textures/{texture_file.name}")
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
    else:
        try:
            color = np.asarray(visual.main_color[:3], dtype=np.float64) / 255.0
        except (AttributeError, ValueError):
            color = np.array([0.7, 0.7, 0.7])
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(tuple(float(c) for c in color))
    return material


def write_object_usd(meshes: list[trimesh.Trimesh], name: str, out_dir: Path, mass: float, friction: float) -> Path:
    """Write a rigid-body USD of the meshes, whose root prim is the object; return its path."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "textures").mkdir(exist_ok=True)
    usd_path = out_dir / f"{name}.usd"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(mass)

    contact_material = UsdShade.Material.Define(stage, f"/{name}/physics_material")
    physics_material = UsdPhysics.MaterialAPI.Apply(contact_material.GetPrim())
    physics_material.CreateStaticFrictionAttr(friction)
    physics_material.CreateDynamicFrictionAttr(friction)
    physics_material.CreateRestitutionAttr(0.0)

    num_triangles = sum(len(mesh.faces) for mesh in meshes)
    approximation = "convexDecomposition" if num_triangles <= MAX_DECOMPOSITION_TRIANGLES else "convexHull"
    for index, mesh in enumerate(meshes):
        usd_mesh = UsdGeom.Mesh.Define(stage, f"/{name}/mesh_{index}")
        usd_mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, v)) for v in mesh.vertices]))
        usd_mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(mesh.faces)))
        usd_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(mesh.faces.reshape(-1).tolist()))
        usd_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        if isinstance(mesh.visual, trimesh.visual.TextureVisuals) and mesh.visual.uv is not None:
            uv = UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
            )
            uv.Set(Vt.Vec2fArray([Gf.Vec2f(*map(float, t)) for t in mesh.visual.uv]))
        material = _write_material(stage, f"/{name}/Looks/material_{index}", mesh, out_dir / "textures", index)
        UsdShade.MaterialBindingAPI.Apply(usd_mesh.GetPrim()).Bind(material)
        UsdPhysics.CollisionAPI.Apply(usd_mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(usd_mesh.GetPrim()).CreateApproximationAttr(approximation)
        UsdShade.MaterialBindingAPI(usd_mesh.GetPrim()).Bind(
            contact_material, UsdShade.Tokens.weakerThanDescendants, "physics"
        )
    stage.GetRootLayer().Save()
    return usd_path


def import_objects(
    uids: list[str], category: str, size: float, mass: float, friction: float, start_index: int = 0
) -> list[str]:
    """Download, convert and register Objaverse objects; return their Arena object names."""
    import objaverse

    out_root = local_object_dir()
    entries = load_local_object_entries()
    paths = objaverse.load_objects(uids=uids)
    names = []
    for offset, uid in enumerate(uids):
        name = f"objaverse_{_sanitize(category)}_{start_index + offset}"
        meshes = _load_meshes(paths[uid])
        if not meshes:
            print(f"[objaverse_import] Skipping {uid}: no triangle meshes.")
            continue
        _normalize(meshes, size)
        meshes = _decimate(meshes, MAX_TRIANGLES)
        usd_path = write_object_usd(meshes, name, out_root / name, mass=mass, friction=friction)
        entries[name] = {
            "usd_path": str(usd_path),
            "source": f"objaverse:{uid}",
            "category": category,
            "size": size,
            "mass": mass,
        }
        names.append(name)
        print(f"[objaverse_import] {name}: {uid} -> {usd_path}")
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "objects.json", "w") as f:
        json.dump(entries, f, indent=1)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", type=str, help="Objaverse LVIS category, e.g. mug, banana, apple.")
    parser.add_argument("--uid", type=str, nargs="*", default=None, help="Objaverse uids to import instead.")
    parser.add_argument("--count", type=int, default=1, help="How many objects of the category to import.")
    parser.add_argument("--size", type=float, default=0.10, help="Longest side of each object after scaling [m].")
    parser.add_argument("--mass", type=float, default=0.15, help="Mass of each object [kg].")
    parser.add_argument("--friction", type=float, default=1.0, help="Static and dynamic friction of the object.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for picking objects of the category.")
    parser.add_argument("--list_categories", type=str, default=None, help="Print categories containing this text.")
    args = parser.parse_args()

    import objaverse

    if args.list_categories is not None:
        lvis = objaverse.load_lvis_annotations()
        for category in sorted(c for c in lvis if args.list_categories.lower() in c.lower()):
            print(f"{category}: {len(lvis[category])} objects")
        return

    if args.uid:
        uids, category = args.uid, args.category or "object"
    else:
        assert args.category is not None, "Give --category or --uid."
        lvis = objaverse.load_lvis_annotations()
        assert args.category in lvis, f"Unknown LVIS category '{args.category}'; try --list_categories."
        uids = random.Random(args.seed).sample(lvis[args.category], min(args.count, len(lvis[args.category])))
        category = args.category
    existing = [name for name in load_local_object_entries() if name.startswith(f"objaverse_{_sanitize(category)}_")]
    names = import_objects(uids, category, args.size, args.mass, args.friction, start_index=len(existing))
    print(
        "[objaverse_import] Spawn with: g1_kitchen_pick_and_place --object <name>  (and the policy's "
        "--pick_up_object_name <name>), for name in:",
        " ".join(names),
    )


if __name__ == "__main__":
    main()
