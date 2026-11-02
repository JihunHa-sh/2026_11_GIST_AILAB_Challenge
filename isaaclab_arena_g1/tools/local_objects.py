# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Registry of objects imported on this machine (for example from Objaverse), usable as Arena objects by name.

``objaverse_import.py`` writes each imported object's USD and an entry in ``objects.json`` under the local object
directory. :func:`register_local_objects` turns every entry into a registered Arena object asset, so environments
take it like any library object, e.g. ``g1_kitchen_pick_and_place --object objaverse_mug_0``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

LOCAL_OBJECT_DIR_ENV = "ISAACLAB_ARENA_LOCAL_OBJECTS"
"""Environment variable overriding the directory of locally imported objects."""


def local_object_dir() -> Path:
    """Return the directory holding the locally imported objects and their ``objects.json``."""
    default = Path.home() / ".cache" / "isaaclab_arena" / "local_objects"
    return Path(os.environ.get(LOCAL_OBJECT_DIR_ENV, default))


def load_local_object_entries() -> dict[str, dict]:
    """Return the entries of ``objects.json`` by object name, or an empty dict if nothing was imported."""
    registry = local_object_dir() / "objects.json"
    if not registry.exists():
        return {}
    with open(registry) as f:
        return json.load(f)


def register_extra_fixtures() -> None:
    """Register kitchen fixtures that Arena's object library does not ship yet."""
    from isaaclab_arena.affordances.openable import Openable
    from isaaclab_arena.assets.lightwheel_lazy import LightwheelLazyPath
    from isaaclab_arena.assets.object_library import LibraryObject
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.assets.register import register_asset
    from isaaclab_arena.assets.registries import AssetRegistry

    if AssetRegistry().is_registered("toaster_oven", ensure_loaded=False):
        return

    class ToasterOven(LibraryObject, Openable):
        """A countertop toaster oven whose door hinges at the bottom and folds down to open."""

        name = "toaster_oven"
        tags = ["object", "openable"]
        usd_path = LightwheelLazyPath(registry_type="fixtures", file_name="ToasterOven008", file_type="USD")
        object_type = ObjectType.ARTICULATION
        openable_joint_name = "door_joint"
        openable_threshold = 0.5

        def __init__(self, instance_name: str | None = None, prim_path: str | None = None, initial_pose=None):
            super().__init__(
                instance_name=instance_name,
                prim_path=prim_path,
                initial_pose=initial_pose,
                openable_joint_name=self.openable_joint_name,
                openable_threshold=self.openable_threshold,
            )

    register_asset(ToasterOven)


def register_local_objects() -> list[str]:
    """Register every locally imported object as an Arena object asset; already registered names are skipped.

    Returns:
        The names of all locally imported objects.
    """
    from isaaclab_arena.assets.object_library import LibraryObject
    from isaaclab_arena.assets.register import register_asset
    from isaaclab_arena.assets.registries import AssetRegistry

    register_extra_fixtures()
    names = []
    for name, entry in load_local_object_entries().items():
        names.append(name)
        if AssetRegistry().is_registered(name, ensure_loaded=False):
            continue
        asset_type = type(
            f"LocalObject_{name}",
            (LibraryObject,),
            {
                "name": name,
                "tags": ["object", "graspable", "local"],
                "usd_path": entry["usd_path"],
                "scale": (1.0, 1.0, 1.0),
                "__doc__": f"Locally imported object '{name}' ({entry.get('source', 'unknown source')}).",
            },
        )
        register_asset(asset_type)
    return names
