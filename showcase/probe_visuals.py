"""Feasibility probe: showcase camera orbit + runtime material/light randomization in the G1 kitchen."""

import math
import os
import sys
import traceback

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import showcase_lib as sl  # noqa: E402

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser  # noqa: E402
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext  # noqa: E402
from isaaclab_arena_environments.cli import (  # noqa: E402
    get_arena_builder_from_cli,
    get_isaaclab_arena_environments_cli_parser,
)

OUT = os.path.join(sl.HERE, "probe")


def main():
    parser = get_isaaclab_arena_cli_parser()
    parser.set_defaults(device="cpu")
    args, _ = parser.parse_known_args(sys.argv[1:])
    with SimulationAppContext(args):
        try:
            run(parser)
        except Exception:
            traceback.print_exc()


def run(parser):
    os.makedirs(OUT, exist_ok=True)
    parser = get_isaaclab_arena_environments_cli_parser(parser)
    args, hydra = parser.parse_known_args(sys.argv[1:])
    builder = get_arena_builder_from_cli(args, hydra_overrides=hydra)
    env_cfg, env_kwargs = builder.compose_manager_cfg()
    env_cfg.recorders = {}
    sl.add_showcase_camera(env_cfg)
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    env_cfg.scene.small_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/SmallCam", update_period=0.0, height=480, width=640, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=14.0, clipping_range=(0.05, 40.0)),
    )
    env = builder.make_registered(env_cfg, env_kwargs)
    sl.stub_keyboard()
    from isaaclab.devices.teleop_device_factory import create_teleop_device

    device = create_teleop_device("g1_keyboard", env.unwrapped.cfg.teleop_devices.devices, {})
    device._scene_list_printed = True
    world = env.unwrapped.arena_world
    rng = np.random.default_rng(1)
    pool = sl.texture_pool()
    print("texture pool:", len(pool), flush=True)
    rigid = list(env.unwrapped.scene.rigid_objects.keys())
    print("rigid objects:", rigid, flush=True)

    with torch.inference_mode():
        env.reset()
    pelvis = world.get_body_pose_w("robot", "pelvis")[0].cpu().numpy().astype(float)
    center = (pelvis[0], pelvis[1] + 0.5, 0.75)
    print("pelvis", pelvis[:3], flush=True)
    views = {
        "A_behind_high": ((3.95, -2.6, 1.7), (3.95, -0.6, 0.9)),
        "B_side": ((2.6, -1.6, 1.4), (3.95, -0.9, 0.9)),
        "C_close_front": ((3.95, -2.0, 1.2), (3.95, -1.4, 0.8)),
        "D_top": ((3.95, -1.5, 2.8), (3.95, -1.0, 0.5)),
    }
    for name, (eye, tgt) in views.items():
        for _ in range(6):
            sl.look_at(env, eye, tgt)
            c2 = env.unwrapped.scene["small_cam"]
            c2.set_world_poses_from_view(torch.tensor([eye]), torch.tensor([tgt]))
            with torch.inference_mode():
                env.step(device.advance().repeat(1, 1))
        cam = env.unwrapped.scene["showcase_cam"]
        pos = cam.data.pos_w
        pos = pos.torch if hasattr(pos, "torch") else pos
        q = cam.data.quat_w_world
        q = q.torch if hasattr(q, "torch") else q
        arr = sl.grab(env)
        print(name, "pos", pos[0].tolist(), "quat_world", [round(v, 3) for v in q[0].tolist()], "mean", arr.mean(), flush=True)
        Image.fromarray(arr).save(f"{OUT}/dbg_{name}.png")
        arr2 = sl.grab(env, "small_cam")
        print("   small_cam mean", arr2.mean(), arr2.shape, flush=True)
        Image.fromarray(arr2).save(f"{OUT}/dbg_small_{name}.png")
    env.close()


if __name__ == "__main__":
    main()
