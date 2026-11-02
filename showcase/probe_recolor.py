"""Probe in-place recoloring: three randomizations, head-cam + showcase frames after each."""
import os, sys, traceback
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import showcase_lib as sl
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser
OUT = os.path.join(sl.HERE, "probe")

def run(parser):
    parser = get_isaaclab_arena_environments_cli_parser(parser)
    args, hydra = parser.parse_known_args(sys.argv[1:])
    b = get_arena_builder_from_cli(args, hydra_overrides=hydra)
    cfg, kw = b.compose_manager_cfg(); cfg.recorders = {}
    sl.add_showcase_camera(cfg)
    env = b.make_registered(cfg, kw)
    sl.stub_keyboard()
    from isaaclab.devices.teleop_device_factory import create_teleop_device
    dev = create_teleop_device("g1_keyboard", env.unwrapped.cfg.teleop_devices.devices, {}); dev._scene_list_printed = True
    rng = np.random.default_rng(5); pool = sl.texture_pool()
    rigid = list(env.unwrapped.scene.rigid_objects.keys())
    with torch.inference_mode(): env.reset()
    tiles = []
    for k in range(4):
        if k:
            for n in rigid:
                print(n, sl.recolor_prim(f"/World/envs/env_0/{n}", rng, pool), flush=True)
        for _ in range(8):
            sl.look_at(env, (3.95, -1.05, 1.45), (3.97, -0.45, 0.95))
            with torch.inference_mode(): env.step(dev.advance().repeat(1, 1))
        tiles.append(sl.grab(env))
    W = Image.new("RGB", (1280, 720))
    for i, t in enumerate(tiles): W.paste(Image.fromarray(t).resize((640, 360)), ((i % 2) * 640, (i // 2) * 360))
    W.save(f"{OUT}/recolor.png"); print("saved", flush=True)
    env.close()

if __name__ == "__main__":
    p = get_isaaclab_arena_cli_parser(); p.set_defaults(device="cpu")
    a, _ = p.parse_known_args(sys.argv[1:])
    with SimulationAppContext(a):
        try: run(p)
        except Exception: traceback.print_exc()
