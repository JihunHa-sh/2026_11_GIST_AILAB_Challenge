"""Isaac Lab ``--external_callback`` for record_demos.py / teleop_se3_agent.py: registers the ``g1_camera`` teleop
device with Arena, then runs Arena's own environment registration.

    --external_callback ailab_teleop.arena_callback.environment_registration_callback --arena_teleop_device g1_camera
"""

from __future__ import annotations

import os
from collections.abc import Callable


def register_camera_device():
    from isaaclab_arena.assets.device_library import TeleopDeviceBase
    from isaaclab_arena.assets.register import register_device, register_retargeter
    from isaaclab_arena.assets.retargeter_library import RetargetterBase

    # The device emits the embodiment's 23-D action itself (like g1_keyboard), so no isaacteleop pipeline is needed.
    for emb in ("g1_wbc_pink", "g1_wbc_agile_pink"):
        register_retargeter(type(f"G1Camera_{emb}_Retargeter", (RetargetterBase,), {
            "device": "g1_camera", "embodiment": emb, "__init__": lambda self: None,
            "get_pipeline_builder": lambda self, embodiment: None}))

    @register_device
    class G1CameraTeleopCfg(TeleopDeviceBase):
        """D555 + Fast SAM 3D Body camera teleop for the G1 WBC + PINK IK embodiments."""

        name = "g1_camera"

        def __init__(self, sim_device: str | None = None):
            super().__init__(sim_device=sim_device)

        def get_device_cfg(self, pipeline_builder: Callable | None = None, embodiment: object | None = None):
            from ailab_teleop.camera_teleop_device import CameraTeleopCfg

            import os

            cfg = CameraTeleopCfg()
            cfg.auto_calibrate_after_steps = int(os.environ.get("CAMERA_TELEOP_AUTOCAL", "0"))  # testing only
            for key, field in (("CAMERA_TELEOP_WALK_SCALE", "walk_scale"), ("CAMERA_TELEOP_TURN_SCALE", "turn_scale")):
                if os.environ.get(key):
                    setattr(cfg, field, float(os.environ[key]))
            if self.sim_device is not None:
                cfg.sim_device = self.sim_device
            return cfg


class _SuccessFunction:
    """Function-style wrapper of a class-based success term (Arena's TaskSuccessTerm).

    record_demos.py evaluates the success term itself as ``term.func(env, **term.params)`` and disables it in the
    environment, which only works for plain functions. This builds the term once for the env and resets its progress
    whenever a new episode starts.
    """

    def __init__(self, term_cfg):
        self.term_cfg = term_cfg.copy()
        self.term = None
        self.last_step = None

    def __call__(self, env, **params):
        import torch

        if self.term is None:
            self.term = self.term_cfg.func(self.term_cfg, env)
        step = int(env.episode_length_buf[0])
        if self.last_step is not None and step < self.last_step:  # the env was reset
            self.term.reset()
        self.last_step = step
        done = torch.as_tensor(self.term(env, **params))
        fake = int(os.environ.get("CAMERA_TELEOP_FAKE_SUCCESS_AFTER", "0"))  # testing the export path only
        if fake and step >= fake:
            done = torch.ones_like(done, dtype=torch.bool)
        return done


def adapt_for_record_demos(task: str):
    """Recorder with camera images, and a success term record_demos.py can evaluate."""
    import gymnasium as gym

    from isaaclab_arena_g1.g1_env.mdp.recorders.g1_locomanip_recorder_cfg import G1LocomanipRecorderManagerCfg

    spec = gym.spec(task)
    if spec.kwargs.get("demo_recorder_cfg_entry_point") is None:
        spec.kwargs["demo_recorder_cfg_entry_point"] = G1LocomanipRecorderManagerCfg()
    env_cfg = spec.kwargs["env_cfg_entry_point"]
    success = getattr(env_cfg.terminations, "success", None)
    if success is not None and isinstance(success.func, type):
        env_cfg.terminations.success = success.replace(func=_SuccessFunction(success))


def environment_registration_callback():
    import argparse

    register_camera_device()
    from isaaclab_arena.environments.isaaclab_interop import environment_registration_callback as arena_callback

    remaining = arena_callback()
    task = argparse.ArgumentParser(add_help=False)
    task.add_argument("--task", type=str)
    adapt_for_record_demos(task.parse_known_args()[0].task)
    return remaining
