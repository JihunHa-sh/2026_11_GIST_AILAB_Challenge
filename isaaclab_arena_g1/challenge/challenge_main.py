# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Main program of the g1_fast_transport challenge: loads a team's high and low level from its team file and scores
them, one episode per seed.

    python isaaclab_arena_g1/challenge/challenge_main.py --team isaaclab_arena_g1/challenge/policies/baseline/team.yaml \\
        --num_seeds 10

A team file::

    team_name: my_team
    high_level:
      class: isaaclab_arena_g1.challenge.policies.my_team.high_level.MyHighLevel   # a HighLevelPolicy
      options: {}                                                                 # keyword arguments
    low_level:
      class: default        # the built-in controller, or module.Class of a LowLevelPolicy
      options: {}

With ``low_level.class: default`` the environment's built-in controller (walking policy + arm IK) runs the command;
otherwise the robot takes the team low level's 43 joint position targets (``g1_joint_position`` embodiment).
Every other option is the same as challenge_runner.py's (--num_seeds, --seeds, --video, --viz kit, ...). Environment
options may follow ``g1_fast_transport`` at the end, for practice only.
"""

from __future__ import annotations

import datetime
import importlib
import os
import sys
import yaml

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser
from isaaclab_arena_g1.challenge.challenge_runner import (
    CHALLENGE_ENVIRONMENT,
    add_challenge_arguments,
    add_video_camera,
    episode_seeds,
    print_table,
    run_episode,
    summarize,
    write_results,
)

BUILTIN_EMBODIMENT = "g1_wbc_pink"
JOINT_EMBODIMENT = "g1_joint_position"


def load_team(path: str) -> dict:
    """Read a team file and check its keys."""
    with open(path) as f:
        team = yaml.safe_load(f)
    assert isinstance(team, dict) and "high_level" in team, f"{path}: a team file needs at least high_level.class."
    team.setdefault("team_name", os.path.splitext(os.path.basename(os.path.dirname(os.path.abspath(path))))[0])
    team.setdefault("low_level", {"class": "default"})
    for layer in ("high_level", "low_level"):
        assert "class" in team[layer], f"{path}: {layer}.class is missing."
        team[layer].setdefault("options", {})
    return team


def import_class(dotted: str):
    """Import ``package.module.Class``."""
    module_path, class_name = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


class HierarchicalPolicy:
    """Runs a high level and a low level as one policy for the challenge runner."""

    def __init__(self, high_level, low_level):
        from isaaclab_arena_g1.challenge.interfaces import HighLevelPolicy, LowLevelPolicy

        assert isinstance(high_level, HighLevelPolicy), f"{type(high_level)} is not a HighLevelPolicy."
        assert low_level is None or isinstance(low_level, LowLevelPolicy), f"{type(low_level)} is not a LowLevelPolicy."
        self.high_level = high_level
        self.low_level = low_level
        self._set_up = False

    @property
    def phase(self):
        """The high level's phase, when it has one."""
        return getattr(self.high_level, "phase", None)

    def set_task_description(self, description: str | None) -> None:
        self.task_description = description

    def reset(self, env_ids=None) -> None:
        self.high_level.reset()
        if self.low_level is not None:
            self.low_level.reset()

    def setup(self, env) -> None:
        self.high_level.setup(env)
        if self.low_level is not None:
            self.low_level.setup(env)
        self._set_up = True

    def get_action(self, env, observation):
        from isaaclab_arena_g1.challenge.interfaces import NUM_JOINTS, RobotState

        command = self.high_level.compute_command(env, observation)
        if self.low_level is None:
            return command.to_tensor()
        targets = self.low_level.compute_joint_targets(command, RobotState.from_env(env))
        assert targets.shape[-1] == NUM_JOINTS, f"A low level returns {NUM_JOINTS} joint targets, got {targets.shape}."
        return targets.reshape(1, NUM_JOINTS)


def environment_argv(argv: list[str], embodiment: str) -> list[str]:
    """Append the challenge environment (if not given) and the embodiment the team's low level needs."""
    argv = list(argv)
    if CHALLENGE_ENVIRONMENT not in argv:
        argv.append(CHALLENGE_ENVIRONMENT)
    assert "--embodiment" not in argv, "The embodiment follows from the team file's low_level; do not pass it."
    return argv + ["--embodiment", embodiment]


def main():
    """Score a team's high and low level on the fast-transport challenge."""
    args_parser = get_isaaclab_arena_cli_parser()
    add_challenge_arguments(args_parser)
    args_parser.add_argument("--team", type=str, required=True, help="Path of the team file (team.yaml).")
    args_cli, _ = args_parser.parse_known_args()
    if args_cli.video:
        args_cli.enable_cameras = True
    team = load_team(args_cli.team)
    builtin_low_level = team["low_level"]["class"] == "default"
    embodiment = BUILTIN_EMBODIMENT if builtin_low_level else JOINT_EMBODIMENT

    with SimulationAppContext(args_cli):
        args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
        args_cli, hydra_overrides = args_parser.parse_known_args(environment_argv(sys.argv[1:], embodiment))
        assert_hydra_overrides(hydra_overrides, args_parser)
        if args_cli.video:
            args_cli.enable_cameras = True
        assert args_cli.num_envs == 1, "The challenge runs one environment at a time."

        arena_builder = get_arena_builder_from_cli(args_cli, hydra_overrides=hydra_overrides)
        env_cfg, env_kwargs = arena_builder.compose_manager_cfg()
        if args_cli.video:
            add_video_camera(env_cfg)
        env = arena_builder.make_registered(env_cfg, env_kwargs)

        high_level = import_class(team["high_level"]["class"])(**team["high_level"]["options"])
        low_level = (
            None if builtin_low_level else import_class(team["low_level"]["class"])(**team["low_level"]["options"])
        )
        policy = HierarchicalPolicy(high_level, low_level)
        policy.setup(env)

        team_name = team["team_name"]
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(args_cli.challenge_output_dir, f"{team_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        rows = []
        for seed in episode_seeds(args_cli):
            video_path = os.path.join(output_dir, f"seed_{seed:04d}.mp4") if args_cli.video else None
            row = run_episode(env, policy, seed, video_path, (args_cli.video_eye, args_cli.video_lookat))
            rows.append(row)
            print(
                f"[challenge] seed {seed}: {row['termination']} after {row['sim_time_s']:.2f} s sim"
                f" ({row['wall_time_s']:.1f} s wall)",
                flush=True,
            )

        summary = summarize(rows, float(env.unwrapped.max_episode_length_s))
        metadata = {
            "challenge": CHALLENGE_ENVIRONMENT,
            "team_name": team_name,
            "team_file": os.path.abspath(args_cli.team),
            "high_level": team["high_level"],
            "low_level": team["low_level"],
            "embodiment": embodiment,
            "timestamp": timestamp,
            "seeds": [row["seed"] for row in rows],
            "control_dt_s": env.unwrapped.step_dt,
            "video": bool(args_cli.video),
        }
        print_table(
            rows, summary, f"{team_name} (high: {team['high_level']['class']}, low: {team['low_level']['class']})"
        )
        write_results(output_dir, rows, summary, metadata)
        print(f"[challenge] Results written to {os.path.abspath(output_dir)}")
        env.close()


if __name__ == "__main__":
    main()
