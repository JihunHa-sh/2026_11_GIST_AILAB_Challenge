# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

r"""Roll out a policy in an Arena environment and record its successful episodes as demonstrations.

The episodes are written in the same HDF5 format as Isaac Lab's ``record_demos.py``, so they can be
replayed, annotated and augmented with Mimic, or converted to LeRobot like teleoperated demonstrations.
An episode is kept when the task's success term ends it; episodes that fail or time out are dropped.

Record 50 demonstrations of the scripted G1 expert carrying the box to the bin:

    python isaaclab_arena/scripts/record_policy_demos.py \
        --viz kit \
        --device cpu \
        --policy_type isaaclab_arena_g1.policy.g1_scripted_bimanual_pick_and_place_policy.G1ScriptedBimanualPickAndPlacePolicy \
        --dataset_file /datasets/isaaclab_arena/scripted/g1_box_scripted.hdf5 \
        --num_demos 50 \
        galileo_g1_locomanip_pick_and_place \
        --object brown_box \
        --embodiment g1_wbc_pink

Record demonstrations of the scripted one-hand G1 expert putting a bottle into a bowl in a kitchen:

    python isaaclab_arena/scripts/record_policy_demos.py \
        --device cpu \
        --policy_type isaaclab_arena_g1.policy.g1_scripted_one_hand_pick_and_place_policy.G1ScriptedOneHandPickAndPlacePolicy \
        --dataset_file /datasets/isaaclab_arena/scripted/g1_kitchen_scripted.hdf5 \
        --num_demos 50 \
        g1_kitchen_pick_and_place
"""

from __future__ import annotations

import argparse
import os
import torch
from typing import TYPE_CHECKING

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.evaluation.policy_runner import get_policy_cls
from isaaclab_arena.evaluation.policy_runner_cli import add_policy_cli_args, build_policy_from_cli
from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

if TYPE_CHECKING:
    import gymnasium as gym

    from isaaclab_arena.policy.policy_base import PolicyBase


def add_recording_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the policy and dataset arguments of the recorder."""
    parser.add_argument(
        "--policy_type",
        type=str,
        required=True,
        help="Registered policy name, or dotted import path of the policy class.",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="./datasets/policy_demos.hdf5",
        help="HDF5 file the successful episodes are written to.",
    )
    parser.add_argument(
        "--num_demos",
        type=int,
        default=10,
        help="Number of successful episodes to record.",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=None,
        help="Number of episodes after which recording stops even if fewer succeeded (default: 3 x num_demos).",
    )


def create_recording_environment(args_cli: argparse.Namespace, hydra_overrides: list[str]) -> gym.Env:
    """Create the Arena environment with a demonstration recorder that keeps successful episodes only."""
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers import DatasetExportMode

    dataset_dir = os.path.dirname(os.path.abspath(args_cli.dataset_file))
    dataset_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    os.makedirs(dataset_dir, exist_ok=True)

    arena_builder = get_arena_builder_from_cli(args_cli, hydra_overrides=hydra_overrides)
    env_cfg, env_kwargs = arena_builder.compose_manager_cfg()
    # Stored in the dataset, so replay and Mimic scripts find the task without --task, as with record_demos.py.
    env_cfg.env_name = arena_builder.arena_env.name
    # Same recorder as record_demos.py: the embodiment's demo recorder (with cameras) or actions and states.
    if env_cfg.demo_recorder_config is not None:
        env_cfg.recorders = env_cfg.demo_recorder_config
    else:
        env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = dataset_dir
    env_cfg.recorders.dataset_filename = dataset_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.episode_recorders = {}
    return arena_builder.make_registered(env_cfg, env_kwargs)


def record_demos(
    simulation_app: SimulationAppContext,
    env: gym.Env,
    policy: PolicyBase,
    num_demos: int,
    max_attempts: int,
) -> int:
    """Roll out the policy until enough episodes succeeded or the attempts run out.

    Policies may expose a ``gave_up`` property; when it turns true the attempt is cut short instead of
    waiting for the episode to time out.

    Returns:
        The number of successful episodes written to the dataset.
    """
    recorder_manager = env.unwrapped.recorder_manager
    obs, _ = env.reset()
    policy.reset()
    policy.set_task_description(env.unwrapped.get_language_instruction())
    attempts = 0
    with torch.inference_mode():
        while simulation_app.is_running() and attempts < max_attempts:
            obs, _, terminated, truncated, _ = env.step(policy.get_action(env, obs))
            if terminated.any() or truncated.any():
                # The env reset itself; the recorder kept the episode only if the success term ended it.
                ended = True
            elif getattr(policy, "gave_up", False):
                recorder_manager.reset()
                obs, _ = env.reset()
                ended = True
            else:
                ended = False
            if not ended:
                continue
            attempts += 1
            policy.reset()
            num_recorded = recorder_manager.exported_successful_episode_count
            print(f"[record_policy_demos] Attempt {attempts}/{max_attempts}: {num_recorded}/{num_demos} recorded.")
            if num_recorded >= num_demos:
                break
    return recorder_manager.exported_successful_episode_count


def main() -> None:
    """Record successful policy episodes from one Arena environment."""
    args_parser = get_isaaclab_arena_cli_parser()
    args_cli, _ = args_parser.parse_known_args()

    with SimulationAppContext(args_cli) as simulation_app:
        add_recording_arguments(args_parser)
        args_cli, _ = args_parser.parse_known_args()
        policy_cls = get_policy_cls(args_cli.policy_type)

        args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
        args_parser = add_policy_cli_args(args_parser, policy_cls)
        args_cli, hydra_overrides = args_parser.parse_known_args()
        assert_hydra_overrides(hydra_overrides, args_parser)
        assert args_cli.num_envs == 1, "record_policy_demos records one environment at a time"

        env = create_recording_environment(args_cli, hydra_overrides)
        policy = build_policy_from_cli(policy_cls, args_cli)
        max_attempts = args_cli.max_attempts if args_cli.max_attempts is not None else 3 * args_cli.num_demos
        try:
            num_recorded = record_demos(simulation_app, env, policy, args_cli.num_demos, max_attempts)
            print(f"[record_policy_demos] Recorded {num_recorded} successful episodes to {args_cli.dataset_file}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
