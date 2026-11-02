# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
import traceback

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True
IDLE_STEPS = 20
MAX_SETTLE_STEPS = 100
WALK_STEPS = 100


def _test_g1_fast_transport_goal_and_terminations(simulation_app) -> bool:
    import torch

    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser
    from isaaclab_arena_g1.challenge.challenge_runner import termination_reason
    from isaaclab_arena_g1.challenge.fast_transport_task import FAST_TRANSPORT_STATE_ATTR, START_OFFSET_SEED_ATTR
    from isaaclab_arena_g1.challenge.template_policy import rest_action, split_challenge_observation

    parser = get_isaaclab_arena_environments_cli_parser(get_isaaclab_arena_cli_parser())
    args_cli = parser.parse_args(["g1_fast_transport"])
    env = get_arena_builder_from_cli(args_cli).make_registered()
    try:
        unwrapped = env.unwrapped
        object_name = "bbq_sauce_bottle_hope_robolab"

        def step_until_done(max_steps: int) -> str | None:
            for _ in range(max_steps):
                _, _, terminated, truncated, _ = env.step(rest_action(1, unwrapped.device))
                if bool(terminated[0]) or bool(truncated[0]):
                    return termination_reason(env)
            return None

        def teleport_object(position_e: torch.Tensor, quat_xyzw: torch.Tensor) -> None:
            pose = torch.cat([position_e + unwrapped.scene.env_origins[0], quat_xyzw]).unsqueeze(0)
            unwrapped.scene[object_name].write_root_pose_to_sim(pose)
            unwrapped.scene[object_name].write_root_velocity_to_sim(torch.zeros(1, 6, device=unwrapped.device))

        with torch.inference_mode():
            # The same seed gives the same start offset.
            offsets = []
            for _ in range(2):
                setattr(unwrapped, START_OFFSET_SEED_ATTR, 7)
                obs, _ = env.reset(seed=7)
                offsets.append(getattr(unwrapped, FAST_TRANSPORT_STATE_ATTR).start_offset[0].clone())
            assert torch.allclose(offsets[0], offsets[1]), f"Start offsets differ for one seed: {offsets}"
            assert offsets[0].abs().max() <= 0.05 + 1e-6, f"Start offset beyond 5 cm: {offsets[0]}"

            state = split_challenge_observation(obs)
            assert obs["challenge"].shape == (1, 22), f"Unexpected challenge observation shape {obs['challenge'].shape}"
            _check_api(env, obs)
            # The goal is at the center of the table across the room (x from the counter's left end: table -2.3,
            # object 0.44), turned by 180 deg about the vertical.
            goal_pos, object_pos = state["goal_pos"][0], state["object_pos"][0]
            assert (
                abs((goal_pos[0] - object_pos[0]) - (-2.3 - 0.44)) < 0.01
            ), f"Goal x {goal_pos[0]}, object x {object_pos[0]}"
            assert abs(goal_pos[1] + 2.3) < 1e-3, f"Goal y {goal_pos[1]} is not the table center"
            goal_yaw = 2.0 * math.atan2(float(state["goal_quat"][0, 2]), float(state["goal_quat"][0, 3]))
            assert abs(math.remainder(goal_yaw - math.radians(90.0), 2.0 * math.pi)) < 1e-3, f"Goal yaw {goal_yaw}"

            # Standing still does not end the episode.
            assert step_until_done(IDLE_STEPS) is None, "The episode ended while the robot stood still."

            # The object resting at the goal succeeds.
            teleport_object(
                state["goal_pos"][0] + torch.tensor([0.0, 0.0, 0.005], device=unwrapped.device), state["goal_quat"][0]
            )
            reason = step_until_done(MAX_SETTLE_STEPS)
            assert reason == "success", f"Expected success with the object at the goal, got {reason}"

            # The object on the floor is a drop.
            env.reset(seed=8)
            teleport_object(torch.tensor([3.0, -1.5, 0.1], device=unwrapped.device), state["goal_quat"][0])
            reason = step_until_done(5)
            assert reason == "object_dropped", f"Expected object_dropped, got {reason}"
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        env.close()
    return True


def _check_api(env, obs) -> None:
    """The participant API agrees with the challenge observation and returns the documented shapes."""
    import torch

    from isaaclab_arena_g1.challenge import api

    named = api.read_challenge_observation(obs)
    assert torch.allclose(api.object_pose(env)[:, :3], named["object_pos"], atol=1e-4)
    assert torch.allclose(api.goal_pose(env)[:, :3], named["goal_pos"], atol=1e-5)
    assert torch.allclose(api.pelvis_pose(env)[:, :3], named["pelvis_pos"], atol=1e-4)
    assert api.object_name(env) == "bbq_sauce_bottle_hope_robolab", api.object_name(env)
    assert api.object_velocity(env).shape == (1, 6)
    assert api.pelvis_velocity(env).shape == (1, 6)
    assert api.object_corners(env).shape == (1, 8, 3)
    distance, angle = api.goal_error(env)
    assert distance.shape == (1,) and angle.shape == (1,)
    assert api.joint_positions(env).shape == (1, 43) and api.joint_velocities(env).shape == (1, 43)
    assert len(api.joint_names(env)) == 43 and len(api.body_names(env)) == 53
    assert api.body_pose(env, "right_hand_palm_link").shape == (1, 7)
    # The desk top is at 0.75 m and the counter top at about 0.92 m.
    assert abs(float(api.desk_corners(env)[0, :, 2].max()) - 0.75) < 0.01, api.desk_corners(env)
    assert abs(float(api.counter_top_corners(env)[0, :, 2].max()) - 0.92) < 0.02, api.counter_top_corners(env)
    assert api.control_dt(env) == 0.02 and api.time_limit(env) == 90.0
    assert api.elapsed_time(env).shape == (1,)
    # Arms hang down at the start: wrists beside the hips, below the pelvis, in the pelvis frame.
    for side, sign in (("left", 1.0), ("right", -1.0)):
        wrist = api.wrist_pose_in_pelvis(env, side)
        assert wrist[0, 2] < 0.0 and sign * wrist[0, 1] > 0.1, f"{side} wrist {wrist}"
    # The pelvis seen from the pelvis frame is at its origin.
    origin = api.env_to_pelvis_points(env, api.pelvis_pose(env)[:, :3])
    assert origin.abs().max() < 1e-4, origin
    action = api.make_action(env, right_hand_closed=True, walk_forward=0.2)
    assert action.shape == (1, 23) and action[0, 1] == 1.0 and abs(float(action[0, 16]) - 0.2) < 1e-6


def test_g1_fast_transport_goal_and_terminations():
    result = run_function_with_persistent_simulation_app(
        _test_g1_fast_transport_goal_and_terminations, headless=HEADLESS
    )
    assert result, "Test failed"


def _test_g1_fast_transport_custom_low_level(simulation_app) -> bool:
    import torch

    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser
    from isaaclab_arena_g1.challenge import api
    from isaaclab_arena_g1.challenge.challenge_main import HierarchicalPolicy
    from isaaclab_arena_g1.challenge.high_level.template import TemplateHighLevel
    from isaaclab_arena_g1.challenge.interfaces import Command
    from isaaclab_arena_g1.challenge.low_level.wbc import WbcLowLevel

    parser = get_isaaclab_arena_environments_cli_parser(get_isaaclab_arena_cli_parser())
    args_cli = parser.parse_args(["g1_fast_transport", "--embodiment", "g1_joint_position"])
    env = get_arena_builder_from_cli(args_cli).make_registered()
    try:
        assert env.unwrapped.action_manager.total_action_dim == 43, env.unwrapped.action_manager.total_action_dim
        policy = HierarchicalPolicy(TemplateHighLevel(), WbcLowLevel())
        policy.setup(env)
        with torch.inference_mode():
            obs, _ = env.reset(seed=0)
            policy.reset()
            # A command survives the trip through the 23-D layout.
            command = Command.from_tensor(api.make_action(env, right_hand_closed=True, walk_forward=0.2))
            assert torch.equal(Command.from_tensor(command.to_tensor()).to_tensor(), command.to_tensor())
            start = api.pelvis_pose(env)[0, :3].clone()
            for _ in range(WALK_STEPS):
                obs, _, terminated, truncated, _ = env.step(policy.get_action(env, obs))
                assert not bool(terminated[0]) and not bool(truncated[0]), "The episode ended while walking."
            pelvis = api.pelvis_pose(env)[0, :3]
            # The template walks toward the counter (+y) through the team low level and stays upright.
            assert pelvis[2] > 0.6, f"Pelvis dropped to {pelvis[2]}"
            assert pelvis[1] - start[1] > 0.1, f"The robot did not walk forward: {start} -> {pelvis}"
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False
    finally:
        env.close()
    return True


def test_g1_fast_transport_custom_low_level():
    result = run_function_with_persistent_simulation_app(_test_g1_fast_transport_custom_low_level, headless=HEADLESS)
    assert result, "Test failed"
