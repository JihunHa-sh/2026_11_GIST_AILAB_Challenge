# G1 showcase videos and page

Two videos and a web page that introduce the G1 keyboard teleop, the G1 kitchen environment and the
scripted demo generators. Published page: https://claude.ai/artifact/QxTDKQ1NfnvbVYk6ZMQ214 (private to the owner).

Media files (`*.mp4`, `*.jpg`, `*.png`) are git-ignored; keep them out of commits.

## Files

| Path | What it is |
|---|---|
| `showcase_lib.py` | Shared helpers: extra scene camera, headless keyboard stub, in-place material recolor, light randomization, HUD overlays, ffmpeg encoder |
| `capture_clip.py` | Captures one clip (mp4 + json) in one Isaac Sim process. Mode via env vars, Arena env CLI as arguments |
| `capture_orbit.py` | Captures one slice of the kitchen orbit: robot standing or hidden (`ORBIT_HIDE_ROBOT=1`), physics paused, camera on a path shared by every style (`ORBIT_INDEX`, `ORBIT_SEG`, `ORBIT_FADE`) |
| `run_orbit.sh` | Captures the 10 orbit slices: robot standing `clips/r00_*`-`r09_*` (default), or robot hidden with `./run_orbit.sh norobot` -> `clips/o00_*`-`o09_*` (`assemble.py intro_norobot` -> `final/g1_env_intro_norobot.mp4`); `assemble.py` crossfades them (1 s) and adds the style labels |
| `capture_grasp.py` | Runs one-hand expert episodes until one lifts the object; writes the episode mp4 (close-up from REACH_SIDE on) plus json/npz with every grasp-planner call, cuRobo pre-grasp attempt, per-frame camera pose and the object mesh |
| `render_grasp_steps.py` | Re-runs the grasp planner on those inputs with every sample logged and animates the steps (matplotlib 3D), then the sim footage with the chosen grasp projected on it -> `segments/grasp_steps.mp4` + chapters |
| `run_selected.sh` | Captures the auto-data clips `clips/f*`: one successful episode per user-chosen object/destination (`--object`, `--destination`), with floating pick (blue) / place (green) arrows (`SHOWCASE_MARKERS=1`) and the object's own look (`SHOWCASE_RECOLOR_OBJECT=0`) |
| `selection_cards.py` | Terminal segments that type the real `objaverse_import` and run commands (user's choices highlighted) next to a spinning preview of the downloaded model |
| `run_all.sh` | Captures every clip in sequence; skips clips that already have both mp4 and json. `./run_all.sh tour` or `./run_all.sh expert` for one half |
| `assemble.py` | Builds the two final videos from the clips: title cards, cuts, speed-ups, outcome badges, 2-pass H.264 to about 14 MB, chapter json |
| `clips/` | Raw clips. `t00` warehouse tour, `t01`-`t10` kitchen style tours, `t20` keyboard teleop, `e11`-`e14` kitchen expert, `e20` warehouse box expert. Each json has labels and, for experts, per-episode frames and outcome |
| `final/` | `g1_env_intro.mp4` (66 s), `g1_auto_data.mp4` (120 s), posters and chapter json |
| `site/index.html` | The page; videos in `site/videos/`. Chapter lists are the two `<script type="application/json">` blocks at the bottom |
| `textures/` | Texture pool for recoloring (procedural `proc_*.png`, generated on first use) |
| `probe_visuals.py`, `probe_recolor.py` | Small probes used to check the camera and the recolor |

## Running

From the repo root, with the native `.venv` and `export OMNI_KIT_ACCEPT_EULA=YES`:

```bash
# one clip, e.g. a kitchen expert run with 2 episodes
SHOWCASE_MODE=expert SHOWCASE_EPISODES=2 SHOWCASE_SEED=12 SHOWCASE_OUT=showcase/clips/e12_modern2.mp4 \
  .venv/bin/python showcase/capture_clip.py --device cpu --enable_cameras \
  g1_kitchen_pick_and_place --teleop_device g1_keyboard \
  --kitchen_background lightwheel_kitchen_one_wall_modern2 --object ketchup_bottle_hope_robolab

# everything that is missing, then the two videos
showcase/run_all.sh all
cd showcase && ../.venv/bin/python assemble.py          # or: assemble.py intro | assemble.py auto
```

`capture_clip.py` environment variables:

| Variable | Meaning |
|---|---|
| `SHOWCASE_MODE` | `tour` (robot stands, camera sweeps), `teleop` (scripted key presses, see `TELEOP_SCRIPT`), `expert` (scripted policy episodes) |
| `SHOWCASE_OUT` | Output mp4; a json with the same name is written next to it |
| `SHOWCASE_LABEL` | Scene label shown in the HUD |
| `SHOWCASE_SECONDS` | Tour length |
| `SHOWCASE_EPISODES`, `SHOWCASE_SEED` | Expert episodes and randomization seed |
| `SHOWCASE_EXPERT` | `one_hand` (default) or `bimanual` |
| `SHOWCASE_DEMO_COUNT` | Starting value of the "recorded demos" HUD counter |
| `SHOWCASE_RECOLOR_COUNTER` | `1` (default) also recolors the kitchen counter top |

To change what the videos show, edit the clip list in `run_all.sh` (scenes, objects, seeds), delete the clips
you want re-shot, rerun `run_all.sh`, then `assemble.py`. Card text and pacing are in `assemble.py`
(`env_intro()`, `auto_data()`; each expert episode is sped up to about 10 s). The page text is plain HTML in
`site/index.html`; after replacing `site/videos/*`, republish the page to the same URL.

`assemble.py auto` builds the current auto-data video from the `f*` clips (selection segment + first successful
episode each, plus the warehouse box); `assemble.py auto_v1` rebuilds the first version (every `e*` episode,
failures included; kept as `final/g1_auto_data_v1.mp4`).

Grasp-planning video (`final/g1_grasp_steps.mp4`, page section "영상 3"):

```bash
SHOWCASE_OUT=showcase/clips/g00_mug.mp4 SHOWCASE_SEED=3 .venv/bin/python showcase/capture_grasp.py --device cpu \
  --enable_cameras g1_kitchen_pick_and_place --teleop_device g1_keyboard \
  --kitchen_background lightwheel_kitchen_one_wall_farmhouse1 --object objaverse_mug_0
cd showcase && ../.venv/bin/python render_grasp_steps.py clips/g00_mug
```

## Things that were not obvious

- An extra camera must live under `{ENV_REGEX_NS}` (e.g. `{ENV_REGEX_NS}/ShowcaseCam`). Placed at `/World/...` it
  renders a flat white image. Isaac Lab 3 no longer returns frames from `render_mode="rgb_array"`.
- Recoloring edits the inputs of the objects' existing MDL shaders (`diffuse_texture`, `diffuse_color_constant`).
  New material bindings either do not render (`full` purpose) or would also replace physics materials
  (all-purpose), which changes grasp friction.
- RoboCasa counter tops are `Cube` prims bound to a shared material in the kitchen's `Looks` scope, so the recolor
  follows the bindings of any `Gprim`, not only meshes.
- The policy's `reset()` must run inside `torch.inference_mode()` between episodes.
- In the Galileo warehouse a wall stands right behind the robot's start pose, and the floor is at about
  z = -0.8 m, so tours sweep only the open side and the expert clip uses a fixed wide camera.
- When an episode ends the env resets in the same step, so outcome badges are drawn in `assemble.py` on the last
  action frame, not on the frame captured after the step.

## Final capture results

Kitchen one-hand expert: farmhouse1 ranch 0/2 (bottle tipped over), modern2 ketchup 2/2, scandinavian bbq 1/2
(one 90 s time-out), industrial soup can 2/2. Warehouse two-hand box 2/2. All 10 episodes are in the video.
