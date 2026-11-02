#!/usr/bin/env bash
# Record g1_fast_transport demonstrations with the camera teleop (run from anywhere; uses the challenge repo's .venv).
#   ailab_teleop/record_demos.sh [NUM_DEMOS] [DATASET_FILE] [extra record_demos.py args...]
# Successful episodes only are saved (the task's success = bottle on the goal, still, 0.5 s).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
NUM_DEMOS="${1:-10}"
DATASET="${2:-datasets/g1_fast_transport_camera.hdf5}"
shift $(( $# > 2 ? 2 : $# )) || true
mkdir -p "$(dirname "$DATASET")"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python submodules/IsaacLab/scripts/tools/record_demos.py \
  --viz kit --device cpu --enable_cameras \
  --dataset_file "$DATASET" --num_demos "$NUM_DEMOS" --num_success_steps 1 \
  --external_callback ailab_teleop.arena_callback.environment_registration_callback \
  --task g1_fast_transport --teleop_device g1_camera --arena_teleop_device g1_camera "$@"
