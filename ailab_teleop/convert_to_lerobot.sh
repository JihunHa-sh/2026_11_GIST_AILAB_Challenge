#!/usr/bin/env bash
# Convert recorded demos (HDF5) to the LeRobot format GR00T N1.6 trains on (Arena's converter).
#   ailab_teleop/convert_to_lerobot.sh [DATASET_FILE]      (default datasets/g1_fast_transport_camera.hdf5)
# Output: <dataset dir>/<name>/lerobot/ (parquet states/actions, mp4 ego_view videos, meta/).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DATASET="${1:-datasets/g1_fast_transport_camera.hdf5}"
CFG="$(mktemp --suffix=.yaml)"
sed -e "s|^data_root: .*|data_root: \"$(cd "$(dirname "$DATASET")" && pwd)\"|" \
    -e "s|^hdf5_name: .*|hdf5_name: \"$(basename "$DATASET")\"|" ailab_teleop/lerobot_g1_fast_transport.yaml > "$CFG"
.venv/bin/python isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py --yaml_file "$CFG"
rm -f "$CFG"
