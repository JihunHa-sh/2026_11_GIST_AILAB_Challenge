#!/usr/bin/env bash
# Capture the "user picks object and destination" expert clips (f*): one successful episode each, with
# floating pick (blue) / place (green) arrows and the picked object's own look. Skips clips that exist.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=$ROOT/.venv/bin/python
cd "$ROOT"
export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p $D/clips $D/logs
KITCHEN="g1_kitchen_pick_and_place --teleop_device g1_keyboard"
TOASTER="--counter_prim_path counter_main_main_group/top_geometry_right --object_counter_x 0.25 --robot_counter_x 0.45 \
  --destination_counter_x 0.9 --destination_y -0.37 --destination_open"

clip() {  # name style seed object destination [extra env args...]
  local name=$1 style=$2 seed=$3 obj=$4 dst=$5; shift 5
  if [ -f "$D/clips/$name.mp4" ] && [ -f "$D/clips/$name.json" ]; then echo "skip $name"; return; fi
  echo "=== $(date +%T) $name"
  env SHOWCASE_MODE=expert SHOWCASE_MARKERS=1 SHOWCASE_STOP_AFTER_SUCCESS=1 SHOWCASE_RECOLOR_OBJECT=0 \
    SHOWCASE_EPISODES=${EPISODES:-4} SHOWCASE_SEED=$seed SHOWCASE_LABEL="kitchen one_wall / $style" \
    SHOWCASE_OUT=$D/clips/$name.mp4 timeout 3000 $PY $D/capture_clip.py --device cpu --enable_cameras \
    $KITCHEN --kitchen_background lightwheel_kitchen_one_wall_$style --object $obj --destination $dst "$@" \
    > $D/logs/$name.log 2>&1
  echo "    exit=$? $(grep -aE '\[showcase\] (episode|wrote)' $D/logs/$name.log | tr '\n' ' ')"
}

clip f01_pear coastal 31 objaverse_pear_3 plate_small_vomp_robolab
clip f02_water_bottle modern1 32 objaverse_water_bottle_2 bowl_ycb_robolab
clip f03_duck rustic 33 objaverse_duck_0 wooden_bowl_hot3d_robolab
clip f04_ketchup modern2 34 ketchup_bottle_hope_robolab bowl_ycb_robolab
clip f05_soup_toaster farmhouse1 35 alphabet_soup_can_hope_robolab toaster_oven $TOASTER
clip f06_mug scandinavian 36 objaverse_mug_0 serving_bowl_vomp_robolab
echo "=== $(date +%T) done"
