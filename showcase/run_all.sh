#!/usr/bin/env bash
# Capture every clip of the two showcase videos, one Isaac Sim process per clip.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=$ROOT/.venv/bin/python
cd "$ROOT"
export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p $D/clips $D/logs
ONLY="${1:-all}"

clip() {  # name mode label extra-env... -- cli args...
  local name=$1 mode=$2 label=$3; shift 3
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  if [ -f "$D/clips/$name.mp4" ] && [ -f "$D/clips/$name.json" ]; then echo "skip $name"; return; fi
  echo "=== $(date +%T) $name"
  env SHOWCASE_MODE=$mode SHOWCASE_LABEL="$label" SHOWCASE_OUT=$D/clips/$name.mp4 "${envs[@]}" \
    timeout 2400 $PY $D/capture_clip.py --device cpu --enable_cameras "$@" > $D/logs/$name.log 2>&1
  echo "    exit=$? $(grep -E '\[showcase\] (episode|wrote)' $D/logs/$name.log | tr '\n' ' ')"
}

KITCHEN="g1_kitchen_pick_and_place --teleop_device g1_keyboard"

if [ "$ONLY" = all ] || [ "$ONLY" = tour ]; then
  clip t00_galileo tour "Galileo warehouse  |  official Arena G1 task" SHOWCASE_SECONDS=5 -- \
    galileo_g1_locomanip_pick_and_place --teleop_device g1_keyboard
  i=1
  for pair in farmhouse1:ranch_dressing_hope_robolab farmhouse2:ketchup_bottle_hope_robolab \
              coastal:mustard_bottle_hope_robolab industrial:bbq_sauce_bottle_hope_robolab \
              mediterranean:mayonnaise_bottle_hope_robolab modern1:alphabet_soup_can_hope_robolab \
              modern2:objaverse_mug_0 rustic:apple_01_objaverse_robolab \
              scandinavian:objaverse_teacup_0 traditional:objaverse_banana_0; do
    style=${pair%%:*}; obj=${pair##*:}
    clip t$(printf %02d $i)_$style tour "Kitchen one_wall / $style" SHOWCASE_SECONDS=4 -- \
      $KITCHEN --kitchen_background lightwheel_kitchen_one_wall_$style --object $obj
    i=$((i + 1))
  done
  clip t20_teleop teleop "" -- $KITCHEN
fi

if [ "$ONLY" = all ] || [ "$ONLY" = expert ]; then
  count() { $PY -c "import json,sys; print(json.load(open(sys.argv[1])).get('recorded',0))" "$1" 2>/dev/null || echo 0; }
  n=0
  seed=11
  for pair in farmhouse1:ranch_dressing_hope_robolab modern2:ketchup_bottle_hope_robolab \
              scandinavian:bbq_sauce_bottle_hope_robolab industrial:alphabet_soup_can_hope_robolab; do
    style=${pair%%:*}; obj=${pair##*:}
    name=e$(printf %02d $seed)_$style
    clip $name expert "kitchen one_wall / $style" SHOWCASE_EPISODES=2 SHOWCASE_SEED=$seed SHOWCASE_DEMO_COUNT=$n -- \
      $KITCHEN --kitchen_background lightwheel_kitchen_one_wall_$style --object $obj
    n=$(count $D/clips/$name.json)
    seed=$((seed + 1))
  done
  clip e20_galileo expert "Galileo warehouse (two-hand box)" SHOWCASE_EXPERT=bimanual SHOWCASE_EPISODES=2 \
    SHOWCASE_SEED=20 SHOWCASE_DEMO_COUNT=$n SHOWCASE_RECOLOR_COUNTER=0 -- \
    galileo_g1_locomanip_pick_and_place --object brown_box --embodiment g1_wbc_pink
fi
echo "=== $(date +%T) done"
