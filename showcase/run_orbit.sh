#!/usr/bin/env bash
# Capture the 10 kitchen-style orbit slices (no HUD; assemble.py adds the style labels).
# ./run_orbit.sh          robot standing in every kitchen -> clips/r00_* .. r09_*
# ./run_orbit.sh norobot  robot hidden                     -> clips/o00_* .. o09_*
set -u
D="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=$ROOT/.venv/bin/python
cd "$ROOT"
export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p $D/clips $D/logs
if [ "${1:-robot}" = norobot ]; then PREFIX=o; HIDE=1; else PREFIX=r; HIDE=0; fi
STYLES="farmhouse1 coastal industrial mediterranean modern1 modern2 rustic scandinavian traditional farmhouse2"
i=0
for style in $STYLES; do
  name=$PREFIX$(printf %02d $i)_$style
  if [ -f "$D/clips/$name.mp4" ] && [ -f "$D/clips/$name.json" ]; then echo "skip $name"; i=$((i + 1)); continue; fi
  echo "=== $(date +%T) $name"
  env SHOWCASE_OUT=$D/clips/$name.mp4 ORBIT_HIDE_ROBOT=$HIDE ORBIT_INDEX=$i ORBIT_COUNT=10 \
    ORBIT_SEG=3.0 ORBIT_FADE=1.0 timeout 1200 $PY $D/capture_orbit.py --device cpu --enable_cameras \
    g1_kitchen_pick_and_place --teleop_device g1_keyboard --kitchen_background lightwheel_kitchen_one_wall_$style \
    > $D/logs/$name.log 2>&1
  echo "    exit=$? $(grep -a '\[orbit\] wrote' $D/logs/$name.log)"
  i=$((i + 1))
done
echo "=== $(date +%T) done"
