#!/bin/sh
# Waits for the test cache, then runs the whole held-out analysis in order.
cd "$(dirname "$0")/.." || exit 1
export HF_HOME="$PWD/data/hf_cache"
while pgrep -f "resolution_sweep.cache --split test" >/dev/null; do sleep 20; done
n=$(find archive/test -name '*.npz' | wc -l)
echo "test cache complete: $n chips" 
[ "$n" -ge 400 ] || { echo "cache incomplete, stopping"; exit 1; }
.venv/bin/python -u scripts/add_colour_variant.py --cache archive/test --variants r120_c15 || exit 1
echo "colour variant added"
.venv/bin/python -u -m resolution_sweep.report --cache archive/test --split test --out results/test || exit 1
echo "report done"
.venv/bin/python -u scripts/real_sensor_check.py --out results/test_real_sensor || exit 1
echo "ALL DONE"
