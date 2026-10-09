#!/bin/sh
cd "$(dirname "$0")/.." || exit 1
export HF_HOME="$PWD/data/hf_cache"
.venv/bin/python -u scripts/add_colour_variant.py --cache archive/test --variants r100_c15 r140_c15 || exit 1
echo "more colour variants added"
.venv/bin/python -u -m resolution_sweep.report --cache archive/test --split test --out results/test || exit 1
echo "report done"
.venv/bin/python -u scripts/real_sensor_check.py --out results/test_real_sensor || exit 1
echo "ALL DONE"
