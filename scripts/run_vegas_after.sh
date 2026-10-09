#!/bin/sh
# Wait for the Vegas cache to finish, then evaluate everything for that city.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
while pgrep -f "resolution_sweep.cache --config configs/vegas.yaml" > /dev/null; do sleep 20; done
echo "cache done $(date)"
$PY -m resolution_sweep.report --cache archive/vegas --split test --out results/vegas --chance-reps 20
echo "report done $(date)"
$PY scripts/oracle_baseline.py --cache archive/vegas --split test --out results/vegas --reps 10 --iou 0.25
$PY scripts/oracle_baseline.py --cache archive/vegas --split test --out results/vegas --reps 10 --iou 0.5
$PY scripts/oracle_baseline.py --cache archive/vegas --split test --out results/vegas --reps 10 --iou 0.25 --method blur_resize
$PY scripts/oracle_diff.py --dir results/vegas --method area --iou 0.25
$PY scripts/oracle_diff.py --dir results/vegas --method area --iou 0.5
$PY scripts/oracle_diff.py --dir results/vegas --method blur_resize --iou 0.25
echo "oracle done $(date)"
$PY scripts/sentinel2_check.py --cache archive/vegas --cities AOI_2_Vegas --out results/vegas_sentinel2
echo "s2 done $(date)"
$PY scripts/per_city.py --dirs results/test results/vegas --out results/per_city.csv
$PY scripts/make_figures.py --dir results/vegas --out docs/figures --tag _vegas
echo "ALL DONE $(date)"
