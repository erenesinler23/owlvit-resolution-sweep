# owlvit-resolution-sweep

Controlled resolution sweep: at what ground resolution do OWL-ViT (and OWLv2, and a colour rule)
break down on buildings? Imagery comes from SpaceNet 2 (CC BY-SA 4.0) for Paris, Khartoum and Las Vegas,
plus real Sentinel-2 L2A crops over the same chip footprints. Chips are degraded to ten ground
resolutions, detectors run once, and the raw model output is cached so every table recomputes without more forward passes.
The cache (`archive/`, 2.3 GB) is not in the repository. The commands below regenerate it.

Status: complete. Paper: `docs/PAPER.md` (PDF and DOCX in `docs/export/`). All tables: `docs/RESULTS.md`. Figures: `docs/figures/`.
Every choice and its date: `docs/DECISIONS.md`.

The matcher, colour rule and OWL-ViT wrapper reproduce the saved archives of the earlier
Sentinel-2 paper (`scripts/parity_check.py`, `scripts/parity_owl.py`).

## Setup

```
uv venv --python /opt/homebrew/bin/python3.14 .venv
uv pip install --python .venv/bin/python -e ".[data,model,dev]"
.venv/bin/python -m pytest -q
```

## Workflow

```
# data
.venv/bin/python scripts/download_spacenet2.py                    # Paris, Khartoum (about 10 GB)
.venv/bin/python scripts/download_city_subset.py --city AOI_2_Vegas --n 300
.venv/bin/python scripts/setup_city.py --city AOI_2_Vegas --name vegas

# cache raw model output, tune on dev, freeze, score test once
.venv/bin/python -m resolution_sweep.cache --split dev
.venv/bin/python -m resolution_sweep.evaluate tune --cache archive/dev
.venv/bin/python scripts/freeze.py
.venv/bin/python -m resolution_sweep.cache --split test --models owlvit owlv2
.venv/bin/python -m resolution_sweep.report --cache archive/test --split test --out results/test

# extra analyses
.venv/bin/python scripts/oracle_baseline.py --cache archive/test --split test --out results/test
.venv/bin/python scripts/oracle_diff.py --dir results/test
.venv/bin/python scripts/real_sensor_check.py --out results/test_real_sensor        # native 1.2 m
.venv/bin/python scripts/fetch_sentinel2.py --cities AOI_3_Paris AOI_5_Khartoum     # real 10 m
.venv/bin/python scripts/sentinel2_check.py --cities AOI_3_Paris AOI_5_Khartoum --out results/test_sentinel2
.venv/bin/python scripts/per_city.py --dirs results/test results/vegas
.venv/bin/python scripts/make_figures.py --dir results/test
.venv/bin/python scripts/make_results_md.py
```

Settings were chosen on 150 development chips and frozen in `configs/frozen.yaml` before any test
chip was scored. Las Vegas was never used for tuning.

## Licence

Code is under the MIT licence (`LICENSE`). SpaceNet 2 imagery and labels are CC BY-SA 4.0 and are not redistributed here. Sentinel-2 crops come from the Copernicus programme.
