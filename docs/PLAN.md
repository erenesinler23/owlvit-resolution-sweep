# Scope and schedule

Question: at what resolution does open-vocabulary building detection break down, and how does a colour rule compare?

Work ran from 8 to 9 October 2026 and is complete. The final write-up is in `paper/`, the source text in `docs/PAPER.md`, and every table in `docs/RESULTS.md`.

## What was planned and what changed

- The first plan covered Paris and Khartoum, area averaging, the colour rule and OWL-ViT, with block-bootstrap intervals.
- Added later: OWLv2, blur and resize as a second degradation method, a native 1.2 m check, a third city (Las Vegas) with a prediction written down before the run, an oracle-location baseline, and real Sentinel-2 crops over the same footprints.
- Dropped: a NAIP check. Real Sentinel-2 imagery answered the same question more directly.

## Order of work

1. Download SpaceNet 2 and inspect one chip (layout, bit depth, projection, band order).
2. Split on spatial blocks and cache raw model output on 150 development chips.
3. Tune on the development chips, then freeze settings in `configs/frozen.yaml`.
4. Cache and score the test chips once.
5. Run the baselines, the Las Vegas test and the Sentinel-2 comparison.
6. Write the paper and regenerate `docs/RESULTS.md`.

## Compute

Each chip takes 10 levels and 2 resize methods, so 20 forward passes per model. With 150 development and 400 test chips that is about 11,000 passes per model, run overnight on an Apple M3 Pro with `PYTORCH_ENABLE_MPS_FALLBACK=1`.
