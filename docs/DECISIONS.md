# Decisions, verified facts and open items

## Dataset choice

Primary: SpaceNet 2 building detection, cities Paris and Khartoum. Reasons: polygon labels per
building, a clear licence, a public bucket, and about 2,160 chips across two different building
types. Paris has 1,148 images and Khartoum 1,012 per the torchgeo documentation table.

Verified in this project from public pages:

- Chips: 650 x 650 pixels, pan-sharpened RGB at 0.30 m (torchgeo SpaceNet 2 documentation). That is
  about 195 m on the ground. The SpaceNet page says 200 m x 200 m for training tiles.
- Labels: building polygons, GeoJSON (torchgeo documentation).
- Licence: CC BY-SA 4.0, stated on https://spacenet.ai/spacenet-buildings-dataset-v2/. The AWS
  registry only says "Various" and points to spacenet.ai. Share-alike applies to adapted data, so
  commit code and result tables, not degraded images. Not legal advice.
- Download: `s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/`, no AWS account needed with
  `--no-sign-request`. Sizes: Paris 5.3 GB, Khartoum 4.7 GB (Las Vegas 23 GB and Shanghai 23.4 GB
  are skipped).
- Model input sizes from the Hugging Face configs: OWL-ViT base patch32 uses 768 px, OWLv2 base
  patch16 ensemble uses 960 px. OWLv2 is Apache-2.0.

Rejected:

- WHU aerial: COCO labels at 0.2 m and a single city (Christchurch). Its page states no licence and
  asks only for citation, so reuse terms are unclear.
- INRIA: mask labels, with binary masks merging touching buildings into one region (expected
  behaviour of binary masks, not checked here). Terms were read from a secondary documentation
  mirror, not the original source.

## Review against the original repo (read-only, 8 Oct)

- Matcher and metrics: same logic as `DetectionValidator` (assignment reward, size rule, formulas).
- Colour rule: the original uses scipy's default 4-connectivity. This repo first defaulted to 8, which
  gives 1,540 regions on Ankara instead of 2,756. Fixed, with a test.
- OWL wrapper: the original clips boxes to the image and drops empty ones before the size filter.
  `evaluate.select_boxes` now does the same from a stored chip extent.
- Interface: the original `detect` returns a GeoDataFrame in EPSG:4326 with a 1-based `detection_id`.
  This repo returns `list[Detection]` in a metric frame. Add an adapter when merging into `src/`.
- Parity (scripts/parity_check.py, all seven archives): saved detections and colour detections rebuilt
  from image.tif both reproduce the paper's raw counts, kept counts and match counts exactly.
- Still to run: `scripts/parity_owl.py` (needs torch) to check the OWL wrapper against owlvit_raw.json.

## Design decisions

1. Windows are whole chips. A 650 px chip is below the models' input sizes, so no tiling is needed.
2. Each chip is reduced to the target GSD and enlarged back to 650 px with bicubic. Detector boxes stay
   in native coordinates and are scored against the original polygons.
3. Levels: 0.3, 0.45, 0.6, 0.9, 1.2, 1.8, 2.4, 3.6, 5.0, 10.0 m. Pixel counts: 650, 433, 325, 217,
   163, 108, 81, 54, 39, 20. Integer rounding makes the 10 m level 9.75 m effective.
4. Resize methods: area averaging, and Gaussian blur (sigma = 0.5 x factor) then bilinear decimation.
5. Matching follows the paper: one-to-one, maximise pair count first, total IoU second, IoU 0.25 and
   0.5. Metrics use the paper's formulas and are regression-tested against its Table 2.
6. Everything is cached before thresholding. OWL scores are stored per prompt, so prompt choice is a
   dev-time decision made without another forward pass.
7. Bootstrap resamples geographic blocks, not chips or buildings. Curves use paired replicates so a
   break point can be computed per replicate.

## Proposals to freeze on Sunday 11 October, before the test split is touched

- Dev tuning rule: choose the setting with the highest mean pooled F1 across all levels at IoU 0.25.
  Also report a variant tuned at 0.3 m only.
- Floor rule (replaces the arbitrary 5% idea): a chance baseline relocates every predicted box uniformly
  at random on its chip, keeping count and sizes, and the F1 it earns is the floor at each level
  (`baseline.py`, 20 relocations, seeded per chip). A detector counts as above chance at the finest level
  only if the lower bootstrap bound of its F1 exceeds the chance F1 (`analysis.above_chance`). If it does
  not, report a floor result and the best level, not a break point.
- Break point: the GSD where normalised F1, (F1 - chance F1) / (finest F1 - finest chance F1), first
  falls to 0.5, interpolated in log GSD, with a CI from paired block-bootstrap replicates.
- Real-sensor check: first the 1.24 m multispectral data from the same dataset, if available and band
  order is verified on disk. NAIP via STAC only on Thu 15 Oct, one day maximum, and cut if the MVP is
  not done by Wed 14.
- The test split is run once per frozen configuration.

## Open items for Friday (not verified on disk)

- Extracted folder layout, so the globs in `configs/sweep.yaml` are guesses.
- Bit depth of PS-RGB. The reader raises on anything other than uint8 so a scaling rule is chosen on
  purpose.
- CRS of the chips and the unit of `block_cell`.
- Whether polygons are already clipped to the chip and how empty-label chips appear.
- Band order and availability of the 163 x 163 multispectral file at 1.24 m. If it is in the tarball, it
  allows a small real-sensor comparison against area-averaged 1.2 m imagery at no extra download.
- AWS CLI is not installed on this Mac.

## Limits to state in the write-up

- SpaceNet 2 is WorldView-3 pan-sharpened imagery. Degrading it simulates resolution, not Sentinel-2
  point spread, noise or spectral response.
- Colour thresholds were chosen for Sentinel-2 display colours and are run unchanged as one variant.
- Two cities do not support claims about other places.
- Reference labels are human-drawn and their quality varies by city.


## Verified on disk (Fri 9 Oct, 00:00)

- Layout: `<tarball>/AOI_x_Train/{RGB-PanSharpen,geojson/buildings}`. 1,148 Paris and 1,012 Khartoum chips.
- PS-RGB chips are uint16, 650 x 650, EPSG:4326, about 0.3 m. Band order is R, G, B (red roofs render red,
  label polygons overlay buildings correctly on a checked chip).
- Brightness differs a lot by city (median red about 190 Paris, about 520 Khartoum). Scale constants per city
  (700 and 950, about the 99.5th percentile of all band values on 150 random chips, intensities only), one
  constant for all three bands so band ratios survive.
- About 8% (Paris) and 17% (Khartoum) of pixels are zero (chip borders). Left as black, not masked. A limit.
- Empty chips: about 51% of Paris chips and 10% of Khartoum chips have no building labels. Counts pool, so they
  only add false positives.
- OWL-ViT wrapper reproduces the paper's saved raw output on the Ankara archive (boxes within 0.01 px, scores exact).

## Frozen on dev before the test split was evaluated

Rule: highest mean pooled dev F1 over the ten levels, IoU 0.25, area method. Settings in `configs/frozen.yaml`.
The tuning grid was widened twice after the first pass hit its edges (thresholds, size caps). The first pass
had the paper prompts looking worse; with lower thresholds they came out best for OWL-ViT. Top settings are
within about one F1 point of each other, so the choice among them is close to a tie.
Tuned on the area method only. The blur+resize method reuses the same settings.
A per-level setting (best dev setting at each level) is reported as an upper bound.
The test cache was built before the freeze. That stores raw model outputs only and does not touch the labels.

## 9 Oct 2026: held-out results and coarse-GSD diagnostic
- Test split scored once (400 chips). Results in `results/test`, tables in `docs/RESULTS.md`, paper in `docs/PAPER.md`.
- OWLv2 revision used: cfd3195ba4ea9592eec887ded089f4c08eff231d. OWL-ViT: cbc355fb364588351c5d51c7f74465e8e7ec6f72.
- Per-level (re-tuned) curves keep signal at 5 to 10 m (OWLv2 16.3 F1 at 10 m, chance 5.4). `scripts/diagnose_coarse.py` shows OWLv2 at 5 m still matches ordinary-size buildings; OWL-ViT at 10 m matches only large buildings with chip-sized boxes. Mechanism for OWLv2 untested (clustering is not in the chance baseline).
- Paper abstract and discussion report both frozen and per-level results.

## 9 Oct 2026, 14:10: prediction written down before the Las Vegas results exist
Las Vegas (200 chips, never used for tuning) has a median reference side of 14.4 m (Paris 11.1 m, Khartoum 10.6 m; 44 chips measured at this point, final value will be recomputed).
If the break point follows building size in pixels, the pooled Paris and Khartoum values (OWL-ViT area 2.34 m at 10.9 m median, i.e. 4.7 px; OWLv2 area 2.87 m, i.e. 3.8 px) predict for Vegas:
- OWL-ViT, area, IoU 0.25: break near 14.4 / 4.7 = 3.1 m
- OWLv2, area, IoU 0.25: break near 14.4 / 3.8 = 3.8 m
Frozen settings are used unchanged. The settings and the Vegas scale constant (1500) were fixed before any Vegas image went through a model.
Result (14:45): break points came out at 2.31 m [2.16, 2.53] (OWL-ViT) and 3.10 m [2.90, 3.32] (OWLv2). Both predictions were too high and outside their intervals. Final Vegas median side over all 200 chips is 13.9 m (14.4 m was from the first 44 chips); predictions with 13.9 m would be 3.0 and 3.7 m. Paper reports the failed prediction and drops the "five pixels per building" reading.
