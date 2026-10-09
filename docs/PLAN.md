# Plan: 9 to 20 October 2026

Question: at what resolution does OWL-ViT stop finding buildings, and how does the colour rule compare?
Hard stop: if the study is not finished by 20 October, describe it as in progress in the applications.

## Minimum viable version (due Wed 14 Oct)

One dataset (SpaceNet 2: Paris and Khartoum), area averaging, colour rule and OWL-ViT, IoU 0.25,
block bootstrap, one F1-versus-resolution figure. Extras are added in this order only if the MVP exists:
IoU 0.5, blur plus resize, OWLv2 on a subset, real-sensor check.

## Day by day

| Date | Block | Tasks |
|---|---|---|
| Thu 8 | Setup | Scaffold done. Start `scripts/download_spacenet2.py` (about 10 GB of tarballs). |
| Fri 9 | Data | Extract. Inspect one chip: folder layout, bit depth, CRS, band order, empty labels. Fix globs in `configs/sweep.yaml`. Install `.[data,model]`. Build the index and split. Run `cache --split dev --limit 20` with no model to test the colour path. |
| Sat 10 | Pipeline | `scripts/smoke_owl.py`. Time 20 chips on MPS for OWL-ViT and OWLv2. Matcher and colour parity are already done (see DECISIONS.md). Run `scripts/parity_owl.py` to check the OWL wrapper against the paper's saved raw output. Full dev cache for OWL-ViT with area averaging plus all colour variants. |
| Sun 11 | Tuning | `evaluate tune` on dev. Freeze prompts, threshold, size filters and colour variant in `configs/frozen.yaml`. Freeze the break-point rule. Tag the commit. The test split is not touched before this. |
| Mon 12 | Test run | `cache --split test` for OWL-ViT and the frozen colour variant. Evaluate at IoU 0.25 and 0.5. Draft the figure. |
| Tue 13 | Extras | Blur plus resize. OWLv2 on a 100-chip subset. Extra levels near the transition. Rerun any failed chips. |
| Wed 14 | Checkpoint | MVP complete or cut scope. Go or no-go on the NAIP check. |
| Thu 15 | Analysis | Block bootstrap CIs, break point with CI from paired replicates, pixel quantisation floor, failure montages. NAIP only on go, one day maximum. |
| Fri 16 to Sun 18 | Write-up | 3,000 to 4,000 words. Friday outline and Methods, Saturday Results and Discussion, Sunday polish and limits. |
| Mon 19 | Buffer | Final checks, README, replay test from archives, tag. |
| Tue 20 | Apply | KCL and UCL applications. |

## Compute budget (to be measured on Saturday, not assumed)

Per chip: 10 levels x 2 resize methods = 20 forward passes per model. With 150 dev and 400 test chips
that is about 11,000 passes per model. Run overnight jobs with `PYTORCH_ENABLE_MPS_FALLBACK=1`.
If OWLv2 is too slow, cut it to 100 chips and 5 levels.

## Risks

- Coursework and group-project work are not scheduled here. Tell me the real hours per day.
- Colour thresholds came from Sentinel-2 display colours. On 0.3 m imagery they may behave very differently.
- OWL-ViT may score near zero at every level. The write-up rule for that case is in DECISIONS.md.
