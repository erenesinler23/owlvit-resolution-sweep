# Where does open-vocabulary building detection stop? A resolution sweep from 0.3 m aerial imagery to real Sentinel-2

Lutfi Eren Esinler, University of Nottingham

Draft, 9 October 2026. Every number below comes from `docs/RESULTS.md` and the files in `results/`.

## Abstract

In an earlier study I tried to find buildings in Sentinel-2 crops of seven cities. A hand-written colour rule found some roofs with low precision. OWL-ViT, an open-vocabulary detector, matched nothing at an intersection-over-union (IoU) of 0.25. That left a question open. Does the model fail because it cannot recognise buildings, or because a 10 m pixel leaves nothing to recognise?

I tested this with a controlled sweep. I took 750 aerial chips from SpaceNet 2 (Paris, Khartoum and Las Vegas, 0.3 m per pixel), degraded each to ten ground sampling distances (GSD) between 0.3 m and 10 m, and ran OWL-ViT, OWLv2 and a colour rule. I chose prompts, thresholds and size filters on 150 development chips, froze them, and scored 600 test chips once. Las Vegas was never used for tuning. I then ran the same detectors on real Sentinel-2 crops of the same footprints.

With frozen settings, OWL-ViT reaches 37% to 45% F1 at about 1 m in the three cities and halves that score at 2.2 to 2.6 m. OWLv2 breaks at 2.0 to 3.1 m. At 5 m both score under 1%. I predicted before the Las Vegas run that larger buildings there would push the break to 3.1 m (OWL-ViT) and 3.8 m (OWLv2). It came at 2.3 m and 3.1 m, and a split of 411 chips by building size confirms that the break barely moves when building size nearly doubles. So the limit is better stated in metres, about 2 to 3 m, than in pixels per building. On real Sentinel-2 crops both models score 0.0% with these settings, which reproduces the zero in my earlier paper.

Re-tuning the score threshold and size cap at each resolution recovers part of the loss. On real Sentinel-2, OWLv2 then reaches about 10% F1 and OWL-ViT about 2%. Both stay at or below what the same boxes would score if placed on true building centres, and most of the score comes from dense chips. I read this as weak evidence of finding built-up ground, not of detecting buildings. Simulated degradation overstates the real result, so it works as an upper bound.

