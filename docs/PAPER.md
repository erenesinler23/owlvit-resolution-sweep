# At what resolution does open-vocabulary building detection break down? A sweep from 0.3 m satellite imagery to real Sentinel-2

Lutfi Eren Esinler, University of Nottingham

Repository: https://github.com/erenesinler23/owlvit-resolution-sweep

9 October 2026. Every number below comes from `docs/RESULTS.md` and the files in `results/`.

## Abstract

In an earlier study I tried to find buildings in Sentinel-2 crops of seven cities. A hand-written colour rule found some roofs with low precision. OWL-ViT, an open-vocabulary detector, matched nothing at an intersection-over-union (IoU) of 0.25. That left a question open. Does the model fail because it cannot recognise buildings, or because a 10 m pixel leaves nothing to recognise?

I tested this with a controlled sweep. I took 750 chips from SpaceNet 2 (WorldView-3 satellite images of Paris, Khartoum and Las Vegas, 0.3 m per pixel), degraded each to ten ground sampling distances (GSD) between 0.3 m and 10 m, and ran OWL-ViT, OWLv2 and a colour rule. I chose prompts, thresholds and size filters on 150 development chips, froze them, and scored 600 test chips once. Las Vegas was never used for tuning. I then ran the same detectors on real Sentinel-2 crops of the same footprints.

With frozen settings, OWL-ViT reaches 37% to 45% F1 at about 1 m in the three cities, and its F1 above chance falls to half of that peak at 2.2 to 2.6 m. OWLv2 breaks at 2.0 to 3.1 m. At 5 m both score under 1%. I predicted before the Las Vegas run that larger buildings there would push the break to 3.1 m (OWL-ViT) and 3.8 m (OWLv2). It came at 2.3 m and 3.1 m, and a split of 411 chips by building size moves the break by only 4% to 12% when building size rises by a factor of 1.8. In these data the break tracks metres better than pixels per building, and it sits at about 2 to 3 m. On real Sentinel-2 crops both models score 0.0% with these settings, which reproduces the zero in my earlier paper.

Re-tuning the score threshold and size cap at each resolution recovers part of the loss. On real Sentinel-2, OWLv2 then reaches about 10% F1 and OWL-ViT about 2%. Most of that score comes from dense chips. OWLv2 stays below what its own boxes would score if placed on true building centres, while OWL-ViT's 2% sits about one point above its equivalent. The data cannot separate finding built-up ground from finding single buildings. Simulated 10 m images are a poor forecast of the real result: area averaging overstated it for both models, and blur and resize missed the OWLv2 score by 3 to 5 points in either direction.

## 1. Introduction

Open-vocabulary detectors take a text prompt and return boxes. OWL-ViT (Minderer et al., 2022) and OWLv2 (Minderer et al., 2023) pair a vision transformer with a text encoder trained in the CLIP style (Radford et al., 2021). They need no task-specific training, so they are an obvious thing to try on satellite imagery, where labelled data is scarce and the object list changes from one project to the next.

My earlier study tried exactly that on Sentinel-2 crops of seven cities (Esinler, 2026). Sentinel-2 has a ground sampling distance (GSD) of 10 m in its visible bands (Drusch et al., 2012). I compared a colour rule, OWL-ViT and a hybrid of the two. The colour rule reached precision between 4.8% and 24.9%. OWL-ViT and the hybrid matched nothing at an intersection-over-union (IoU) of 0.25.

I reported that result and stopped. The trouble is that it mixes two causes. The model may be poorly suited to overhead imagery. Or a house at 10 m may cover one or two pixels, in which case no detector could find it. A single resolution cannot separate the two.

This paper separates them with a controlled sweep and then checks the answer on real Sentinel-2 data. The same chips appear at ten resolutions, so the only thing that changes is the pixel size. I ask five questions.

1. At what GSD does OWL-ViT stop finding buildings, and how sharp is the drop?
2. Does the answer hold in a city the settings were never tuned on?
3. How much do the answers depend on how the images are degraded, and do real coarser sensors behave like the simulation?
4. What explains the scores that survive at coarse resolution once the detectors are re-tuned?
5. Does real Sentinel-2 imagery of the same places give the same answer as the simulated 10 m level?

I treat this as an exploratory study. I report intervals wherever I can, and I list the places where the evidence is thin.

## 2. Data

### 2.1 SpaceNet 2

I used the SpaceNet 2 building detection data (Van Etten et al., 2018), released under CC BY-SA 4.0. Each chip is 650 by 650 pixels of pan-sharpened RGB at 0.3 m, so it covers about 195 m on a side. Building footprints come as polygons. I used three cities. Paris (1,148 chips) and Khartoum (1,012 chips) carry the tuning and the main results. Las Vegas serves as a third city that no setting was tuned on. I downloaded 300 random Las Vegas chips from the public per-file bucket, because the full Las Vegas archive is about 23 GB, and used the first 200 in sorted order.

I read the files on disk before trusting any documentation. The imagery is 16-bit, georeferenced in EPSG:4326, with bands in red, green, blue order. I checked the band order by overlaying labels on a rendered chip. The red roofs came out red and the footprints sat on the buildings.

### 2.2 Preparing the images

The cities differ a lot in brightness (median red value about 190 in Paris and about 520 in Khartoum). I converted each city to 8-bit with one scale constant per city: 700 for Paris, 950 for Khartoum and 1,500 for Las Vegas. I set each from the 99.5th percentile of all band values on 150 random chips of that city, rounded to 50. I used the same constant for all three bands so that colour ratios survive, because the colour rule depends on them. I looked only at pixel values when choosing these constants, never at labels.

About 8% of Paris pixels and 17% of Khartoum pixels are zero. They sit on chip borders where the source mosaic ends. I left them as black. In the test sample 51% of the Paris chips and 8% of the Khartoum chips contain no labelled building. I kept them, because a detector that fires on empty ground should lose precision.

### 2.3 Development and test split

Neighbouring chips are not independent, since the same street appears in several. I split Paris and Khartoum on spatial blocks to stop information leaking between development and test. I assigned each chip to a 0.01 degree grid cell (roughly one kilometre) and shuffled whole cells into development and test pools of about 30% and 70% of the chips. That gave 164 blocks, with 695 development chips and 1,465 test chips in the pools.

I then drew 150 chips for development (40 blocks, 80 from Paris and 70 from Khartoum) and 400 for test (104 blocks, 220 and 180). The test sample holds 7,320 reference buildings: 2,879 in Paris and 4,441 in Khartoum. The median reference footprint has an equivalent side (the square root of its area) of 10.9 m, and 19% have a side under 5 m. The development sample holds 3,202 references. All Las Vegas chips are test chips. The 200 of them hold 5,977 reference buildings with a median equivalent side of 13.9 m, and 7% of the chips contain no building.

I fixed all settings on the development chips and wrote them to a config file before I scored a single test chip. I built the test cache earlier to save time, but that step stores raw model output only and never compares it with labels.

### 2.4 Real Sentinel-2 imagery

SpaceNet chips come with geographic bounds, so I can fetch the Sentinel-2 view of exactly the same ground. I used Sentinel-2 level 2A scenes from the public AWS archive (red, green and blue bands at 10 m), found through the Earth Search catalogue. For each chip I took a scene from 2018 to 2021 with under 8% scene cloud cover, and I rejected a scene for that chip if the scene classification layer showed cloud, cloud shadow, saturation or missing data anywhere inside the footprint. I resampled each band onto a 20 by 20 pixel grid over the chip footprint with bilinear interpolation. The simulated 10 m level has the same 20 by 20 shape, so the two are directly comparable. All 400 Paris and Khartoum test chips and 200 Las Vegas chips found a clean scene. The scenes are not contemporaneous with the SpaceNet images, which I come back to in the limitations.

I converted the digital numbers to 8-bit with one constant per city, from the same percentile recipe as above, and upsampled to 650 pixels with bicubic interpolation like every simulated level.

## 3. Method

### 3.1 Degrading the images

I produced ten levels: 0.3, 0.45, 0.6, 0.9, 1.2, 1.8, 2.4, 3.6, 5.0 and 10 m. A chip at level g shrinks by a factor of g/0.3. The 10 m level comes out at 20 pixels, which is an effective 9.75 m, and I call it 10 m throughout. I used two ways of shrinking.

- Area averaging. Each output pixel is the mean of the source pixels it covers (OpenCV `INTER_AREA`). This mimics a sensor that integrates light over a larger footprint.
- Blur and resize. A Gaussian blur with sigma equal to half the shrink factor, then bilinear resampling. This is the shortcut many pipelines use.

After shrinking, I scaled each image back up to 650 pixels with bicubic interpolation. This keeps boxes in the original coordinate frame and gives every detector the same input size. The upsampling adds no information. A 10 m image still holds 20 by 20 real pixels.

### 3.2 Detectors

All three detectors share one interface. Each takes an RGB array and returns boxes.

Colour rule. A pixel counts as roof if red is above a threshold and exceeds green and blue each by a margin. I label 4-connected groups of such pixels, and each group becomes one box. My earlier paper used red above 120 with both margins above 25. I kept those values as the baseline and added a development grid of red thresholds (100, 120, 140) and margins (15, 25, 40).

OWL-ViT. I used `google/owlvit-base-patch32` at the same pinned revision as my earlier paper. The model resizes the image to 768 pixels and returns 576 candidate boxes with a score for each text prompt. I cache the score for every box and every prompt before any thresholding, so tuning costs no further forward passes. I checked my wrapper against the saved raw output of the earlier paper on its Ankara archive. Boxes matched within 0.01 pixels and scores matched exactly.

OWLv2. I used `google/owlv2-base-patch16-ensemble` (revision `cfd3195b`) as a sanity check. It works at 960 pixels with 3,600 candidate boxes. I treat it as a second opinion, not a second experiment.

I tried six prompts: the two colour-based prompts from my earlier paper, "a satellite image of a building", "an aerial view of a building", "a rooftop seen from above", and "a house". A box keeps its best score over the prompts in the chosen set. Both models together hold about 308 million parameters, and I trained none of them.

### 3.3 Matching and metrics

I used the same matcher as my earlier paper. Predicted boxes and reference polygons are paired one to one. A pair counts only if the IoU reaches the threshold. Among all valid assignments the matcher first maximises the number of pairs and then the total IoU, using the Hungarian method (Kuhn, 1955). I report precision (matches over predictions), recall (matches over references) and F1 at IoU 0.25 and 0.5. I checked this matcher against the seven archives from the earlier paper and it reproduced their counts.

Two details matter when reading the numbers. First, I compare a box with a polygon, so a rotated building can never reach an IoU of 1, even with a perfect box. Second, I pool counts over all test chips before computing a ratio. A chip with no buildings therefore adds only false positives.

### 3.4 Choosing settings on the development split

For each model I searched over prompt set, score threshold and maximum box size. A size filter drops boxes wider or taller than the cap, which removes the large boxes OWL-ViT produces around whole blocks. For the colour rule I searched the red threshold and margin, with the same size caps. I chose the setting with the highest mean F1 over all ten resolutions at IoU 0.25, scored on area-averaged images. I did not re-tune per resolution for the main results, because someone applying a detector to unknown imagery cannot.

I widened the grid twice, because the first two passes put the best setting on the edge of the grid. The final choices are:

- OWL-ViT: the two colour prompts from my earlier paper, score threshold 0.002, size cap 30 m (mean dev F1 26.0).
- OWLv2: "a house", threshold 0.02, size cap 30 m (mean dev F1 26.4).
- Colour rule: red above 120, margins above 15 (mean dev F1 5.9, against 2.5 for the thresholds in my earlier paper).

The top few settings sit within about one F1 point of each other, so I treat the choice as a near tie and not as a discovery. The prompts from my earlier paper came out best for OWL-ViT once the threshold was low enough, even though I had assumed the generic ones would win. A first pass with a coarser threshold grid pointed the other way, which is a reminder of how much a grid edge can mislead.

I also froze a per-resolution setting (the best development setting at each level, chosen on the Paris and Khartoum development chips). I call these the resolution-specific settings. They were selected on development data, and I report them as what such tuning achieves on the test chips.

### 3.5 Uncertainty, chance level and the break point

I resampled whole spatial blocks with replacement, 2,000 times, and recomputed the pooled metrics (Efron, 1979; Cameron et al., 2008). I applied the same resample to every resolution, so each replicate gives a full curve and I can compute a break point per replicate.

A detector can score above zero by luck. To find that floor I took each set of predicted boxes, kept its count and sizes, and moved every box to a random position on its chip, 20 times. The F1 this earns is the chance level at that resolution. It runs from 8% to 15% for the two OWL models at fine resolution, because their boxes are large.

I define the break point as the first GSD, coarser than the best one, where F1 minus chance falls to half of its value at the best resolution. I interpolate in log GSD. I first defined it against the finest level, but the curves are not monotone, so I switched to the peak before I scored the test split. I give the finest-level version too. For the pooled OWL-ViT and OWLv2 curves the two versions differ by at most 0.2 m. They differ more where a curve is flat or noisy: by 0.6 m for OWLv2 in Las Vegas and by 2.0 m for the colour rule.

### 3.6 An oracle-location baseline

The chance level above moves boxes to random positions on the chip. That is a fair floor for a detector that has no idea where buildings are, but buildings cluster, and a detector that only finds built-up areas gets matches the uniform baseline never would. I therefore added a second, more generous baseline. For each set of predicted boxes I keep the count and the sizes, and I place each box centre on the centroid of a randomly chosen true building (without replacement while buildings last). I average over 10 random draws per chip, level and setting.

This oracle knows where buildings are and knows nothing about what size they are. Its F1 is what a detector would score if it located buildings perfectly but drew boxes of its own, size-agnostic shapes. This makes it a reference point and not a ceiling. A detector far below it may be misplacing boxes or drawing poorly shaped ones, and I cannot tell which. A detector near it gets about as much from its box count and sizes as perfect location would give. A detector above it also uses the image to fit box size and shape. I report the detector minus the oracle with a block-bootstrap interval, resampling the same blocks for both.

### 3.7 Held-out city and real Sentinel-2 protocol

Las Vegas ran through exactly the same code and the same frozen settings as the test chips of the other two cities. Before any Las Vegas image went through a model, I wrote down a prediction for its break point in the project log. Las Vegas buildings are larger than the other two cities (median equivalent side 13.9 m against 11.1 m in Paris and 10.6 m in Khartoum, measured on all 200 chips after the run; I had measured 14.4 m on the first 44 chips when I made the prediction), so if the break depends on pixels per building, it should come at a coarser GSD. Scaling the pooled Paris and Khartoum break points by building size predicts about 3.1 m for OWL-ViT and 3.8 m for OWLv2 with area averaging at IoU 0.25. With the final 13.9 m the predictions would be 3.0 m and 3.7 m, so the final value does not change the test.

For the Sentinel-2 comparison I ran the frozen detectors, unchanged, on the real 20 by 20 crops and scored them against the SpaceNet footprints. I compare the real score with the simulated 10 m score (both degradation methods) on the same chips with a paired block bootstrap. I do this for the frozen setting and for the setting tuned for 10 m on the development chips.

## 4. Results

All numbers come from held-out chips, scored once. F1 is in percent. Intervals are 95% block-bootstrap intervals. Sections 4.1 to 4.3 and 4.6 to 4.8 pool the 400 Paris and Khartoum test chips. Section 4.4 splits them by city and adds Las Vegas. The full tables are in `docs/RESULTS.md`, and Figure 1 shows the curves.

![Figure 1. F1 against ground sampling distance at IoU 0.25 with area averaging, 400 held-out Paris and Khartoum chips. Solid blue: the setting frozen on development data, with its 95% interval. Dashed red: the setting re-tuned at each resolution. Dotted grey: chance (boxes moved to random positions). Dash-dot green: oracle locations (the frozen boxes moved onto random true building centres).](figures/fig1_f1_vs_gsd.png)

### 4.1 The frozen-setting curves

With one setting per detector, OWL-ViT peaks at 39.9% F1 [36.9, 42.4] at 0.9 m (IoU 0.25, area averaging). It stays near that value to 1.2 m (39.8%), drops to 33.8% at 1.8 m, to 16.5% at 2.4 m, and to 1.0% [0.6, 1.5] at 3.6 m. At 5 m and 10 m it scores 0.0%. OWLv2 follows the same shape with a later fall: 39.0% at 0.9 m, 28.6% at 2.4 m, 5.0% [2.8, 7.2] at 3.6 m and 0.2% at 5 m.

The colour rule has a flat, low curve. It scores 3.7% at 0.3 m, peaks at 6.4% [4.1, 8.5] at 0.9 m, and still holds 4.4% [2.7, 6.0] at 3.6 m and 3.0% [1.9, 4.1] at 5 m. At 10 m it falls to 0.4%. Its chance level is under 1% at every resolution, so its small scores are real, if modest. Under the frozen settings it beats OWL-ViT from 3.6 m on, and the intervals at 3.6 m do not overlap (4.4 against 1.0).

At IoU 0.5 every score roughly halves. OWL-ViT peaks at 17.9% and OWLv2 at 16.5%, and the colour rule at 2.0%. The shapes stay the same. The strict threshold mostly measures how well a box fits a rotated polygon, and I return to that in the limitations.

### 4.2 The break point

The break point is where F1 above chance falls to half of its peak value. For the frozen settings on area-averaged images at IoU 0.25:

- OWL-ViT: 2.34 m [2.19, 2.47]
- OWLv2: 2.87 m [2.59, 3.02]
- Colour rule: 4.77 m [4.13, 5.63]

The colour rule's number deserves care. Its peak is low and its curve is flat, so half of the peak is a small absolute score, and the break point moves a lot when the curve wobbles. It says that the rule degrades more slowly, not that it works better.

At IoU 0.5 the OWL-ViT break moves to 2.18 m [2.06, 2.28]. A median building has an equivalent side of 10.9 m, so the OWL-ViT break at IoU 0.25 corresponds to a median building 4.7 pixels across. The 19% of buildings narrower than 5 m are about two pixels wide or less at that break. I first thought the pixel count would be the better way to read the result, because it would carry over to other sensors. Section 4.4 tests that, and it does not hold.

### 4.3 How the images are degraded

Blur and resize is harsher than area averaging. The OWL-ViT break at IoU 0.25 drops from 2.34 m to 1.97 m [1.85, 2.03], a gap of 0.4 m. At 2.4 m OWL-ViT scores 16.5% with area averaging and 5.2% with blur and resize. The OWLv2 break drops from 2.87 m to 1.90 m. The colour rule's break moves from 4.77 m to 3.39 m.

So the break point has a spread of 0.4 to 1.0 m for the OWL models (1.4 m for the colour rule) from this choice alone. A paper that reports one break point from one degradation method hides that spread. Section 4.8 compares both methods with real Sentinel-2 data, and neither forecasts it reliably.

### 4.4 Three cities, and a prediction that failed

The pooled curves hide differences between cities. Table 1 gives the frozen-setting results for each city (IoU 0.25, area averaging). Figure 2 shows the curves.

Table 1. Frozen settings, IoU 0.25, area averaging. Break points in metres with 95% intervals.

| City | Chips | Median building side | OWL-ViT peak F1 | OWL-ViT break | OWLv2 peak F1 | OWLv2 break | Colour rule peak F1 |
|---|---|---|---|---|---|---|---|
| Paris | 220 | 11.1 m | 45.2 | 2.56 [2.34, 2.70] | 46.5 | 3.14 [3.04, 3.23] | 13.0 |
| Khartoum | 180 | 10.6 m | 36.8 | 2.16 [1.95, 2.30] | 36.0 | 2.04 [1.71, 2.28] | 1.4 |
| Las Vegas | 200 | 13.9 m | 43.7 | 2.31 [2.16, 2.53] | 43.5 | 3.10 [2.90, 3.32] | 1.6 |

![Figure 2. Left and middle: frozen-setting F1 against GSD in each city for OWL-ViT and OWLv2 (IoU 0.25, area averaging). Right: F1 at 10 m for the re-tuned settings on simulated 10 m images (two degradation methods), on real Sentinel-2 crops of the same footprints, and for the oracle-location baseline on the real crops.](figures/fig2_cities_and_sentinel2.png)

OWL-ViT breaks between 2.2 and 2.6 m in all three cities. OWLv2 breaks between 2.0 and 3.1 m, and Khartoum breaks about a metre earlier than Paris and Las Vegas. The colour rule works in Paris (13.0% at its peak) and barely at all in Khartoum or Las Vegas (about 1.5%). Its pooled 6.4% mostly reflects Paris, where red tile roofs are common. Peak F1 is lower in Khartoum for every detector, and for the OWL models by 6.9 to 10.5 points.

The prediction. Las Vegas buildings are larger (median side 13.9 m against 10.6 to 11.1 m). In Section 3.7 I wrote down, before any Las Vegas image went through a model, that if the break depends on pixels per building it should come at 3.1 m for OWL-ViT and 3.8 m for OWLv2. It came at 2.31 m [2.16, 2.53] and 3.10 m [2.90, 3.32]. Both predictions fall outside their intervals and both were too high. The Las Vegas curves look like the Paris curves, not like a version stretched toward coarser resolution.

Splitting by building size. To test the idea within the data, I pooled all 600 test chips, kept the 411 with at least five reference buildings, and cut them into three groups of 137 by the median building side in each chip (8.7, 12.3 and 16.0 m). Table 2 gives the break points.

Table 2. Break points by building size, frozen settings, IoU 0.25, area averaging.

| Chip group (median side) | OWL-ViT break | OWLv2 break | OWL-ViT F1 at 3.6 m | OWLv2 F1 at 3.6 m |
|---|---|---|---|---|
| Small (8.7 m) | 2.15 [1.97, 2.31] | 2.87 [2.63, 3.05] | 1.0 | 4.8 |
| Medium (12.3 m) | 2.42 [2.23, 2.61] | 3.02 [2.84, 3.17] | 2.2 | 8.3 |
| Large (16.0 m) | 2.41 [2.21, 2.63] | 2.99 [2.74, 3.24] | 3.8 | 9.7 |

Building size rises by a factor of 1.8 from the small to the large group, and the OWL-ViT break moves by 12% and the OWLv2 break by 4%. In pixels, the OWL-ViT break falls at 4.0, 5.1 and 6.6 building widths. With blur and resize the breaks are 1.76, 1.97 and 1.93 m for OWL-ViT. Larger buildings do leave a longer tail (F1 at 3.6 m is two to four times higher in the large group), but the point where F1 halves is close to fixed in metres.

So the "five pixels per building" reading in Section 4.2 does not hold up. It came from two cities with similar buildings. A better summary is that, with settings frozen on fine imagery, the break stays between 2 and 3 m across the range of building sizes I could test (a factor of 1.8 in median size). I do not know why. One possibility is that the models see every chip at a fixed input size (768 pixels for OWL-ViT, 960 for OWLv2), so a degraded image always arrives upsampled to the same grid, and what is lost at a given GSD in metres is the same for every building. I have not tested that.

### 4.5 A native 1.2 m image against the simulated one

SpaceNet 2 ships a separate 8-band multispectral product at 1.2 m. I took its red, green and blue bands, upsampled them the same way, and ran the frozen detectors on the same 400 chips. The result splits by model.

For OWL-ViT the native image and the simulation agree: 40.8% against 39.8% for area averaging (difference 1.0 [-0.2, 2.0]). For OWLv2 the native image scores higher by 3.3 [1.6, 5.4] points against area averaging and 7.1 [4.8, 10.0] against blur and resize. The colour rule gains 2.4 [1.4, 3.2] and 2.9 points. I read this as the simulation being slightly pessimistic for OWLv2 and the colour rule, and about right for OWL-ViT. It also suggests that blur and resize understates what a real sensor delivers, because a real sensor at 1.2 m has its own sharpening and radiometry that I do not model.

This check has a weak point. The multispectral product is processed differently from the pan-sharpened one, and I assumed the WorldView-3 band order to pick red, green and blue. Both could shift the numbers by a small amount.

### 4.6 Tuning for each resolution

The frozen settings are the fair test of a detector one cannot tune. A user with some labelled chips at the target resolution could tune. To measure that I picked the best development setting at each level and scored it on the test chips. These resolution-specific settings were selected on development data. They show what threshold and size tuning can buy when labelled chips exist at the target resolution, and they are not a fair estimate for unseen imagery.

It buys a lot at coarse resolution (IoU 0.25, area averaging):

| GSD (m) | OWL-ViT frozen | OWL-ViT per level | OWLv2 per level | Chance for OWLv2 |
|---|---|---|---|---|
| 2.4 | 16.5 | 30.7 | 36.4 | 9.0 |
| 3.6 | 1.0 | 22.0 | 32.4 | 9.4 |
| 5.0 | 0.0 | 10.2 | 27.3 | 6.9 |
| 10.0 | 0.0 | 2.8 | 16.3 | 5.4 |

The per-level OWLv2 curve barely falls until 3.6 m, and still scores 16.3% at 10 m with a chance level of 5.4%. The break points move to 3.46 m [3.04, 3.77] for OWL-ViT and 7.24 m [6.51, 8.19] for OWLv2. With blur and resize they sit at 2.42 m and 5.15 m, and the OWLv2 interval there is wide [2.26, 5.84], which tells me the curve is not well pinned down. In Paris and Khartoum the colour rule gains nothing from per-level tuning, because its red threshold hardly changes the result.

Las Vegas, which no setting was tuned on, behaves the same way. With the per-level settings OWL-ViT scores 23.9% at 3.6 m, 16.9% at 5 m and 3.3% at 10 m (chance 10.2%, 6.9% and 1.3%). OWLv2 scores 32.7%, 25.9% and 20.7% (chance 14.3%, 9.9% and 8.1%). The per-level break points are 3.14 m [2.86, 3.52] for OWL-ViT and 3.42 m [3.03, 4.05] for OWLv2, which is closer to the Paris and Khartoum OWL-ViT figure than to the pooled OWLv2 figure of 7.2 m. The Las Vegas OWLv2 peak (52.2% at 0.6 m) is noisy, so I would not read much into the exact break.

I did not expect this. My first reading of the frozen curves was that detection ends near 3 m. The per-level curves say that the models carry signal well past that point if one lowers the score threshold to 0.001 and lets the size cap grow. Sections 4.7 and 4.8 look at what that signal is.

### 4.7 What the coarse scores are

The matched buildings. For the per-level settings I took the matched reference buildings at 5 m and 10 m and compared them with all references (IoU 0.25, area averaging, 400 chips).

| Setting | Predictions | Matches | Median matched building (m²) | Share of matched above 1,000 m² |
|---|---|---|---|---|
| All references | | | 118 | 4% |
| OWLv2, 1.2 m, frozen | 5,004 | 2,330 | 169 | 1% |
| OWLv2, 5 m, per level | 5,087 | 1,691 | 175 | 1% |
| OWLv2, 10 m, per level | 3,086 | 848 | 314 | 13% |
| OWL-ViT, 5 m, per level | 1,308 | 442 | 342 | 12% |
| OWL-ViT, 10 m, per level | 1,546 | 125 | 1,124 | 54% |

At 5 m OWLv2 still matches buildings of ordinary size, with precision 33% and recall 23%. At 10 m it skews toward larger buildings and recall falls to 12%. OWL-ViT at 10 m is a different case. Its median prediction covers 15,210 m², about 40% of a 38,025 m² chip, and it matches only the largest buildings (median 1,124 m²). I do not count that as detection of individual buildings.

The oracle-location baseline. The oracle of Section 3.6 puts each detector's own boxes (same count, same sizes) on random true building centres. Detector minus oracle, area averaging, IoU 0.25:

| Detector, setting | 0.9 m | 1.8 m | 2.4 m | 3.6 m | 5 m | 10 m |
|---|---|---|---|---|---|---|
| OWL-ViT, frozen | -5.3 [-6.4, -4.2] | -3.1 [-4.5, -1.6] | +1.6 [0.8, 2.4] | +0.1 [-0.1, 0.4] | 0.0 | 0.0 |
| OWL-ViT, per level | -4.4 [-6.0, -2.7] | -7.2 [-8.2, -6.2] | -0.6 [-1.7, 0.6] | -0.8 [-1.8, 0.2] | +1.9 [1.2, 2.7] | +1.1 [0.7, 1.6] |
| OWLv2, frozen | -1.7 [-3.0, -0.2] | -0.1 [-1.3, 1.2] | +1.7 [0.4, 2.8] | +1.0 [0.4, 1.6] | +0.1 | 0.0 |
| OWLv2, per level | -1.3 [-2.8, 0.5] | -4.6 [-5.9, -3.1] | -5.1 [-6.4, -3.5] | -6.8 [-8.3, -5.4] | -7.3 [-9.3, -5.3] | -1.8 [-3.0, -0.6] |
| Colour rule | -1.2 [-1.9, -0.5] | -1.0 [-1.7, -0.4] | -0.8 [-1.2, -0.3] | -0.7 [-1.2, -0.3] | -0.7 [-1.2, -0.2] | -0.3 [-0.5, -0.1] |

At fine resolution every detector sits below its oracle, by 1 to 7 points (a few intervals include zero). One possible reason is that the detectors do not place every box on a building, and the oracle does. The comparison does not show that, since the gap could also come from box shape. For the frozen settings the gap closes from 2.4 m on. For the per-level settings it closes for OWL-ViT but not for OWLv2, which stays 5 to 7 points below through 5 m. In the area-averaged table no detector exceeds its oracle by more than 2 points anywhere, and the two largest positive gaps are +1.9 and +1.7. With blur and resize the largest positive gap is +2.8 (OWLv2, frozen, 1.8 m). The OWLv2 per-level setting at 5 m scores 27.3%, which is 7 points below the 34.6% that its boxes would earn on perfectly placed centres, and at 10 m it scores 16.3% against 18.1%. With blur and resize and at IoU 0.5 the pattern is the same (OWL-ViT per level at 3.6 m with blur and resize: +2.3 [1.5, 3.1]; OWLv2 per level at 5 m and IoU 0.5: -7.3 [-8.5, -6.1]).

Dependence on building density. I split the chips into thirds by their number of reference buildings (0 to 1, 1 to 23, and 23 to 175 buildings; 134, 133 and 133 chips). At 10 m the OWLv2 per-level setting has about 1% precision in the sparse third, 21% in the middle third and 32% in the dense third. At 0.9 m the same detector reaches about 1%, 30% and 49%. In the dense third at 10 m its F1 is 16% against an oracle of 17%.

Putting these together: at 5 to 10 m the surviving F1 comes mostly from dense chips, and it is no higher than boxes of the same count and size would earn on perfectly located buildings. This is what a detector would produce if it found built-up areas and drew building-sized boxes in them. It is also what one would see if it found some individual buildings. I cannot separate the two with these data. What I can say is that the extra score from re-tuning is a score for putting building-sized boxes on built-up ground. Boxes of the same count and size, placed on true building centres, would add 0 to 7 points to the re-tuned OWLv2 scores from 2.4 m on in Paris and Khartoum. OWL-ViT's re-tuned score is already at that level or slightly above it. The oracle is a reference, not an upper limit. The oracle's own F1 for OWLv2 falls from 41.0% at 0.3 m to 34.6% at 5 m and 18.1% at 10 m. So most of the fall in F1 happens in the boxes themselves, in their number and size relative to the buildings, and less in where they land.

Las Vegas is less clean. There the frozen settings again sit within 2.4 points of the oracle from 2.4 m on (OWL-ViT +0.2 [-1.6, 1.8] at 2.4 m and +0.5 [0.1, 1.0] at 3.6 m). The re-tuned settings stay below it: OWL-ViT by 3.9 and 4.0 points at 3.6 m and 5 m (and level with it at 10 m, +0.4 [-0.1, 0.9]), and OWLv2 by 8.1, 12.1 and 5.1 points at 3.6, 5 and 10 m. So in Las Vegas the re-tuned detectors fall further below the oracle. Box location is one possible explanation, and I did not test it directly. This is another place where the three cities do not give one story.

### 4.8 Real Sentinel-2 imagery

This is the check that connects the sweep to the question that started it. I ran the detectors on real Sentinel-2 crops of the 400 Paris and Khartoum test chips and of the 200 Las Vegas chips, and scored them against the same labels (IoU 0.25).

With the frozen settings, OWL-ViT and OWLv2 both score 0.0% on real Sentinel-2 in every city. No box clears the score threshold on any chip. The simulated 10 m level gives 0.0% too. The colour rule is different. On Paris and Khartoum it scores 1.5% [1.1, 1.8] on real data against 0.4% and 0.2% in the two simulations (difference +1.1 [0.7, 1.5] and +1.3 [1.0, 1.7]). On Las Vegas it scores 5.0% [4.0, 6.3] against 0.3% and 0.1% (difference +4.7 [3.7, 5.9]). The oracle on the real crops scores 2.9% and 6.1%, so the colour rule is again no better than boxes of its own sizes put on true buildings. I did not test why real crops suit the rule better. The Sentinel-2 red, green and blue bands differ in spectral response from the WorldView-3 sensor, and the rule depends on band ratios.

With the settings tuned for 10 m on the development chips, real Sentinel-2 gives:

| City | Detector | Real S2 F1 [95% CI] | Simulated, area | Simulated, blur and resize | Oracle on real crops |
|---|---|---|---|---|---|
| Paris + Khartoum | OWL-ViT | 1.7 [1.2, 2.2] | 2.8 | 1.7 | 1.2 |
| Paris + Khartoum | OWLv2 | 9.8 [8.1, 11.5] | 16.3 | 7.2 | 13.8 |
| Las Vegas | OWL-ViT | 2.4 [1.7, 3.2] | 3.3 | 2.6 | 1.7 |
| Las Vegas | OWLv2 | 10.4 [8.8, 12.3] | 20.7 | 15.0 | 12.1 |

The real OWLv2 score is close to 10% in both groups of cities. The simulations are less consistent. In Paris and Khartoum the real score sits between the two simulations: 6.5 points [4.8, 8.2] below the area-averaged one and 2.6 points [1.2, 4.1] above the blur-and-resize one. In Las Vegas it sits below both (10.3 points [8.8, 11.8] below area averaging and 4.6 points [3.1, 6.2] below blur and resize). For OWL-ViT the real score matches blur and resize in both groups (differences 0.0 [-0.5, 0.4] and -0.2 [-0.8, 0.3]) and falls below the area-averaged one. At IoU 0.5 the OWLv2 score in Paris and Khartoum is 1.8% against simulated values of 4.3% and 2.0%. So area averaging overstated real Sentinel-2 performance for both models. Blur and resize came closer for OWL-ViT, but for OWLv2 it erred in opposite directions (too low in Paris and Khartoum, too high in Las Vegas). The colour rule is the exception: its real score is higher than both simulations. I would not trust a simulated 10 m score as a forecast for any of the three detectors.

Three things follow. First, the zero in my earlier paper reproduces under another setup: real Sentinel-2 data, SpaceNet labels, three different cities, and settings chosen for finer imagery. That makes it less likely to come from my earlier choices of prompts, cities or labels, but it does not rule out every such effect. Second, re-tuning rescues a small amount: OWLv2 reaches about 10% on real Sentinel-2 and OWL-ViT about 2%. Third, the better figure sits below the 12% to 14% that an oracle-located baseline reaches with the same OWLv2 boxes, and OWL-ViT's 2% is about one point above its own oracle. I would not call either building detection in any practical sense.

### 4.9 Precision at the finest resolutions

OWL-ViT scores a little lower at 0.3 m (36.5%) than at 0.9 m (39.9%), though the intervals overlap. The counts show where it goes. At 0.3 m the model fires 16,511 boxes and precision is 26.3%. At 0.9 m it fires 10,584 and precision is 33.7%, and at 1.2 m it fires 8,067 and precision is 38.0%. My guess is that sharper images give the model more roof details and more things to split into parts, such as chimneys and extensions. I have not tested that, so I treat it as a hypothesis.

## 5. Discussion

The earlier result holds on real data. My Sentinel-2 study found no OWL-ViT matches at 10 m. Here, with the settings frozen on fine imagery, both OWL models score zero on real Sentinel-2 crops in three cities, with labels that come from a different source. The earlier zero therefore reproduces under another setup, which makes it less likely to be an artefact of that study's prompts, cities or labels. It does not rule those out. A detector set up the usual way has stopped working well before 10 m.

Where the break falls. With frozen settings, OWL-ViT halves its F1 above chance at 2.2 to 2.6 m in each of three cities, and OWLv2 at 2.0 to 3.1 m. At 3.6 m both score 11.2% or less in every city, and at 5 m both are below 1%. Between blur and resize and area averaging the break moves by 0.4 to 1.0 m for these two models, and between cities by up to 1.1 m for OWLv2. A fair summary is "about 2 to 3 m under these settings", not a single number. Half of the peak is a convention I chose. It does not mark the point where the detector can no longer find any building.

In these data the break tracks metres more than pixels per building. I started this study with a tidy idea: the break comes when a typical building is about five pixels wide. The pooled Paris and Khartoum numbers fit it (4.7 pixels). My written prediction for Las Vegas, where buildings are larger, failed, and the split by building size agrees with the failure. Buildings that are 1.8 times larger move the break by 4% to 12%. Anyone carrying the result to another sensor should be wary of a rule stated in pixels. A rule stated in metres fits these three cities, but three cities and one range of building sizes do not make it general, and the limitations below apply.

What re-tuning buys. If a user can tune the score threshold and size cap at each resolution, the break moves to 3.5 m (Paris and Khartoum pooled) or 3.1 m (Las Vegas) for OWL-ViT, and to 7.2 m (pooled) or 3.4 m (Las Vegas) for OWLv2. Both keep a few percent to a few tens of percent beyond it. These are real gains, but they look different from the fine-resolution scores. The retained F1 comes mostly from dense chips (Section 4.7), it is close to what the same boxes would score if placed on true building centres, and on real Sentinel-2 it comes to about 10% for OWLv2 and 2% for OWL-ViT. I would not build a mapping product on that. I would also not conclude that nothing is there. A score of 10% against a uniform chance level of 5% to 8% on the simulated 10 m images means that OWLv2 has some information about where built-up ground is at 10 m. The models can plausibly be used as a weak screening tool at that resolution. They cannot count or outline buildings.

Simulation as a forecast. The simulated 10 m level agrees with the real one on the headline (zero for frozen settings). It does not agree on the re-tuned scores. Area averaging overstates the real OWLv2 score by 6 to 10 points. Blur and resize is 2.6 points too low in Paris and Khartoum and 4.6 points too high in Las Vegas. At 1.2 m, a native image scored as well as or better than the simulation. The colour rule did better on real data than on either simulation. So simulated degradation did not give a consistent forecast for any of these detectors.

The colour rule. It scores low everywhere, so it should not be read as competitive with OWL-ViT where OWL-ViT works. It does better on real Sentinel-2 than on the simulation (5.0% against 0.3% in Las Vegas), but the oracle baseline scores higher still, so one possible reading is that the rule finds bare or reddish ground and draws blobs on it. Its strength in Paris and weakness in Khartoum and Las Vegas show how much it depends on local roof colour.

What I would do differently. I would add a baseline that separates "finds built-up ground" from "finds buildings" (for example, boxes placed at random inside a building mask at the detector's sizes against boxes placed at true centres). I would use several scenes per city for Sentinel-2, with dates close to the SpaceNet acquisition. And I would test at least one detector that was trained on overhead imagery, because OWL-ViT and OWLv2 were not.

## 6. Limitations

- Three cities, one label source. Paris, Khartoum and Las Vegas cover three styles of building and three climates. They are not a sample of the world. The intervals resample blocks, not cities, so they do not capture the variation between cities. Section 4.4 shows that variation is large.
- Sentinel-2 comparison. The real crops come from one scene each for Paris and Khartoum and from up to three scenes for Las Vegas, all dated 2020 to 2021, while the labels describe the WorldView-3 images from about 2015 and 2016. Buildings built or demolished in between count as errors, and the Paris scene was taken in February under winter light. The check covers 10 m only, and the digital numbers go through a per-city scale constant that I chose by the same recipe as the SpaceNet data, not one tuned for Sentinel-2.
- Labels and matching. Footprints are polygons and the detectors return axis-aligned boxes. This caps IoU and hits rotated buildings and tight clusters hardest. The labels also include partial buildings at chip borders and some black border areas. Both limit absolute scores at all resolutions.
- Tuning on area averaging only. I tuned on area-averaged development images and applied the settings to blur and resize. The blur results may therefore be unfairly pessimistic.
- Near ties. The top development settings sit within about one F1 point. A different tie-break would give slightly different frozen curves.
- Baselines. The uniform chance baseline ignores clustering and so understates chance on dense scenes. The oracle baseline knows true building centres but draws boxes of the detector's own sizes. It cannot say whether a detector finds built-up ground or single buildings, and I did not build a baseline that separates the two.
- Per-level results. The resolution-specific settings were selected on 150 development chips per level. They use development chips from the same two cities that make up the test chips of Section 4, though the chips themselves are disjoint and sit in different spatial blocks.
- Real-sensor check at 1.2 m. It covers one resolution, assumes a band order, and compares a differently processed product.
- Per-city scaling. I normalised brightness with one constant per city, computed from pixel values. A deployed pipeline would need an equivalent step, and the colour rule is sensitive to it.
- One detector family. I tested OWL-ViT and OWLv2 with six prompts. Other open-vocabulary detectors (for example Grounding DINO) or a model trained on overhead imagery may break at different resolutions. I did not test them.
- Model revisions. I pinned OWL-ViT to the revision from my earlier paper. I recorded the OWLv2 revision (`cfd3195b`) only for this study, so it may differ from other published OWLv2 numbers.
- Exploratory study. I made some choices after seeing development results, such as widening the tuning grid and switching the break-point reference to the peak. I made both before scoring the test chips, but a pre-registered version would be stronger. The Las Vegas prediction in Section 3.7 is the one place where I wrote down a hypothesis before seeing the data.

## 7. Conclusion

With one fixed setting, OWL-ViT loses half of its F1 above chance at about 2.2 to 2.6 m per pixel in Paris, Khartoum and Las Vegas, and OWLv2 at about 2.0 to 3.1 m. The break does not follow building size in pixels: a prediction that it would, written down before the Las Vegas run, failed, and splitting 411 chips by building size moves the break by only 4% to 12%. On real Sentinel-2 imagery of the same places, both models score 0.0% with these settings, which reproduces the zero in my earlier paper. Re-tuning the threshold and size cap at 10 m recovers about 10% F1 for OWLv2 and 2% for OWL-ViT on real Sentinel-2, close to what the same boxes would score if placed on true building centres. I cannot tell built-up ground from single buildings with these data, and I would not call it building detection.

For anyone planning to apply open-vocabulary detectors to satellite imagery, the practical advice is to start with imagery finer than about 2 m, to treat results at 3 to 10 m as screening at best, and not to trust simulated degradation as a forecast of a real coarse sensor. The next experiment I would run is the one in the discussion: more Sentinel-2 scenes, a baseline that tells built-up ground from buildings, and a detector built for overhead imagery.

## Reproducibility

Repository: https://github.com/erenesinler23/owlvit-resolution-sweep

The repository holds the code, the configs and every table and figure in this paper. `configs/frozen.yaml` holds each setting chosen on development data, and `configs/vegas.yaml` the Las Vegas configuration. The raw model outputs (`archive/`, 2.3 GB) are too large for the repository. The README lists the commands that regenerate them from the SpaceNet 2 download, which needs forward passes of both models. Once they exist, every table and figure recomputes on a CPU. The real Sentinel-2 crops (7 MB) are in `data/sentinel2/`, with the scene used for each chip. `docs/DECISIONS.md` records each choice and when I made it, including the Las Vegas prediction.

## References

- Cameron, A. C., Gelbach, J. B., and Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *Review of Economics and Statistics*, 90(3), 414-427.
- Drusch, M., et al. (2012). Sentinel-2: ESA's optical high-resolution mission for GMES operational services. *Remote Sensing of Environment*, 120, 25-36.
- Efron, B. (1979). Bootstrap methods: another look at the jackknife. *Annals of Statistics*, 7(1), 1-26.
- Esinler, L. E. (2026). Comparing colour thresholding and a vision language model for building matching in satellite images. Submitted to Reinvention, 8 October 2026.
- Kuhn, H. W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly*, 2(1-2), 83-97.
- Minderer, M., et al. (2022). Simple open-vocabulary object detection with vision transformers. *ECCV 2022*. arXiv:2205.06230.
- Minderer, M., Gritsenko, A., and Houlsby, N. (2023). Scaling open-vocabulary object detection. *NeurIPS 2023*. arXiv:2306.09683.
- Radford, A., et al. (2021). Learning transferable visual models from natural language supervision. *ICML 2021*. arXiv:2103.00020.
- Van Etten, A., Lindenbaum, D., and Bacastow, T. M. (2018). SpaceNet: a remote sensing dataset and challenge series. arXiv:1807.01232.

