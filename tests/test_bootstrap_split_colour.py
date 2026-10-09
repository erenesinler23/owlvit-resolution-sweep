import numpy as np
import pytest
from affine import Affine

from resolution_sweep.analysis import break_point
from resolution_sweep.bootstrap import bootstrap_f1_curves, cluster_bootstrap
from resolution_sweep.detectors.colour import ColourRuleDetector
from resolution_sweep.matching import prf
from resolution_sweep.split import assign_split, subsample


def test_bootstrap_point_estimate_matches_pooled_counts():
    counts = np.array([[2, 5, 8], [1, 4, 6], [0, 3, 5], [3, 6, 9], [1, 2, 4], [2, 2, 3]])
    clusters = np.array([0, 0, 1, 1, 2, 2])
    out = cluster_bootstrap(counts, clusters, n_boot=500, seed=3)
    exp = prf(*counts.sum(axis=0))
    assert out["f1"][0] == pytest.approx(float(exp[2]))
    assert out["precision"][0] == pytest.approx(float(exp[0]))
    assert out["f1"][1] <= out["f1"][2]


def test_bootstrap_is_deterministic():
    counts = np.random.default_rng(0).integers(0, 10, (30, 3))
    clusters = np.repeat(np.arange(10), 3)
    a = cluster_bootstrap(counts, clusters, n_boot=300, seed=7)
    b = cluster_bootstrap(counts, clusters, n_boot=300, seed=7)
    assert a == b


def test_curve_bootstrap_shapes_and_pairing():
    rng = np.random.default_rng(1)
    counts = rng.integers(1, 10, (4, 20, 3))
    clusters = np.repeat(np.arange(5), 4)
    point, boot = bootstrap_f1_curves(counts, clusters, n_boot=200, seed=2)
    assert point.shape == (4,) and boot.shape == (200, 4)


def test_split_keeps_blocks_whole_and_is_deterministic():
    n = 20
    ids, cities, xs, ys = [], [], [], []
    for i in range(n):
        for j in range(n):
            ids.append(f"c{i}_{j}")
            cities.append("A")
            xs.append(i * 0.002)
            ys.append(j * 0.002)
    recs = assign_split(ids, cities, xs, ys, cell=0.01, dev_fraction=0.3, seed=5)
    again = assign_split(ids, cities, xs, ys, cell=0.01, dev_fraction=0.3, seed=5)
    assert recs == again
    by_block = {}
    for r in recs:
        by_block.setdefault(r["block"], set()).add(r["split"])
    assert all(len(s) == 1 for s in by_block.values())
    dev_share = np.mean([r["split"] == "dev" for r in recs])
    assert 0.1 <= dev_share <= 0.5


def test_subsample_is_seeded_and_sized():
    recs = [{"chip_id": f"c{i}"} for i in range(100)]
    a = subsample(recs, 10, seed=1)
    assert a == subsample(recs, 10, seed=1) and len(a) == 10


def test_colour_rule_boxes_in_metric_frame():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[4:9, 6:12] = (200, 50, 50)
    tf = Affine.scale(0.3, 0.3)
    out = ColourRuleDetector().detect(img[..., 0], img[..., 1], img[..., 2], tf, None)
    assert len(out) == 1
    x0, y0, x1, y1 = out[0].geometry.bounds
    assert (x0, y0, x1, y1) == pytest.approx((6 * 0.3, 4 * 0.3, 12 * 0.3, 9 * 0.3))


def test_colour_rule_uses_strict_inequalities():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[2:5, 2:5] = (120, 50, 50)   # red exactly 120 is not "above 120"
    out = ColourRuleDetector().detect(img[..., 0], img[..., 1], img[..., 2], Affine.identity(), None)
    assert out == []


def test_break_point_interpolates_and_handles_floors():
    assert break_point([0.3, 0.6, 1.2, 2.4], [40, 38, 20, 2]) == pytest.approx(1.2)
    assert break_point([0.3, 0.6, 1.2], [40, 39, 38]) is None
    assert break_point([0.3, 0.6, 1.2], [0, 0, 0]) is None
    mid = break_point([1.0, 10.0], [40, 0])
    assert mid == pytest.approx(10 ** 0.5)

def test_colour_rule_defaults_to_four_connectivity_like_the_original():
    img = np.zeros((6, 6, 3), dtype=np.uint8)
    img[1, 1] = (200, 50, 50)
    img[2, 2] = (200, 50, 50)   # diagonal neighbour
    args = (img[..., 0], img[..., 1], img[..., 2], Affine.identity(), None)
    assert len(ColourRuleDetector().detect(*args)) == 2
    assert len(ColourRuleDetector(connectivity=8).detect(*args)) == 1
