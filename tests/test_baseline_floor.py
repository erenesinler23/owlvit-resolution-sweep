import numpy as np
from shapely.geometry import box

from resolution_sweep.analysis import above_chance, break_point
from resolution_sweep.baseline import chance_matches, relocate_boxes


def test_relocate_keeps_size_and_stays_inside():
    rng = np.random.default_rng(0)
    b = np.array([[10, 10, 30, 25], [0, 0, 5, 5]], dtype=float)
    out = relocate_boxes(b, (100, 80), rng)
    assert np.allclose(out[:, 2] - out[:, 0], b[:, 2] - b[:, 0])
    assert np.allclose(out[:, 3] - out[:, 1], b[:, 3] - b[:, 1])
    assert (out[:, :2] >= 0).all() and (out[:, 2] <= 100).all() and (out[:, 3] <= 80).all()


def test_chance_full_chip_box_matches_everything():
    refs = [box(0, 0, 100, 100)]
    assert chance_matches(np.array([[0, 0, 100, 100]]), refs, (100, 100), 0.5) == 1.0


def test_chance_tiny_box_rarely_matches():
    refs = [box(10, 10, 20, 20)]
    m = chance_matches(np.array([[0, 0, 10, 10]]), refs, (1000, 1000), 0.5, n_rep=50)
    assert m == 0.0


def test_chance_empty():
    assert chance_matches(np.zeros((0, 4)), [box(0, 0, 1, 1)], (10, 10), 0.5) == 0.0


def test_break_point_with_floor():
    g = [0.3, 0.6, 1.2, 2.4]
    bp = break_point(g, [41, 39, 21, 3], floor=[1, 1, 1, 1])
    assert abs(bp - 1.2) < 1e-6  # normalised F1 is exactly 0.5 at 1.2 m
    bp2 = break_point(g, [41, 39, 22, 3], floor=[1, 1, 1, 1])
    assert 1.2 < bp2 < 2.4
    assert break_point(g, [41, 39, 21, 3]) is not None


def test_break_point_none_cases():
    g = [0.3, 0.6, 1.2]
    assert break_point(g, [1, 1, 1], floor=[2, 2, 2]) is None
    assert break_point(g, [40, 39, 38]) is None


def test_above_chance():
    boot = np.random.default_rng(0).normal(20, 1, 500)
    assert above_chance(boot, 5.0)
    assert not above_chance(boot, 25.0)
    assert not above_chance([], 1.0)


def test_break_point_peak_reference():
    g = [0.3, 0.6, 1.2, 2.4, 3.6]
    f1 = [30, 40, 40, 20, 2]
    fin = break_point(g, f1)
    pk = break_point(g, f1, ref="peak")
    assert fin is not None and pk is not None
    assert pk > fin or pk > 1.2          # peak reference only searches coarser than the peak
    assert 1.2 < pk < 2.4 or 2.4 <= pk < 3.6
    # a flat low curve that never halves from its peak has no break point
    assert break_point(g, [8, 8, 8, 8, 8], ref="peak") is None
