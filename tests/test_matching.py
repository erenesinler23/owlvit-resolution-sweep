import numpy as np
import pytest
from shapely.geometry import box

from resolution_sweep.matching import match, match_boxes, prf, size_mask


def test_identical_boxes_match_once():
    res = match([box(0, 0, 2, 2)], [box(0, 0, 2, 2)], 0.25)
    assert (res.matches, res.predictions, res.references) == (1, 1, 1)


def test_duplicate_predictions_earn_one_match():
    res = match([box(0, 0, 2, 2), box(0, 0, 2, 2)], [box(0, 0, 2, 2)], 0.25)
    assert res.matches == 1 and res.predictions == 2


def test_boundary_contact_is_not_a_match():
    res = match([box(0, 0, 2, 2)], [box(2, 0, 4, 2)], 0.25)
    assert res.matches == 0


def test_threshold_decides_validity():
    # intersection 2, union 6, IoU = 1/3
    pred, ref = [box(1, 0, 3, 2)], [box(0, 0, 2, 2)]
    assert match(pred, ref, 0.25).matches == 1
    assert match(pred, ref, 0.5).matches == 0


def test_assignment_maximises_pair_count_before_iou():
    # Greedy-by-IoU takes P1-R1 (IoU 1.0) and stops at 1 pair.
    # The optimal assignment is P1-R2 plus P2-R1: 2 pairs.
    preds = [box(0, 0, 4, 4), box(0, 0, 2, 4)]
    refs = [box(0, 0, 4, 4), box(2, 0, 6, 4)]
    assert match(preds, refs, 0.25).matches == 2


def test_empty_inputs():
    assert match([], [box(0, 0, 1, 1)], 0.25).matches == 0
    assert match([box(0, 0, 1, 1)], [], 0.25).matches == 0
    p, r, f = prf(0, 0, 10)
    assert np.isnan(p) and r == 0 and f == 0
    p, r, f = prf(0, 5, 0)
    assert p == 0 and np.isnan(r) and f == 0


def test_size_mask_removes_wide_and_tall_boxes():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 600, 10], [0, 0, 10, 600]])
    assert size_mask(boxes, max_size_m=500).tolist() == [True, False, False]
    assert size_mask(boxes, max_size_m=None, min_area_m2=200).tolist() == [False, True, True]


def test_match_boxes_accepts_arrays():
    res = match_boxes(np.array([[0, 0, 2, 2]]), [box(0, 0, 2, 2)], 0.25)
    assert res.matches == 1


@pytest.mark.parametrize("name,m,p,r,prec,rec,f1", [
    ("Ankara", 133, 2750, 6769, 4.84, 1.96, 2.79),
    ("Marrakech", 26, 334, 13402, 7.78, 0.19, 0.38),
    ("Istanbul", 367, 1544, 5819, 23.77, 6.31, 9.97),
    ("Prague", 481, 1929, 8237, 24.94, 5.84, 9.46),
    ("New Delhi", 105, 2081, 1435, 5.05, 7.32, 5.97),
])
def test_metrics_reproduce_paper_table2(name, m, p, r, prec, rec, f1):
    got = prf(m, p, r)
    assert round(float(got[0]), 2) == prec
    assert round(float(got[1]), 2) == rec
    assert round(float(got[2]), 2) == f1
