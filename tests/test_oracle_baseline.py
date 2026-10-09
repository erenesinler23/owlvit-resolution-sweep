import numpy as np
from shapely.geometry import box

from resolution_sweep.baseline import chance_matches, relocate_to_refs


def test_relocate_to_refs_keeps_sizes_and_hits_centroids():
    refs = [box(10, 10, 20, 20), box(100, 50, 110, 70), box(30, 80, 40, 90)]
    boxes = np.array([[0, 0, 4, 6], [0, 0, 8, 2]], dtype=float)
    out = relocate_to_refs(boxes, refs, np.random.default_rng(0))
    assert np.allclose(out[:, 2] - out[:, 0], boxes[:, 2] - boxes[:, 0])
    assert np.allclose(out[:, 3] - out[:, 1], boxes[:, 3] - boxes[:, 1])
    cents = {(15.0, 15.0), (105.0, 60.0), (35.0, 85.0)}
    centres = {(float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)) for b in out}
    assert centres <= cents and len(centres) == 2   # without replacement while centroids last


def test_relocate_to_refs_more_boxes_than_refs():
    refs = [box(0, 0, 10, 10)]
    boxes = np.tile([0.0, 0.0, 5.0, 5.0], (4, 1))
    out = relocate_to_refs(boxes, refs, np.random.default_rng(1))
    assert out.shape == (4, 4)


def test_oracle_beats_uniform_when_buildings_are_clustered():
    refs = [box(10 + 12 * i, 10, 20 + 12 * i, 20) for i in range(6)]
    boxes = np.tile([0.0, 0.0, 10.0, 10.0], (6, 1))
    orc = chance_matches(boxes, refs, (195.0, 195.0), 0.25, n_rep=10, seed=0, mode="oracle")
    uni = chance_matches(boxes, refs, (195.0, 195.0), 0.25, n_rep=10, seed=0, mode="uniform")
    assert orc > uni
    assert orc >= 5
