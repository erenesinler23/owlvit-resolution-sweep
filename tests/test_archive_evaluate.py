import json

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from resolution_sweep import archive
from resolution_sweep.evaluate import (EvalConfig, counts_tensor, evaluate_records,
                                       pooled_table)

LEVELS = [0.3, 0.6]


def make_chip(root, chip_id, city="A"):
    refs = [box(0, 0, 10, 10), box(20, 0, 30, 10)]
    arrays = {}
    b30 = np.array([[0, 0, 10, 10], [20, 0, 30, 10], [100, 100, 190, 190]], dtype=np.float32)
    s30 = np.array([[0.5, 0.1], [0.4, 0.1], [0.02, 0.9]], dtype=np.float32)
    b60 = np.array([[0, 0, 10, 10], [100, 100, 190, 190]], dtype=np.float32)
    s60 = np.array([[0.5, 0.1], [0.02, 0.9]], dtype=np.float32)
    for lv, b, s in ((0.3, b30, s30), (0.6, b60, s60)):
        arrays[archive.array_key("owlvit", "area", lv, "boxes")] = b
        arrays[archive.array_key("owlvit", "area", lv, "scores")] = s
        arrays[archive.array_key("colour-paper_fixed", "area", lv, "boxes")] = b[:1]
    archive.save_chip(root, city, chip_id, arrays, refs)
    return {"chip_id": chip_id, "city": city, "block": f"{city}:0:0", "split": "dev"}


def test_refs_roundtrip_keeps_holes(tmp_path):
    donut = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], [[(3, 3), (7, 3), (7, 7), (3, 7)]])
    archive.save_chip(tmp_path, "A", "c0", {"x": np.zeros(1)}, [donut])
    _, refs = archive.load_chip(tmp_path, "A", "c0")
    assert refs[0].area == pytest.approx(donut.area)


def test_manifest_lists_checksums(tmp_path):
    make_chip(tmp_path, "c0")
    out = archive.write_manifest(tmp_path, {"k": 1})
    data = json.loads(out.read_text())
    assert any(name.endswith("c0.npz") for name in data["checksums"])
    assert data["config"] == {"k": 1}


def test_evaluate_applies_prompt_threshold_and_size_after_caching(tmp_path):
    recs = [make_chip(tmp_path, "c0"), make_chip(tmp_path, "c1")]
    configs = {
        "p0": EvalConfig((0,), 0.1, 60.0, 0.0),
        "p1": EvalConfig((1,), 0.5, 195.0, 0.0),
        "both_small": EvalConfig((0, 1), 0.01, 60.0, 0.0),
    }
    df = evaluate_records(recs, tmp_path, "owlvit", "area", LEVELS, configs, [0.25])
    pooled = pooled_table(df, ["config", "level"]).set_index(["config", "level"])
    # prompt 0, threshold 0.1: two boxes at 0.3 m, both match; one box at 0.6 m
    assert pooled.loc[("p0", 0.3), ["matches", "predictions", "references"]].tolist() == [4, 4, 4]
    assert pooled.loc[("p0", 0.6), ["matches", "predictions"]].tolist() == [2, 2]
    # prompt 1 keeps only the large box, which matches nothing
    assert pooled.loc[("p1", 0.3), ["matches", "predictions"]].tolist() == [0, 2]
    # the 90 m box is removed by the 60 m size filter and counted
    assert pooled.loc[("both_small", 0.3), "removed"] == 2


def test_colour_detector_path_and_counts_tensor(tmp_path):
    recs = [make_chip(tmp_path, f"c{i}") for i in range(3)]
    cfgs = {"fixed": EvalConfig(max_size_m=500.0)}
    df = evaluate_records(recs, tmp_path, "colour-paper_fixed", "area", LEVELS, cfgs, [0.25, 0.5])
    levels, tensor, blocks = counts_tensor(df, "fixed", "area", 0.25)
    assert levels == LEVELS and tensor.shape == (2, 3, 3) and len(blocks) == 3
    assert tensor[0, :, 0].tolist() == [1, 1, 1]


def test_boxes_are_clipped_to_the_chip_and_empty_ones_dropped(tmp_path):
    from resolution_sweep.evaluate import select_boxes
    key = archive.array_key("owlvit", "area", 0.3, "boxes")
    skey = archive.array_key("owlvit", "area", 0.3, "scores")
    arrays = {
        "extent_m": np.array([20.0, 20.0], dtype=np.float32),
        key: np.array([[-5, -5, 10, 10], [25, 25, 30, 30]], dtype=np.float32),
        skey: np.array([[0.9], [0.9]], dtype=np.float32),
    }
    boxes, removed = select_boxes(arrays, "owlvit", "area", 0.3, EvalConfig((0,), 0.0, None, 0.0))
    assert boxes.tolist() == [[0.0, 0.0, 10.0, 10.0]] and removed == 0
