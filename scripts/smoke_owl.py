"""Smoke test: load OWL-ViT, run a synthetic chip three times, print shapes and timing."""
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np

from resolution_sweep.cache import load_config
from resolution_sweep.detectors.owlvit import OwlRunner

cfg = load_config("configs/sweep.yaml")
spec = cfg["detectors"]["owlvit"]
runner = OwlRunner(spec["model_id"], spec.get("revision"), tuple(cfg["detectors"]["prompts"]))
img = (np.random.default_rng(0).random((650, 650, 3)) * 255).astype(np.uint8)
for i in range(3):
    t0 = time.time()
    raw = runner.raw(img)
    print(i, raw.boxes_px.shape, raw.scores.shape, f"{time.time() - t0:.2f}s on {runner.device}")
print("max score on noise:", float(raw.scores.max()))
