"""OWL-ViT and OWLv2 wrappers that cache raw outputs before any thresholding.

The runner returns every predicted box with its per-prompt score. Prompt
subsets, score thresholds and size filters are applied later from the cache,
so development tuning and replay never need another forward pass.

Untested without torch: run scripts/smoke_owl.py on the Mac before relying on
it. OWL-ViT resizes the full image to a square. OWLv2 pads to a square first,
so its boxes are scaled by the longer side. SpaceNet chips are square, which
makes that padding a no-op.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from affine import Affine

from .base import Detection, Detector, pixel_box_to_geometry


@dataclass(frozen=True)
class RawDetections:
    boxes_px: np.ndarray   # (N, 4) x0, y0, x1, y1 in input-image pixels
    scores: np.ndarray     # (N, Q) sigmoid score per prompt
    prompts: tuple


def pick_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class OwlRunner:
    def __init__(self, model_id: str = "google/owlvit-base-patch32",
                 revision: str | None = None, prompts: tuple = (),
                 device: str | None = None):
        import torch
        from transformers import (AutoProcessor, Owlv2ForObjectDetection,
                                  OwlViTForObjectDetection)
        self.model_id = model_id
        self.revision = revision
        self.prompts = tuple(prompts)
        self.device = device or pick_device()
        self.padded_square = "owlv2" in model_id.lower()
        model_cls = Owlv2ForObjectDetection if self.padded_square else OwlViTForObjectDetection
        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self._model = model_cls.from_pretrained(model_id, revision=revision).to(self.device)
        self._model.eval()

    def raw(self, rgb: np.ndarray) -> RawDetections:
        from PIL import Image
        image = Image.fromarray(rgb)
        inputs = self._processor(text=[list(self.prompts)], images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            out = self._model(**inputs)
        logits = out.logits[0].float().cpu().numpy()          # (N, Q)
        boxes = out.pred_boxes[0].float().cpu().numpy()       # (N, 4) cx, cy, w, h in [0, 1]
        scores = 1.0 / (1.0 + np.exp(-logits))
        h, w = rgb.shape[:2]
        sx, sy = ((max(h, w),) * 2) if self.padded_square else (w, h)
        cx, cy, bw, bh = boxes.T
        boxes_px = np.stack([(cx - bw / 2) * sx, (cy - bh / 2) * sy,
                             (cx + bw / 2) * sx, (cy + bh / 2) * sy], axis=1)
        return RawDetections(boxes_px.astype(np.float32), scores.astype(np.float32), self.prompts)


class OwlDetector(Detector):
    """Detector-interface adapter: one prompt subset and one score threshold."""

    def __init__(self, runner: OwlRunner, prompt_idx: tuple, score_threshold: float):
        self.runner = runner
        self.prompt_idx = tuple(prompt_idx)
        self.score_threshold = score_threshold

    @property
    def name(self) -> str:
        family = "owlv2" if self.runner.padded_square else "owlvit"
        return f"{family}_p{'-'.join(map(str, self.prompt_idx))}_t{self.score_threshold:g}"

    def detect(self, r, g, b, transform: Affine, raster_crs=None) -> list[Detection]:
        raw = self.runner.raw(np.dstack([r, g, b]).astype(np.uint8))
        s = raw.scores[:, list(self.prompt_idx)].max(axis=1)
        out = []
        for i in np.flatnonzero(s >= self.score_threshold):
            x0, y0, x1, y1 = raw.boxes_px[i]
            out.append(Detection(f"{self.name}-{i}",
                                 pixel_box_to_geometry(x0, y0, x1, y1, transform),
                                 float(s[i])))
        return out
