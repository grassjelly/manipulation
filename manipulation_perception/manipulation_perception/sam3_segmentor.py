"""
SAM3 backend implementations of PromptToSegment.
No ROS dependency — pure numpy/PIL/torch.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from .prompt_to_segment import LiteLLMClient, PromptToSegment, SegmentResult


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_sam3_processor(device: str, confidence_threshold: float, resolution: int) -> Sam3Processor:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = build_sam3_image_model(bpe_path=None, device=device)
    return Sam3Processor(model, resolution=resolution, device=device, confidence_threshold=confidence_threshold)


def _extract_masks(state: dict) -> tuple[list[np.ndarray], list[float]]:
    nb_objects = len(state.get("scores", []))
    if nb_objects == 0:
        return [], []
    masks_np  = [state["masks"][i][0].cpu().numpy().astype(bool) for i in range(nb_objects)]
    scores_np = [state["scores"][i].item() for i in range(nb_objects)]
    return masks_np, scores_np


def _xyxy_ratio_to_sam3_box(rx1: float, ry1: float, rx2: float, ry2: float) -> list[float]:
    """Convert [rx1, ry1, rx2, ry2] ratio coords to SAM3's [cx, cy, w, h] format."""
    cx = (rx1 + rx2) / 2.0
    cy = (ry1 + ry2) / 2.0
    w  = rx2 - rx1
    h  = ry2 - ry1
    return [cx, cy, w, h]


class Sam3Segmentor(PromptToSegment):
    """
    SAM3 backend: VLM predicts bounding boxes, SAM3 refines them into masks.

    The box-prediction step (VLM call, grid overlay, parsing) is handled by
    the base class ``_segment_by_coord``; this class only implements the
    SAM3-specific geometry via ``add_geometric_prompt``.

    Parameters
    ----------
    llm_client
        VLM connection config for bounding-box prediction.
        Defaults to ``LiteLLMClient()``.
    device
        ``"cuda"`` or ``"cpu"``.
    confidence_threshold
        SAM3 mask confidence threshold (0–1).
    resolution
        Input resolution passed to ``Sam3Processor``.
    """

    def __init__(
        self,
        llm_client: LiteLLMClient | None = None,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        resolution: int = 1008,
    ) -> None:
        super().__init__(llm_client=llm_client or LiteLLMClient())
        self._processor = _build_sam3_processor(device, confidence_threshold, resolution)

    # ------------------------------------------------------------------
    # PromptToSegment abstract method implementations
    # ------------------------------------------------------------------

    def segment(self, rgb_image: np.ndarray, prompt: str) -> list[SegmentResult]:
        return self._segment_by_coord(rgb_image, prompt)

    def generate_masks_from_prompt(
        self, rgb_image: np.ndarray, prompt: str
    ) -> tuple[list[np.ndarray], list[float]]:
        raise NotImplementedError(
            "Sam3Segmentor does not support SOM grounding. Use grounding='coord'."
        )

    def generate_masks_from_bbox(
        self,
        rgb_image: np.ndarray,
        boxes: list[tuple[float, float, float, float]],
    ) -> tuple[list[np.ndarray], list[float]]:
        """Feed all ratio boxes into SAM3's geometric prompt in one pass.

        Boxes are accumulated in the state (no reset between them); SAM3 returns
        one mask per box in a single inference call.
        """
        pil_image = PILImage.fromarray(rgb_image)

        t_encode = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = self._processor.set_image(pil_image)
        print(f"[sam3_bbox] image encoding: {time.perf_counter() - t_encode:.3f}s", flush=True)

        t_prompts = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for rx1, ry1, rx2, ry2 in boxes:
                sam3_box = _xyxy_ratio_to_sam3_box(rx1, ry1, rx2, ry2)
                state = self._processor.add_geometric_prompt(sam3_box, True, state)
        print(f"[sam3_bbox] geometric prompts ({len(boxes)} box(es)): {time.perf_counter() - t_prompts:.3f}s", flush=True)

        return _extract_masks(state)

    def draw_boxes_overlay(
        self,
        rgb_image: np.ndarray,
        boxes: list[tuple[float, float, float, float]] | None = None,
        thickness: int = 3,
        font_scale: float = 0.7,
    ) -> np.ndarray:
        """Return a copy of *rgb_image* with each VLM-predicted bounding box drawn.

        Parameters
        ----------
        rgb_image
            Source image in RGB order.
        boxes
            List of ``(rx1, ry1, rx2, ry2)`` ratio boxes.  Defaults to
            ``self.last_boxes`` when not supplied.
        """
        from .prompt_to_segment import _MASK_COLOURS

        if boxes is None:
            boxes = self.last_boxes

        h, w = rgb_image.shape[:2]
        canvas = rgb_image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_thickness = max(1, round(font_scale * 2))

        for idx, (rx1, ry1, rx2, ry2) in enumerate(boxes):
            x1, y1 = int(rx1 * w), int(ry1 * h)
            x2, y2 = int(rx2 * w), int(ry2 * h)
            colour = _MASK_COLOURS[idx % len(_MASK_COLOURS)]

            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)

            label = str(idx)
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
            pad = 4
            lx, ly = x1, y1 - pad
            if ly - th < 0:
                ly = y1 + th + pad
            cv2.rectangle(canvas, (lx - pad, ly - th - pad), (lx + tw + pad, ly + pad),
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(canvas, label, (lx, ly), font, font_scale, colour, text_thickness, cv2.LINE_AA)

        return canvas
