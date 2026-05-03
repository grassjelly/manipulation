"""
SAM2 backend for PromptToSegment.
No ROS dependency — pure numpy/PIL/torch.
"""
from __future__ import annotations

import os
from typing import Literal

import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .prompt_to_segment import LiteLLMClient, PromptToSegment, SegmentResult

_DEFAULT_CHECKPOINT = os.path.expanduser("~/sam2_checkpoints/sam2.1_hiera_small.pt")
_DEFAULT_MODEL_CFG  = "configs/sam2.1/sam2.1_hiera_s.yaml"


class Sam2Segmentor(PromptToSegment):
    """
    SAM2 backend with selectable grounding strategy.

    Parameters
    ----------
    llm_client
        Required for both grounding modes: ``"som"`` uses it for mask
        arbitration; ``"coord"`` uses it to predict bounding boxes.
    grounding
        ``"som"``   — automatic mask generation + LLM set-of-marks arbitration.
                      Builds ``SAM2AutomaticMaskGenerator``.
        ``"coord"`` — VLM predicts bounding boxes, SAM2 refines them into masks.
                      Builds ``SAM2ImagePredictor``.
    checkpoint
        Path to the SAM2 ``.pt`` weights file.
    model_cfg
        SAM2 model config path (relative to the ``sam2`` package root or absolute).
    device
        ``"cuda"`` or ``"cpu"``.
    points_per_side
        ``"som"`` only — controls mask density. Higher → more masks, slower inference.
    pred_iou_thresh
        ``"som"`` only — discard masks with predicted IoU below this value.
    stability_score_thresh
        ``"som"`` only — discard masks with stability score below this value.
    min_mask_region_area
        ``"som"`` only — discard masks smaller than this many pixels.
    """

    def __init__(
        self,
        llm_client: LiteLLMClient,
        grounding: Literal["som", "coord"] = "coord",
        checkpoint: str = _DEFAULT_CHECKPOINT,
        model_cfg: str = _DEFAULT_MODEL_CFG,
        device: str = "cuda",
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.75,
        stability_score_thresh: float = 0.95,
        min_mask_region_area: int = 100,
    ) -> None:
        super().__init__(llm_client=llm_client)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        sam2_model = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)
        self._grounding = grounding

        if grounding == "som":
            self._generator = SAM2AutomaticMaskGenerator(
                sam2_model,
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                min_mask_region_area=min_mask_region_area,
            )
        else:
            self._predictor = SAM2ImagePredictor(sam2_model)


    def segment(self, rgb_image: np.ndarray, prompt: str) -> list[SegmentResult]:
        if self._grounding == "som":
            return self.segment_som(rgb_image, prompt)
        return self.segment_by_coord(rgb_image, prompt)


    def generate_masks(
        self, rgb_image: np.ndarray
    ) -> tuple[list[np.ndarray], list[float]]:
        """``"som"`` backend — generate all candidate masks automatically."""
        raw_masks = self._generator.generate(rgb_image)
        if not raw_masks:
            return [], []
        raw_masks.sort(key=lambda m: m["predicted_iou"], reverse=True)
        masks  = [m["segmentation"].astype(bool) for m in raw_masks]
        scores = [float(m["predicted_iou"])       for m in raw_masks]
        return masks, scores

    def generate_masks_from_bboxes(
        self,
        rgb_image: np.ndarray,
        boxes: list[tuple[float, float, float, float]],
    ) -> tuple[list[np.ndarray], list[float]]:
        """``"coord"`` backend — refine each ratio-coord box into a mask.

        Image encoding is computed once and reused across all boxes.
        """
        h, w = rgb_image.shape[:2]
        best_masks: list[np.ndarray] = []
        best_scores: list[float] = []

        with torch.no_grad():
            self._predictor.set_image(rgb_image)
            for rx1, ry1, rx2, ry2 in boxes:
                pixel_box = np.array([rx1 * w, ry1 * h, rx2 * w, ry2 * h])
                masks, scores, _ = self._predictor.predict(
                    box=pixel_box,
                    multimask_output=True,
                )
                if len(masks):
                    best_idx = int(np.argmax(scores))
                    best_masks.append(masks[best_idx].astype(bool))
                    best_scores.append(float(scores[best_idx]))

        return best_masks, best_scores
