"""
SAM2 automatic mask generator backend for PromptToSegment.
No ROS dependency — pure numpy/PIL/torch.
"""
from __future__ import annotations

import os
import numpy as np

import torch

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from .prompt_to_segment import LiteLLMClient, PromptToSegment

_DEFAULT_CHECKPOINT = os.path.expanduser("~/sam2_checkpoints/sam2.1_hiera_large.pt")
_DEFAULT_MODEL_CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"


class Sam2Segmentor(PromptToSegment):
    """
    SAM2 automatic mask generator backend.

    ``generate_masks`` ignores the text prompt and generates all masks via
    SAM2's ``SAM2AutomaticMaskGenerator``.  The base-class ``segment`` method
    then renders an overlay and uses an LLM to pick the mask that best matches
    the prompt.

    Parameters
    ----------
    llm_client
        ``LiteLLMClient`` used by the base-class ``segment`` for LLM
        arbitration.  Required — SAM2 is prompt-free so the LLM is the only
        way to select a mask.
    checkpoint
        Path to the SAM2 ``.pt`` weights file.
    model_cfg
        SAM2 model config path (relative to the ``sam2`` package root or an
        absolute path).
    device
        ``"cuda"`` or ``"cpu"``.
    points_per_side
        Controls mask density.  Higher → more masks, slower inference.
    pred_iou_thresh
        Masks with predicted IoU below this value are discarded.
    stability_score_thresh
        Masks with stability score below this value are discarded.
    """

    def __init__(
        self,
        llm_client: LiteLLMClient,
        checkpoint: str = _DEFAULT_CHECKPOINT,
        model_cfg: str = _DEFAULT_MODEL_CFG,
        device: str = "cuda",
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
    ) -> None:
        super().__init__(llm_client=llm_client)

        # Enable TF32 on Ampere GPUs for faster matmul
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        sam2_model = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)
        self._generator = SAM2AutomaticMaskGenerator(
            sam2_model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
        )

    def generate_masks(
        self, rgb_image: np.ndarray, prompt: str
    ) -> tuple[list[np.ndarray], list[float]]:
        """Run SAM2 automatic mask generation; *prompt* is ignored."""
        # SAM2 expects a uint8 HWC RGB numpy array
        raw_masks = self._generator.generate(rgb_image)

        if not raw_masks:
            return [], []

        # Sort by predicted_iou descending
        raw_masks.sort(key=lambda m: m["predicted_iou"], reverse=True)

        masks  = [m["segmentation"].astype(bool) for m in raw_masks]
        scores = [float(m["predicted_iou"])       for m in raw_masks]
        return masks, scores
