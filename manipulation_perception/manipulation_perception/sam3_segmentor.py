"""
SAM3 backend implementations of PromptToSegment.
No ROS dependency — pure numpy/PIL/torch.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image as PILImage

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from .prompt_to_segment import LiteLLMClient, PromptToSegment, SegmentResult


def _build_sam3_processor(device: str, confidence_threshold: float, resolution: int) -> Sam3Processor:
    """Build and return a configured Sam3Processor with TF32 acceleration enabled."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = build_sam3_image_model(bpe_path=None, device=device)
    return Sam3Processor(model, resolution=resolution, device=device, confidence_threshold=confidence_threshold)


def _extract_masks(state: dict) -> tuple[list[np.ndarray], list[float]]:
    """Extract boolean masks and scores from a SAM3 inference state dict."""
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
    the base class ``segment_by_coord``; this class only implements the
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
        super().__init__(llm_client=llm_client)
        self._processor = _build_sam3_processor(device, confidence_threshold, resolution)


    def segment(self, rgb_image: np.ndarray, prompt: str) -> list[SegmentResult]:
        if self._llm is None:
            masks, _ = self.generate_masks_from_prompt(rgb_image, prompt)
            return self.build_results(masks)
        return self.segment_by_coord(rgb_image, prompt)

    def generate_masks_from_prompt(
        self, rgb_image: np.ndarray, prompt: str
    ) -> tuple[list[np.ndarray], list[float]]:
        """Feed the text prompt directly into SAM3's text encoder."""
        pil_image = PILImage.fromarray(rgb_image)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = self._processor.set_image(pil_image)
            self._processor.reset_all_prompts(state)
            state = self._processor.set_text_prompt(state=state, prompt=prompt)
        nb_objects = len(state.get("scores", []))
        if nb_objects == 0:
            return [], []
        masks_np  = [state["masks"][i].squeeze(0).cpu().numpy().astype(bool) for i in range(nb_objects)]
        scores_np = [state["scores"][i].item() for i in range(nb_objects)]
        order = np.argsort(scores_np)[::-1]
        return [masks_np[i] for i in order], [scores_np[i] for i in order]

    def generate_masks_from_bboxes(
        self,
        rgb_image: np.ndarray,
        boxes: list[tuple[float, float, float, float]],
    ) -> tuple[list[np.ndarray], list[float]]:
        """For each box, run a separate SAM3 inference and keep the highest-scoring mask.

        SAM3 generates multiple candidate masks per prompt; processing each box
        independently and taking the best score ensures exactly one mask per box.
        Image encoding is done once and reused across all boxes.
        """
        pil_image = PILImage.fromarray(rgb_image)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = self._processor.set_image(pil_image)

        best_masks: list[np.ndarray] = []
        best_scores: list[float] = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for rx1, ry1, rx2, ry2 in boxes:
                self._processor.reset_all_prompts(state)
                sam3_box = _xyxy_ratio_to_sam3_box(rx1, ry1, rx2, ry2)
                state = self._processor.add_geometric_prompt(sam3_box, True, state)
                masks, scores = _extract_masks(state)
                if masks:
                    best_idx = int(np.argmax(scores))
                    best_masks.append(masks[best_idx])
                    best_scores.append(scores[best_idx])

        return best_masks, best_scores

