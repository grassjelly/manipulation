"""
SAM3 backend implementations of PromptToSegment.
No ROS dependency — pure numpy/PIL/torch.
"""
from __future__ import annotations

import torch
import numpy as np
from PIL import Image as PILImage

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from .prompt_to_segment import LiteLLMClient, PromptToSegment, SegmentResult


def _build_sam3_processor(device: str, confidence_threshold: float, resolution: int) -> Sam3Processor:
    # Match notebook setup: enable TF32 for Ampere GPUs and use bfloat16 globally
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    model = build_sam3_image_model(bpe_path=None, device=device)
    return Sam3Processor(model, resolution=resolution, device=device, confidence_threshold=confidence_threshold)


def _extract_masks(state: dict) -> tuple[list[np.ndarray], list[float]]:
    nb_objects = len(state.get("scores", []))
    if nb_objects == 0:
        return [], []

    masks_np  = [state["masks"][i].squeeze(0).cpu().numpy().astype(bool) for i in range(nb_objects)]
    scores_np = [state["scores"][i].item() for i in range(nb_objects)]

    order = np.argsort(scores_np)[::-1]
    return [masks_np[i] for i in order], [scores_np[i] for i in order]


class Sam3Segmentor(PromptToSegment):
    """
    SAM3 backend that uses the prompt directly and bypasses LLM arbitration.

    The prompt is fed into SAM3's text encoder; the highest-scoring mask
    is returned immediately without overlay rendering or an LLM call.

    Parameters
    ----------
    device
        ``"cuda"`` or ``"cpu"``.
    confidence_threshold
        SAM3 mask confidence threshold (0–1).
    resolution
        Input resolution passed to ``Sam3Processor``.
    """

    def __init__(
        self,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        resolution: int = 1008,
    ) -> None:
        super().__init__(llm_client=None)
        self._processor = _build_sam3_processor(device, confidence_threshold, resolution)

    def generate_masks(
        self, rgb_image: np.ndarray, prompt: str
    ) -> tuple[list[np.ndarray], list[float]]:
        pil_image = PILImage.fromarray(rgb_image)
        inference_state = self._processor.set_image(pil_image)
        self._processor.reset_all_prompts(inference_state)
        inference_state = self._processor.set_text_prompt(state=inference_state, prompt=prompt)
        return _extract_masks(inference_state)

    def segment(self, rgb_image: np.ndarray, prompt: str) -> list[SegmentResult]:
        masks, _ = self.generate_masks(rgb_image, prompt)
        results: list[SegmentResult] = []
        for idx, mask in enumerate(masks):
            if not mask.any():
                continue
            ys, xs = np.where(mask)
            results.append(SegmentResult(
                mask=mask,
                centroid_px=(int(xs.mean()), int(ys.mean())),
                mask_ids=[idx],
            ))
        return results
