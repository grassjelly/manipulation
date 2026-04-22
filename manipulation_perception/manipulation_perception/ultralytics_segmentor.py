"""
Ultralytics SAM backend implementations of PromptToSegment.
No ROS dependency — pure numpy/PIL.
"""
from __future__ import annotations

import numpy as np
import cv2
from PIL import Image as PILImage

from .prompt_to_segment import LiteLLMClient, PromptToSegment, SegmentResult


_DEFAULT_MODEL = "sam2.1_b.pt"


def _build_ultralytics_model(model_name: str, device: str):
    from ultralytics import SAM
    model = SAM(model_name)
    model.to(device)
    return model


def _parse_results(
    results, shape: tuple[int, int]
) -> tuple[list[np.ndarray], list[float]]:
    if not results or results[0].masks is None:
        return [], []

    H, W = shape
    masks_tensor = results[0].masks.data
    scores_raw   = results[0].boxes.conf if results[0].boxes is not None else None

    masks_np  = masks_tensor.cpu().numpy().astype(bool)
    scores_np = (
        scores_raw.cpu().numpy()
        if scores_raw is not None and len(scores_raw) == len(masks_np)
        else np.ones(len(masks_np))
    )

    if masks_np.shape[1:] != (H, W):
        masks_np = np.stack([
            cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            for m in masks_np
        ])

    order = np.argsort(scores_np)[::-1]
    return [masks_np[i] for i in order], [float(scores_np[i]) for i in order]


class UltralyticsSamSegmentor(PromptToSegment):
    """
    ``PromptToSegment`` backend using Ultralytics SAM in auto-everything mode.

    SAM segments all visible objects with no text input; the LLM arbitrates
    over the labelled overlay to select the mask matching the user's prompt.

    Parameters
    ----------
    llm_client
        ``LiteLLMClient`` used for mask arbitration. Required.
    model_name
        Ultralytics model checkpoint, e.g. ``"sam2.1_b.pt"``.
    device
        ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        llm_client: LiteLLMClient,
        model_name: str = _DEFAULT_MODEL,
        device: str = "cuda",
    ) -> None:
        super().__init__(llm_client)
        self._model = _build_ultralytics_model(model_name, device)

    def generate_masks(
        self, rgb_image: np.ndarray, prompt: str  # prompt intentionally unused
    ) -> tuple[list[np.ndarray], list[float]]:
        results = self._model(PILImage.fromarray(rgb_image), verbose=False)
        return _parse_results(results, rgb_image.shape[:2])


class UltralyticsTextSamSegmentor(PromptToSegment):
    """
    Ultralytics SAM backend that uses a text prompt directly and bypasses
    LLM arbitration.

    The prompt is passed to the SAM model's text-grounding interface; the
    highest-scoring mask is returned immediately without overlay rendering
    or an LLM call.

    Parameters
    ----------
    model_name
        Ultralytics model checkpoint, e.g. ``"sam2.1_b.pt"``.
    device
        ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "cuda",
    ) -> None:
        super().__init__(llm_client=None)
        self._model = _build_ultralytics_model(model_name, device)

    def generate_masks(
        self, rgb_image: np.ndarray, prompt: str
    ) -> tuple[list[np.ndarray], list[float]]:
        results = self._model(PILImage.fromarray(rgb_image), texts=prompt, verbose=False)
        return _parse_results(results, rgb_image.shape[:2])

    def segment(self, rgb_image: np.ndarray, prompt: str) -> SegmentResult | None:
        masks, scores = self.generate_masks(rgb_image, prompt)
        if not masks:
            return None

        best = int(np.argmax(scores))
        mask = masks[best]
        if not mask.any():
            return None

        ys, xs = np.where(mask)
        return SegmentResult(
            mask=mask,
            centroid_px=(int(xs.mean()), int(ys.mean())),
            mask_id=best,
        )
