"""
Falcon-perception segmentation wrapper.
No ROS dependency — pure numpy/PIL/torch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image as PILImage
from pycocotools import mask as mask_utils

from falcon_perception import (
    PERCEPTION_MODEL_ID,
    build_prompt_for_task,
    load_and_prepare_model,
)
from falcon_perception.data import ImageProcessor
from falcon_perception.paged_inference import PagedInferenceEngine, SamplingParams, Sequence


@dataclass
class SegmentResult:
    mask: np.ndarray        # bool (H, W)
    centroid_px: tuple[int, int]  # (u, v) pixel coordinates


class FalconSegmentor:
    """
    Loads a falcon-perception model once and exposes a single
    ``segment(rgb_image, prompt)`` call.

    Parameters
    ----------
    hf_model_id
        HuggingFace model ID.  Defaults to ``PERCEPTION_MODEL_ID`` from the
        falcon_perception package when *None*.
    device
        ``"cuda"`` or ``"cpu"``.
    dtype
        ``"float32"`` or ``"float16"``.
    segmentation_threshold
        Confidence threshold passed to ``SamplingParams``.
    hr_upsample_ratio
        High-resolution upsampling ratio passed to ``SamplingParams``.
    max_new_tokens
        Token budget for the decoder.
    """

    def __init__(
        self,
        hf_model_id: str | None = None,
        device: str = "cuda",
        dtype: str = "float32",
        segmentation_threshold: float = 0.3,
        hr_upsample_ratio: int = 8,
        max_new_tokens: int = 256,
    ) -> None:
        model_id = hf_model_id if hf_model_id else PERCEPTION_MODEL_ID
        model, tokenizer, _ = load_and_prepare_model(
            hf_model_id=model_id,
            device=device,
            dtype=dtype,
            compile=True,
        )

        image_processor = ImageProcessor(patch_size=16, merge_size=1)
        self._engine = PagedInferenceEngine(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            max_batch_size=32,
            max_seq_length=8192,
            n_pages=512,
            page_size=128,
            enable_hr_cache=True,
            max_hr_cache_entries=128,
        )
        self._sampling_params = SamplingParams(
            max_new_tokens=max_new_tokens,
            segmentation_threshold=segmentation_threshold,
            hr_upsample_ratio=hr_upsample_ratio,
        )

    def segment(self, rgb_image: np.ndarray, prompt: str) -> list[SegmentResult]:
        """
        Run segmentation on *rgb_image* for the given *prompt*.

        Parameters
        ----------
        rgb_image
            ``(H, W, 3)`` uint8 array in **RGB** order.
        prompt
            Natural-language description of the target object(s).

        Returns
        -------
        list[SegmentResult]
            One entry per detected instance, ordered by mask area (largest first).
            Empty list when no segments are found.
        """
        pil_image = PILImage.fromarray(rgb_image)
        text = build_prompt_for_task(prompt, task="segmentation")
        seq = Sequence(
            text=text,
            image=pil_image,
            min_image_size=256,
            max_image_size=1024,
            task="segmentation",
        )

        self._engine.generate(
            sequences=[seq],
            sampling_params=self._sampling_params,
            use_tqdm=False,
        )

        results: list[SegmentResult] = []
        for rle_dict in seq.output_aux.masks_rle:
            rle = rle_dict
            if isinstance(rle.get("counts"), str):
                rle = {**rle_dict, "counts": rle_dict["counts"].encode("utf-8")}
            mask = mask_utils.decode(rle).astype(bool)
            if not mask.any():
                continue
            ys, xs = np.where(mask)
            centroid_px = (int(xs.mean()), int(ys.mean()))
            results.append(SegmentResult(mask=mask, centroid_px=centroid_px))

        # largest mask first so index 0 is the most prominent detection
        results.sort(key=lambda r: int(r.mask.sum()), reverse=True)
        return results
