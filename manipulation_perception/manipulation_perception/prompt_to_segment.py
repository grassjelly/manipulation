"""
Abstract interface for prompt-driven instance segmentation.
No ROS dependency — pure numpy/PIL.
"""
from __future__ import annotations

import base64
import io
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont
import litellm
import logging

_MASK_COLOURS: list[tuple[int, int, int, int]] = [
    (220,  50,  50, 120),  # red
    ( 50, 180,  50, 120),  # green
    ( 50, 100, 220, 120),  # blue
    (220, 180,  50, 120),  # yellow
    (180,  50, 220, 120),  # purple
    ( 50, 210, 200, 120),  # cyan
    (230, 120,  50, 120),  # orange
    (200,  50, 130, 120),  # pink
]

_SYSTEM_PROMPT_TEMPLATE = (
    "You are a vision assistant helping a robot manipulation system identify "
    "objects in a scene.\n\n"
    "Context: The image has been pre-processed for instance segmentation. "
    "Each detected region is filled with a distinct colour and labelled with "
    "a numeric mask ID.\n\n"
    "Task: {prompt}\n\n"
    "Instructions:\n"
    "- Examine the labelled regions carefully.\n"
    "- Identify which mask ID best corresponds to the task described above.\n"
    "- If no region matches, set mask_id to -1.\n\n"
    "Reply ONLY with valid JSON in exactly this format:\n"
    '{{"mask_id": <integer>, "confidence": "<high|medium|low>", '
    '"reasoning": "<one sentence>"}}'
)


@dataclass
class LiteLLMClient:
    """Connection config passed to ``PromptToSegment`` for LLM arbitration."""
    model: str   = "ollama/gemma4:e4b"
    api_base: str = "http://localhost:11434"
    api_key: str  = "sk-1234"


@dataclass
class SegmentResult:
    mask: np.ndarray           # bool (H, W)
    centroid_px: tuple[int, int]  # (u, v) pixel coords
    mask_id: int               # index assigned during overlay


class PromptToSegment(ABC):
    """
    Base class for prompt-driven instance segmentation backends.

    Subclasses implement ``generate_masks`` to produce candidate masks from a
    backend of their choice.  This class handles overlay rendering, LLM
    arbitration, and result construction.

    Parameters
    ----------
    llm_client
        ``LiteLLMClient`` carrying the model name, host, and API key.
    """

    def __init__(self, llm_client: LiteLLMClient | None = None) -> None:
        self._llm = llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(self, rgb_image: np.ndarray, prompt: str) -> SegmentResult | None:
        """
        Locate the object described by *prompt* in *rgb_image*.

        Parameters
        ----------
        rgb_image
            ``(H, W, 3)`` uint8 array in **RGB** order.
        prompt
            Natural-language description of the target object.

        Returns
        -------
        SegmentResult | None
            Best-matching instance, or *None* when no match is found.
        """
        masks, _ = self.generate_masks(rgb_image, prompt)
        if not masks:
            print("No masks generated", flush=True)

            return None

        results = self._build_results(masks)
        annotated_image = self._draw_overlay(rgb_image, results)
        chosen_id = self._query_llm(annotated_image, prompt)

        if chosen_id < 0 or chosen_id >= len(results):
            return None
        return results[chosen_id]

    # ------------------------------------------------------------------
    # Abstract hooks — subclasses provide the segmentation backend
    # ------------------------------------------------------------------
    @abstractmethod
    def generate_masks(
        self, rgb_image: np.ndarray, prompt: str
    ) -> tuple[list[np.ndarray], list[float]]:
        """
        Run backend segmentation and return ``(masks, scores)``.

        Parameters
        ----------
        rgb_image
            ``(H, W, 3)`` uint8 array in **RGB** order.
        prompt
            Natural-language description of the target object.

        Returns
        -------
        tuple[list[np.ndarray], list[float]]
            Parallel lists of boolean masks ``(H, W)`` and confidence scores,
            sorted by score descending.
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _build_results(self, masks: list[np.ndarray]) -> list[SegmentResult]:
        results: list[SegmentResult] = []
        for idx, mask in enumerate(masks):
            if not mask.any():
                continue
            ys, xs = np.where(mask)
            centroid_px = (int(xs.mean()), int(ys.mean()))
            results.append(SegmentResult(mask=mask, centroid_px=centroid_px, mask_id=idx))
        return results

    def _draw_overlay(
        self, rgb_image: np.ndarray, results: list[SegmentResult]
    ) -> PILImage.Image:
        """Return a copy of *rgb_image* with each mask tinted and labelled."""
        base = PILImage.fromarray(rgb_image).convert("RGBA")

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
            )
        except OSError:
            font = ImageFont.load_default()

        for result in results:
            colour = _MASK_COLOURS[result.mask_id % len(_MASK_COLOURS)]

            overlay = PILImage.new("RGBA", base.size, (0, 0, 0, 0))
            colour_layer = PILImage.new("RGBA", base.size, colour)
            mask_pil = PILImage.fromarray(result.mask.astype(np.uint8) * 255, mode="L")
            overlay.paste(colour_layer, mask=mask_pil)
            base = PILImage.alpha_composite(base, overlay)

            draw = ImageDraw.Draw(base)
            u, v = result.centroid_px
            label = str(result.mask_id)
            bbox = draw.textbbox((u, v), label, font=font, anchor="mm")
            draw.rectangle(bbox, fill=(0, 0, 0, 180))
            draw.text((u, v), label, fill=(255, 255, 255, 255), font=font, anchor="mm")

        return base.convert("RGB")

    def _image_to_base64(self, image: PILImage.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _query_llm(self, annotated_image: PILImage.Image, prompt: str) -> int:
        """Send the annotated image to the LLM and return the chosen mask ID."""
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(prompt=prompt)
        b64 = self._image_to_base64(annotated_image)

        response = litellm.completion(
            model=self._llm.model,
            api_base=self._llm.api_base,
            api_key=self._llm.api_key,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                        {
                            "type": "text",
                            "text": (
                                "Based on the labelled regions in this image, "
                                f'which mask ID corresponds to: "{prompt}"? '
                                "Reply with JSON only."
                            ),
                        },
                    ],
                },
            ],
        )

        raw = response.choices[0].message.content or ""
        return self._parse_mask_id(raw)

    @staticmethod
    def _parse_mask_id(raw: str) -> int:
        """Extract mask_id from the LLM JSON reply. Returns -1 on failure."""
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            data = json.loads(cleaned)
            return int(data["mask_id"])
        except (json.JSONDecodeError, KeyError, ValueError):
            match = re.search(r'"mask_id"\s*:\s*(-?\d+)', cleaned)
            if match:
                return int(match.group(1))
            return -1
