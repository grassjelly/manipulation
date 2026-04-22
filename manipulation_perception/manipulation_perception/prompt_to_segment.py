"""
Abstract interface for prompt-driven instance segmentation.
No ROS dependency — pure numpy/cv2.
"""
from __future__ import annotations

import base64
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np
import litellm
import logging

_MASK_COLOURS: list[tuple[int, int, int]] = [
    (220,  50,  50),  # red
    ( 50, 180,  50),  # green
    ( 50, 100, 220),  # blue
    (220, 180,  50),  # yellow
    (180,  50, 220),  # purple
    ( 50, 210, 200),  # cyan
    (230, 120,  50),  # orange
    (200,  50, 130),  # pink
]

_SYSTEM_PROMPT_TEMPLATE = (
    "You are a vision assistant helping a robot manipulation system identify "
    "objects in a scene.\n\n"
    "Context: The image has been pre-processed for instance segmentation. "
    "Each detected region is outlined with a distinct colour and labelled with "
    "a numeric mask ID on a white background.  A single physical object may be "
    "split across multiple regions (e.g. separate tines of a fork, a handle "
    "and a head).\n\n"
    "Task: {prompt}\n\n"
    "Instructions:\n"
    "- Examine the labelled regions carefully.\n"
    "- Find EVERY separate physical instance of the described object.\n"
    "- For each instance, list ALL mask IDs whose regions together form that "
    "one physical object.  Fragments that clearly belong together (adjacent, "
    "same object, same colour context) should be grouped under the same "
    "instance entry.\n"
    "- Verify each candidate annotation matches the target object by checking:\n"
    "  * Size: the annotated region should be proportional to a real instance "
    "of the object — reject regions that are far too small (e.g. a stray pixel "
    "cluster) or far too large.\n"
    "  * Silhouette / shape: the outline of the annotated region should match "
    "the expected contour of the object (e.g. elongated for a knife, roughly "
    "circular for a ball).\n"
    "  * Completeness: prefer groupings where the merged silhouette best "
    "resembles the whole object rather than a fragment.\n"
    "- Only include a mask ID if you are confident the annotation genuinely "
    "corresponds to the requested object after the above checks.\n"
    "- Be aware of nearby objects: if an annotated region visually overlaps "
    "with or bleeds into a different object (e.g. a background item, an "
    "adjacent unrelated object), discard that region entirely — do not include "
    "it even as part of a grouped instance.\n"
    "- When in doubt between two candidate regions, prefer the one whose "
    "boundary is cleanest and most tightly contained within the target object.\n"
    "- If there are no matching objects at all, return an empty instances list.\n\n"
    "Reply ONLY with valid JSON in exactly this format:\n"
    '{{"instances": [{{"mask_ids": [<int>, ...], "confidence": "<high|medium|low>", '
    '"reasoning": "<one sentence>"}}]}}'
)


@dataclass
class LiteLLMClient:
    """Connection config passed to ``PromptToSegment`` for LLM arbitration."""
    model: str   = "ollama/gemma4:e4b"
    api_base: str = "http://localhost:11434"
    api_key: str  = "sk-1234"


@dataclass
class SegmentResult:
    mask: np.ndarray              # bool (H, W) — union of all chosen mask regions
    centroid_px: tuple[int, int]  # (u, v) pixel coords of the merged mask centroid
    mask_ids: list[int]           # indices of all mask regions that form this instance


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

    def segment(self, rgb_image: np.ndarray, prompt: str) -> list[SegmentResult]:
        """
        Locate all instances of the object described by *prompt* in *rgb_image*.

        Parameters
        ----------
        rgb_image
            ``(H, W, 3)`` uint8 array in **RGB** order.
        prompt
            Natural-language description of the target object.

        Returns
        -------
        list[SegmentResult]
            One entry per detected instance; empty when no match is found.
            Each instance's mask is the union of all its constituent regions.
        """
        t0 = time.perf_counter()
        masks, _ = self.generate_masks(rgb_image, prompt)
        print(f"[segment] generate_masks: {time.perf_counter() - t0:.3f}s", flush=True)
        if not masks:
            print("No masks generated", flush=True)
            return []

        results = self._build_results(masks)
        annotated_image = self._draw_overlay(rgb_image, results)
        _save_path = "/home/ros/ros2_ws/src/debug_overlay.jpg"
        cv2.imwrite(_save_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        print(f"[segment] overlay saved to {_save_path}", flush=True)
        t0 = time.perf_counter()
        instances_ids = self._query_llm(annotated_image, prompt)
        print(f"[segment] _query_llm: {time.perf_counter() - t0:.3f}s", flush=True)

        segment_results: list[SegmentResult] = []
        for id_group in instances_ids:
            valid_ids = [i for i in id_group if 0 <= i < len(results)]
            if not valid_ids:
                continue
            merged_mask = np.zeros(rgb_image.shape[:2], dtype=bool)
            for i in valid_ids:
                merged_mask |= results[i].mask
            ys, xs = np.where(merged_mask)
            centroid_px = (int(xs.mean()), int(ys.mean()))
            segment_results.append(
                SegmentResult(mask=merged_mask, centroid_px=centroid_px, mask_ids=valid_ids)
            )
        return segment_results

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

    def _build_results(
        self, masks: list[np.ndarray], min_mask_region_area: int = 100
    ) -> list[SegmentResult]:
        results: list[SegmentResult] = []
        for idx, mask in enumerate(masks):
            if mask.sum() < min_mask_region_area:
                continue
            ys, xs = np.where(mask)
            centroid_px = (int(xs.mean()), int(ys.mean()))
            results.append(SegmentResult(mask=mask, centroid_px=centroid_px, mask_ids=[idx]))
        return results

    def _draw_overlay(
        self, rgb_image: np.ndarray, results: list[SegmentResult]
    ) -> np.ndarray:
        """Return a copy of *rgb_image* with each mask outlined and labelled."""
        canvas = rgb_image.copy()

        for result in results:
            mask_id = result.mask_ids[0]
            colour = _MASK_COLOURS[mask_id % len(_MASK_COLOURS)]
            mask_u8 = result.mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, colour, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        thickness = 1
        pad = 1
        for result in results:
            mask_id = result.mask_ids[0]
            colour = _MASK_COLOURS[mask_id % len(_MASK_COLOURS)]
            u, v = result.centroid_px
            label = str(mask_id)
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            x0 = u - tw // 2 - pad
            y0 = v - th // 2 - pad
            x1 = u + tw // 2 + pad
            y1 = v + th // 2 + pad + baseline
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 255, 255), cv2.FILLED)
            cv2.putText(canvas, label, (u - tw // 2, v + th // 2),
                        font, font_scale, colour, thickness, cv2.LINE_AA)

        return canvas

    def _image_to_base64(self, image: np.ndarray) -> str:
        # canvas is RGB; imencode expects BGR
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def _query_llm(self, annotated_image: np.ndarray, prompt: str) -> list[list[int]]:
        """Send the annotated image to the LLM; return mask-ID groups per instance."""
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
                                "How many separate instances of "
                                f'"{prompt}" are visible?  '
                                "For each instance list all mask IDs that "
                                "belong to it.  Reply with JSON only."
                            ),
                        },
                    ],
                },
            ],
        )

        raw = response.choices[0].message.content or ""
        return self._parse_instances(raw)

    @staticmethod
    def _parse_instances(raw: str) -> list[list[int]]:
        """Parse the LLM reply into a list of per-instance mask-ID groups."""
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            data = json.loads(cleaned)
            entries = data["instances"]
            if not isinstance(entries, list):
                return []
            result: list[list[int]] = []
            for entry in entries:
                ids = entry.get("mask_ids", [])
                if isinstance(ids, list):
                    result.append([int(i) for i in ids])
                else:
                    result.append([int(ids)])
            return result
        except (json.JSONDecodeError, KeyError, ValueError):
            return []
