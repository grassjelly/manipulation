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
from dataclasses import dataclass, field

import cv2
import numpy as np
import litellm


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


def draw_grid(rgb_image: np.ndarray, n: int = 10) -> np.ndarray:
    """Return a copy of *rgb_image* with a subtle n×n ratio-coordinate grid overlaid.

    Grid lines are drawn at every 1/n interval in both axes.  Each intersection
    is labelled with its ratio coordinate ``rx,ry`` so a vision model can use
    the grid as a visual reference when describing object positions.
    """
    canvas = rgb_image.copy()
    h, w = canvas.shape[:2]

    line_colour       = (180, 180, 180)
    text_colour       = (0, 0, 0)
    outline_colour    = (255, 255, 255)
    font              = cv2.FONT_HERSHEY_SIMPLEX
    font_scale        = 0.28
    thickness         = 1
    outline_thickness = thickness + 2
    pad               = 3

    grid_layer = canvas.copy()
    for i in range(n + 1):
        t  = i / n
        gx = int(t * (w - 1))
        gy = int(t * (h - 1))
        cv2.line(grid_layer, (gx, 0),      (gx, h - 1), line_colour, 1)
        cv2.line(grid_layer, (0,  gy),     (w - 1, gy), line_colour, 1)
    cv2.addWeighted(grid_layer, 0.5, canvas, 0.5, 0, canvas)

    for iy in range(n + 1):
        for ix in range(n + 1):
            rx = ix / n
            ry = iy / n
            px = int(rx * (w - 1))
            py = int(ry * (h - 1))
            label = f"{rx:.1f},{ry:.1f}"
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            tx = px + pad if px + pad + tw < w else px - tw - pad
            ty = py + th + pad if py + th + pad < h else py - pad
            cv2.putText(canvas, label, (tx, ty),
                        font, font_scale, outline_colour, outline_thickness, cv2.LINE_AA)
            cv2.putText(canvas, label, (tx, ty),
                        font, font_scale, text_colour, thickness, cv2.LINE_AA)

    return canvas


_MAX_SEND_DIM = 1024  # longest side sent to the VLM for coord grounding

_BBOX_SYSTEM_PROMPT = (
    "You are a vision assistant helping a robot manipulation system.\n\n"
    'Task: Locate every separate physical instance of "{prompt}" in the image.\n\n'
    "The image has a reference grid overlaid on it.  Each grid line intersection "
    "is labelled with its ratio coordinate in the form rx,ry, where rx is the "
    "fraction of the image width (0.0 = left edge, 0.99 = right edge) and ry is "
    "the fraction of the image height (0.0 = top edge, 0.99 = bottom edge).\n\n"
    "Instructions:\n"
    "- For each instance, draw a tight axis-aligned bounding box expressed as "
    "[rx1, ry1, rx2, ry2] where (rx1, ry1) is the top-left corner and "
    "(rx2, ry2) is the bottom-right corner — all values in [0.0, 0.99].\n"
    "- Use the labelled grid intersections as landmarks to estimate corners accurately.\n"
    "- Only include an instance if you are confident it matches the request.\n"
    "- If there are no matching objects, return an empty instances list.\n\n"
    "Reply ONLY with valid JSON in exactly this format:\n"
    '{{"instances": [{{"bbox": [<rx1>, <ry1>, <rx2>, <ry2>], '
    '"confidence": "<high|medium|low>", "reasoning": "<one sentence>"}}]}}'
)

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


@dataclass
class SegmentDebug:
    """Diagnostic snapshot populated after every ``segment()`` call."""
    vlm_text_prompt: str = ""
    vlm_input_image: np.ndarray | None = None          # image sent to the VLM
    vlm_raw_output: str | None = None                  # raw JSON returned by the VLM
    bbox_prompt: list[tuple[float, float, float, float]] = field(default_factory=list)
    vlm_time_s: float = 0.0           # VLM call (box prediction or SOM arbitration)
    segmentation_time_s: float = 0.0  # mask-generation backend
    inference_time_s: float = 0.0     # total end-to-end


def draw_masks(
    rgb_image: np.ndarray,
    results: list[SegmentResult],
    contour_thickness: int = 2,
    font_scale: float = 0.5,
    show_index: bool = True,
) -> np.ndarray:
    """Return a copy of *rgb_image* with each mask outlined and optionally labelled."""
    canvas = rgb_image.copy()

    for result in results:
        mask_id = result.mask_ids[0]
        colour = _MASK_COLOURS[mask_id % len(_MASK_COLOURS)]
        mask_u8 = result.mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, colour, contour_thickness)

    if show_index:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_thickness = max(1, round(font_scale * 2))
        pad = 1
        for result in results:
            mask_id = result.mask_ids[0]
            colour = _MASK_COLOURS[mask_id % len(_MASK_COLOURS)]
            u, v = result.centroid_px
            label = str(mask_id)
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
            x0 = u - tw // 2 - pad
            y0 = v - th // 2 - pad
            x1 = u + tw // 2 + pad
            y1 = v + th // 2 + pad + baseline
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 255, 255), cv2.FILLED)
            cv2.putText(canvas, label, (u - tw // 2, v + th // 2),
                        font, font_scale, colour, text_thickness, cv2.LINE_AA)

    return canvas


def draw_bboxes(
    rgb_image: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    thickness: int = 3,
    font_scale: float = 0.7,
    show_index: bool = True,
    colours: list[tuple[int, int, int]] | None = None,
    index_offset: int = 0,
) -> np.ndarray:
    """Return a copy of *rgb_image* with each ratio bounding box drawn.

    Parameters
    ----------
    colours
        Colour palette to cycle through.  Defaults to ``_MASK_COLOURS``.
    index_offset
        Added to the box index when forming the label (e.g. ``1`` for 1-based).
    """
    palette = colours if colours is not None else _MASK_COLOURS
    h, w = rgb_image.shape[:2]
    canvas = rgb_image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_thickness = max(1, round(font_scale * 2))

    for idx, (rx1, ry1, rx2, ry2) in enumerate(boxes):
        x1, y1 = int(rx1 * w), int(ry1 * h)
        x2, y2 = int(rx2 * w), int(ry2 * h)
        colour = palette[idx % len(palette)]

        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)

        if show_index:
            label = str(idx + index_offset)
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
            pad = 4
            lx, ly = x1, y1 - pad
            if ly - th < 0:
                ly = y1 + th + pad
            cv2.rectangle(canvas, (lx - pad, ly - th - pad), (lx + tw + pad, ly + pad),
                          (255, 255, 255), cv2.FILLED)
            cv2.putText(canvas, label, (lx, ly), font, font_scale, colour, text_thickness, cv2.LINE_AA)

    return canvas


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
        self.debug: SegmentDebug | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @abstractmethod
    def segment(
        self,
        rgb_image: np.ndarray,
        prompt: str,
    ) -> list[SegmentResult]:
        """
        Locate all instances of the object described by *prompt* in *rgb_image*.

        Subclasses must implement this method and choose which grounding
        technique to use (e.g. call ``segment_som`` or ``segment_by_coord``).

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
        """

    def segment_som(self, rgb_image: np.ndarray, prompt: str) -> list[SegmentResult]:
        """Set-of-marks grounding: numbered mask overlay + LLM arbitration."""
        t_total = time.perf_counter()

        t_seg = time.perf_counter()
        masks, _ = self.generate_masks(rgb_image)
        segmentation_time = time.perf_counter() - t_seg
        if not masks:
            self.debug = SegmentDebug(
                vlm_text_prompt=prompt,
                segmentation_time_s=segmentation_time,
                inference_time_s=time.perf_counter() - t_total,
            )
            return []

        results = self.build_results(masks)
        annotated_image = draw_masks(rgb_image, results)

        t_vlm = time.perf_counter()
        instances_ids, vlm_raw = self._query_llm(annotated_image, prompt)
        vlm_time = time.perf_counter() - t_vlm

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

        self.debug = SegmentDebug(
            vlm_text_prompt=prompt,
            vlm_input_image=annotated_image,
            vlm_raw_output=vlm_raw,
            vlm_time_s=vlm_time,
            segmentation_time_s=segmentation_time,
            inference_time_s=time.perf_counter() - t_total,
        )
        return segment_results

    def segment_by_coord(self, rgb_image: np.ndarray, prompt: str) -> list[SegmentResult]:
        """Coordinate/bbox grounding: VLM predicts boxes, backend refines them into masks."""
        t_total = time.perf_counter()

        h, w = rgb_image.shape[:2]
        scale  = min(_MAX_SEND_DIM / max(h, w), 1.0)
        send_w = int(w * scale)
        send_h = int(h * scale)
        send_img = (cv2.resize(rgb_image, (send_w, send_h), interpolation=cv2.INTER_AREA)
                    if scale < 1.0 else rgb_image.copy())
        send_img = draw_grid(send_img)

        t_vlm = time.perf_counter()
        boxes, vlm_raw = self._ask_vlm_for_boxes(send_img, prompt)
        vlm_time = time.perf_counter() - t_vlm

        if not boxes:
            self.debug = SegmentDebug(
                vlm_text_prompt=prompt,
                vlm_input_image=send_img,
                vlm_raw_output=vlm_raw,
                vlm_time_s=vlm_time,
                inference_time_s=time.perf_counter() - t_total,
            )
            return []

        t_seg = time.perf_counter()
        masks, _ = self.generate_masks_from_bbox(rgb_image, boxes)
        segmentation_time = time.perf_counter() - t_seg

        self.debug = SegmentDebug(
            vlm_text_prompt=prompt,
            vlm_input_image=send_img,
            vlm_raw_output=vlm_raw,
            bbox_prompt=boxes,
            vlm_time_s=vlm_time,
            segmentation_time_s=segmentation_time,
            inference_time_s=time.perf_counter() - t_total,
        )

        if not masks:
            return []
        return self.build_results(masks)

    # ------------------------------------------------------------------
    # Abstract hooks — subclasses provide the segmentation backend
    # ------------------------------------------------------------------

    def generate_masks(
        self, rgb_image: np.ndarray
    ) -> tuple[list[np.ndarray], list[float]]:
        """
        SOM backend: generate many candidate masks from the image alone.

        Returns parallel lists of boolean masks ``(H, W)`` and confidence
        scores, sorted by score descending.  Used by ``segment_som``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement generate_masks."
        )

    def generate_masks_from_bbox(
        self,
        rgb_image: np.ndarray,
        boxes: list[tuple[float, float, float, float]],
    ) -> tuple[list[np.ndarray], list[float]]:
        """
        Coord backend: apply geometric prompts to produce one mask per box.

        Parameters
        ----------
        rgb_image
            ``(H, W, 3)`` uint8 array in **RGB** order.
        boxes
            VLM-predicted bounding boxes as ``[(rx1, ry1, rx2, ry2), ...]``
            in ratio coordinates (0.0–1.0).  One box per detected instance.

        Returns parallel lists of boolean masks ``(H, W)`` and confidence
        scores, one entry per box.  Used by ``segment_by_coord``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement generate_masks_from_bbox."
        )

    def generate_masks_from_prompt(
        self, rgb_image: np.ndarray, prompt: str
    ) -> tuple[list[np.ndarray], list[float]]:
        """
        Text-prompt backend: generate masks directly from a text prompt,
        bypassing LLM arbitration entirely.

        Returns parallel lists of boolean masks ``(H, W)`` and confidence
        scores, sorted by score descending.  Called by ``segment`` when no
        ``llm_client`` is configured.  Subclasses that support native text
        prompting should override this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement generate_masks_from_prompt."
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def build_results(
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

    def _image_to_base64(self, image: np.ndarray) -> str:
        # canvas is RGB; imencode expects BGR
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def _query_llm(
        self, annotated_image: np.ndarray, prompt: str
    ) -> tuple[list[list[int]], str]:
        """Send the annotated image to the LLM; return (mask-ID groups, raw response)."""
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
        return self._parse_instances(raw), raw

    def _ask_vlm_for_boxes(
        self, grid_image: np.ndarray, prompt: str
    ) -> tuple[list[tuple[float, float, float, float]], str]:
        """Send the grid-overlaid image to the VLM; return (ratio bounding boxes, raw response)."""
        system_prompt = _BBOX_SYSTEM_PROMPT.format(prompt=prompt)
        b64 = self._image_to_base64(grid_image)

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
                                f'How many separate instances of "{prompt}" are visible? '
                                "Draw a tight bounding box around each one using ratio "
                                "coordinates [rx1, ry1, rx2, ry2] as shown by the grid. "
                                "Reply with JSON only."
                            ),
                        },
                    ],
                },
            ],
        )

        raw = response.choices[0].message.content or ""
        boxes = self._parse_boxes(raw)
        return boxes, raw

    @staticmethod
    def _parse_boxes(raw: str) -> list[tuple[float, float, float, float]]:
        """Parse the VLM reply into a list of ``(rx1, ry1, rx2, ry2)`` ratio boxes."""
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        decoder = json.JSONDecoder()
        data = None
        pos = 0
        while pos < len(cleaned):
            try:
                obj, end = decoder.raw_decode(cleaned, pos)
                if isinstance(obj, dict) and "instances" in obj:
                    data = obj
                pos = end
            except json.JSONDecodeError:
                pos += 1

        if data is None:
            return []

        boxes: list[tuple[float, float, float, float]] = []
        for entry in data.get("instances", []):
            if not isinstance(entry, dict):
                continue
            b = entry.get("bbox", [])
            if not isinstance(b, (list, tuple)) or len(b) != 4:
                continue
            try:
                rx1, ry1, rx2, ry2 = (float(v) for v in b)
            except (TypeError, ValueError):
                continue
            rx1, ry1, rx2, ry2 = (
                max(0.0, min(1.0, rx1)),
                max(0.0, min(1.0, ry1)),
                max(0.0, min(1.0, rx2)),
                max(0.0, min(1.0, ry2)),
            )
            if rx2 <= rx1 or ry2 <= ry1:
                continue
            boxes.append((rx1, ry1, rx2, ry2))
        return boxes

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
