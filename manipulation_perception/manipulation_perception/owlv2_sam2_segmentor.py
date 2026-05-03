"""
OWLv2 + SAM2 backend for PromptToSegment.
No ROS dependency — pure numpy/PIL/torch.
"""
from __future__ import annotations

import logging
import numpy as np
import torch
from PIL import Image as PILImage
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from .sam2_segmentor import Sam2Segmentor, _DEFAULT_CHECKPOINT, _DEFAULT_MODEL_CFG


def _nms(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """Greedy NMS — returns indices of kept boxes sorted by score descending."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    order = scores.argsort()[::-1]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_threshold]

    return np.array(keep, dtype=np.int64)


class Owlv2Sam2Segmentor(Sam2Segmentor):
    """
    Sam2Segmentor variant that replaces the LLM bounding-box step with OWLv2
    zero-shot object detection, then feeds those boxes into SAM2 for mask
    refinement.

    Parameters
    ----------
    checkpoint
        Path to the SAM2 ``.pt`` weights file.
    model_cfg
        SAM2 model config path.
    device
        ``"cuda"`` or ``"cpu"``.
    owlv2_model
        HuggingFace model ID for OWLv2.
    detection_threshold
        Minimum OWLv2 confidence score to keep a detection.
    nms_threshold
        IoU threshold for non-maximum suppression. Lower values suppress more
        aggressively; 0.3 is a good default for single-instance objects.
    """

    def __init__(
        self,
        checkpoint: str = _DEFAULT_CHECKPOINT,
        model_cfg: str = _DEFAULT_MODEL_CFG,
        device: str = "cuda",
        owlv2_model: str = "google/owlv2-base-patch16-ensemble",
        detection_threshold: float = 0.1,
        nms_threshold: float = 0.3,
    ) -> None:
        super().__init__(
            llm_client=None,
            grounding="coord",
            checkpoint=checkpoint,
            model_cfg=model_cfg,
            device=device,
        )
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        self._owlv2_processor = Owlv2Processor.from_pretrained(owlv2_model)
        self._owlv2_model = Owlv2ForObjectDetection.from_pretrained(owlv2_model).to(device)
        self._owlv2_model.eval()
        self._detection_threshold = detection_threshold
        self._nms_threshold = nms_threshold
        self._device = device

    def generate_bboxes(
        self, rgb_image: np.ndarray, prompt: str
    ) -> tuple[list[tuple[float, float, float, float]], str]:
        """Use OWLv2 zero-shot detection to predict bounding boxes.

        Returns ratio-coordinate boxes ``[(rx1, ry1, rx2, ry2), ...]`` sorted
        by confidence descending, and a summary string in place of a raw LLM
        response.
        """
        h, w = rgb_image.shape[:2]
        pil_image = PILImage.fromarray(rgb_image)
        text_labels = [[prompt]]

        inputs = self._owlv2_processor(
            text=text_labels, images=pil_image, return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            outputs = self._owlv2_model(**inputs)

        target_sizes = torch.tensor([(h, w)], device=self._device)
        results = self._owlv2_processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self._detection_threshold,
            text_labels=text_labels,
        )[0]

        boxes_np  = results["boxes"].cpu().numpy()
        scores_np = results["scores"].cpu().numpy()
        kept = _nms(boxes_np, scores_np, self._nms_threshold)

        ratio_boxes: list[tuple[float, float, float, float]] = []
        for (x1, y1, x2, y2) in boxes_np[kept]:
            rx1 = max(0.0, min(1.0, float(x1) / w))
            ry1 = max(0.0, min(1.0, float(y1) / h))
            rx2 = max(0.0, min(1.0, float(x2) / w))
            ry2 = max(0.0, min(1.0, float(y2) / h))
            if rx2 > rx1 and ry2 > ry1:
                ratio_boxes.append((rx1, ry1, rx2, ry2))

        return ratio_boxes, ""
