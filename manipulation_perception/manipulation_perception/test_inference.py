#!/usr/bin/env python3
"""
Test inference: runs prompt-to-segment with multiple backend configurations.

Usage (inside Docker):
    python3 test_inference.py [--prompt "mug"] [--image path/to/image.png]

Results are written to ./results/<timestamp>_<prompt>/
  <config_name>.png   — original image with detected masks overlaid
  summary.txt         — text log of all runs
"""
from __future__ import annotations

import argparse
import datetime
import os

import cv2
import numpy as np

from manipulation_perception.prompt_to_segment import LiteLLMClient, draw_masks, draw_bboxes
from manipulation_perception.sam3_segmentor import Sam3Segmentor
from manipulation_perception.sam2_segmentor import Sam2Segmentor
from manipulation_perception.vision_banana import VisionBananaSegmentor, VisionBananaBboxSegmentor

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
IMAGE_PATH = os.path.normpath(os.path.join(_DATA_DIR, "image.png"))

# ---------------------------------------------------------------------------
# Shared LLM client (used by SAM2, VLM, and VisionBanana backends)
# ---------------------------------------------------------------------------

_LLM_IMAGE_GEN = LiteLLMClient(
    model="openai/gemini-3.1-flash-image-preview",
    api_base="http://localhost:4000",
    api_key="sk-1234",
)


_LLM_REASONING = LiteLLMClient(
    model="openai/vllm",
    api_base="http://localhost:4000",
    api_key="sk-1234",
)

# ---------------------------------------------------------------------------
# Configurations — add/remove entries to change what gets tested.
# Uncomment alternative variants to compare different parameter settings.
# ---------------------------------------------------------------------------

CONFIGS: list[dict] = [
    # --- SAM3 (VLM bounding-box → SAM3 geometric prompt) ---
    {
        "name": "sam3_llm",
        "factory": lambda: Sam3Segmentor(llm_client=_LLM_REASONING, device="cuda"),
    },

    {
        "name": "sam3_prompt",
        "factory": lambda: Sam3Segmentor(device="cuda"),

    },

    # --- SAM2 (automatic masks + LLM arbitration) ---
    {
        "name": "sam2",
        "factory": lambda: Sam2Segmentor(llm_client=_LLM_REASONING),

    },
    # --- Vision Banana (generative colour-coded mask) ---
    {
        "name": "vision_banana_default",
        "factory": lambda: VisionBananaSegmentor(llm_client=_LLM_IMAGE_GEN),
    },

    # --- Vision Banana Bbox (VLM bbox → image-gen mask) ---
    {
        "name": "vision_banana_bbox",
        "factory": lambda: VisionBananaBboxSegmentor(
            llm_client=_LLM_REASONING,
            image_gen_client=_LLM_IMAGE_GEN,
        ),
    },

]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def run_config(
    cfg: dict,
    rgb_image: np.ndarray,
    prompt: str,
    out_dir: str,
    log_lines: list[str],
) -> None:
    name = cfg["name"]
    header = f"\n{'=' * 60}\nConfig : {name}\nPrompt : {prompt!r}\n{'=' * 60}"
    print(header)
    log_lines.append(header)

    segmentor = cfg["factory"]()
    results = segmentor.segment(rgb_image, prompt)
    dbg = segmentor.debug
    elapsed = f"{dbg.inference_time_s:.2f}s" if dbg is not None else "?"

    if not results:
        line = f"  No instances found  ({elapsed})"
        print(line)
        log_lines.append(line)
    else:
        line = f"  Found {len(results)} instance(s)  ({elapsed})"
        print(line)
        log_lines.append(line)
        for i, r in enumerate(results):
            area = int(r.mask.sum())
            line = f"  [{i}] centroid={r.centroid_px}  area={area}px  mask_ids={r.mask_ids}"
            print(line)
            log_lines.append(line)

    annotated_rgb = draw_masks(rgb_image, results, contour_thickness=6, font_scale=2.0, show_index=False)
    out_path = os.path.join(out_dir, f"{name}_segmentation_results.png")
    cv2.imwrite(out_path, cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
    saved_line = f"  Saved: {out_path}"
    print(saved_line)
    log_lines.append(saved_line)

    if dbg is not None:
        timing = (f"  vlm={dbg.vlm_time_s:.3f}s  "
                  f"seg={dbg.segmentation_time_s:.3f}s  "
                  f"total={dbg.inference_time_s:.3f}s")
        print(timing)
        log_lines.append(timing)

        if dbg.vlm_input_image is not None:
            grid_path = os.path.join(out_dir, f"{name}_llm_input.png")
            cv2.imwrite(grid_path, cv2.cvtColor(dbg.vlm_input_image, cv2.COLOR_RGB2BGR))
            print(f"  Saved: {grid_path}")
            log_lines.append(f"  Saved: {grid_path}")

        if dbg.vlm_raw_output is not None:
            llm_path = os.path.join(out_dir, f"{name}_llm_output.txt")
            with open(llm_path, "w") as f:
                f.write(dbg.vlm_raw_output)
            print(f"  Saved: {llm_path}")
            log_lines.append(f"  Saved: {llm_path}")

        if dbg.bbox_prompt:
            boxes_path = os.path.join(out_dir, f"{name}_llm_boxes.png")
            cv2.imwrite(boxes_path, cv2.cvtColor(draw_bboxes(rgb_image, dbg.bbox_prompt), cv2.COLOR_RGB2BGR))
            print(f"  Saved: {boxes_path}")
            log_lines.append(f"  Saved: {boxes_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Test segmentation inference")
    parser.add_argument("--prompt", default="toy fork", help="Object description to segment")
    parser.add_argument("--image", default=IMAGE_PATH, help="Path to the input image")
    args = parser.parse_args()

    rgb_image = load_image(args.image)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = args.prompt.replace(" ", "_")[:40]
    out_dir = os.path.join("results", f"{timestamp}_{safe_prompt}")
    os.makedirs(out_dir, exist_ok=True)

    log_lines: list[str] = []
    header = f"Image  : {args.image}  shape={rgb_image.shape}"
    print(header)
    log_lines.append(header)

    for cfg in CONFIGS:
        try:
            run_config(cfg, rgb_image, args.prompt, out_dir, log_lines)
        except Exception as exc:
            line = f"\n  ERROR in {cfg['name']}: {exc}"
            print(line)
            log_lines.append(line)

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\nResults written to: {out_dir}/")


if __name__ == "__main__":
    main()
