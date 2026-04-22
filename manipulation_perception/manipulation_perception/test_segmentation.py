"""
Standalone test: load inference.png and run Sam3Segmentor with LLM arbitration.

Usage:
    python test_segmentation.py [--prompt "object description"] [--mode sam3|text] [--device cpu|cuda]
"""
import argparse
import sys
import os

import numpy as np
from PIL import Image

# Allow importing from the manipulation_perception package without installing it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "manipulation_perception"))

from .prompt_to_segment import LiteLLMClient
from .sam3_segmentor import Sam3Segmentor, Sam3TextSegmentor

# IMAGE_PATH = os.path.join(
#     os.path.dirname(__file__),
#     "/home/ros/ros2_ws/src/manipulation_perception/manipulation_perception/inference.png",
# )

import sam3
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

IMAGE_PATH  = f"{sam3_root}/assets/images/test_image.jpg"



def main() -> None:
    parser = argparse.ArgumentParser(description="Test SAM3 segmentation on inference.png")
    parser.add_argument("--prompt",  default="shoe",
                        help="Natural-language description of the target object")
    parser.add_argument("--mode",    choices=["sam3", "text"], default="sam3",
                        help="sam3 = SAM3+LLM arbitration, text = SAM3 text encoder only")
    parser.add_argument("--device",  default="cuda", help="cuda or cpu")
    parser.add_argument("--llm-model",    default="ollama/gemma4:e4b")
    parser.add_argument("--llm-api-base", default="http://localhost:11434")
    parser.add_argument("--output",  default="result.png",
                        help="Path to save the annotated result image")
    args = parser.parse_args()

    print(f"Loading image: {IMAGE_PATH}")
    pil_img = Image.open(IMAGE_PATH).convert("RGB")
    rgb = np.array(pil_img)
    print(f"  Shape: {rgb.shape}, dtype: {rgb.dtype}")

    print(f"Building segmentor (mode={args.mode}, device={args.device}) ...")
    if args.mode == "sam3":
        llm_client = LiteLLMClient(
            model=args.llm_model,
            api_base=args.llm_api_base,
        )
        segmentor = Sam3Segmentor(llm_client=llm_client, device=args.device)
    else:
        segmentor = Sam3TextSegmentor(device=args.device)

    print(f"Running segmentation with prompt: \"{args.prompt}\"")

    # Show all candidate masks before LLM arbitration
    masks, scores = segmentor.generate_masks(rgb, args.prompt)
    print(f"Candidate masks: {len(masks)}, scores: {[f'{s:.3f}' for s in scores]}")
    if masks:
        results = segmentor._build_results(masks)
        overlay = segmentor._draw_overlay(rgb, results)
        overlay_path = args.output.replace(".png", "_overlay.png")
        overlay.save(overlay_path)
        print(f"Candidate overlay saved to: {overlay_path}")

    result = segmentor.segment(rgb, args.prompt)

    if result is None:
        print("No matching segment found.")
        sys.exit(1)

    print(f"Result:")
    print(f"  mask_id   : {result.mask_id}")
    print(f"  centroid  : {result.centroid_px}  (u, v)")
    print(f"  mask area : {result.mask.sum()} px  "
          f"({100 * result.mask.sum() / result.mask.size:.1f}% of image)")

    # Save annotated image (overlay the winning mask)
    from PIL import ImageDraw
    out_img = pil_img.copy().convert("RGBA")
    colour_layer = Image.new("RGBA", out_img.size, (50, 220, 50, 140))
    mask_pil = Image.fromarray(result.mask.astype(np.uint8) * 255, mode="L")
    out_img.paste(colour_layer, mask=mask_pil)
    draw = ImageDraw.Draw(out_img)
    u, v = result.centroid_px
    r = 8
    draw.ellipse([u - r, v - r, u + r, v + r], fill=(255, 50, 50, 255))
    out_img = out_img.convert("RGB")
    out_img.save(args.output)
    print(f"Annotated image saved to: {args.output}")


if __name__ == "__main__":
    main()
