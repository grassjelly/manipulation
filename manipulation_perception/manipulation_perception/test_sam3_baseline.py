"""
Baseline replication of cells 1–8 (+ model build) from:
  facebookresearch/sam3/blob/main/examples/sam3_image_predictor_example.ipynb

Run with:
  ros2 run manipulation_perception test_sam3_baseline
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import sam3
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

COLORS = [
    (0.9, 0.2, 0.2),
    (0.2, 0.7, 0.2),
    (0.2, 0.4, 0.9),
    (0.9, 0.7, 0.1),
    (0.7, 0.2, 0.9),
    (0.1, 0.8, 0.8),
    (0.9, 0.5, 0.1),
    (0.5, 0.9, 0.3),
]


def plot_results(img, results):
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    ax = plt.gca()
    nb_objects = len(results["scores"])
    print(f"found {nb_objects} object(s)")
    for i in range(nb_objects):
        color = COLORS[i % len(COLORS)]
        # overlay mask
        mask = results["masks"][i].squeeze(0).cpu().numpy()
        mask_rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        mask_rgba[..., :3] = color
        mask_rgba[..., 3] = mask * 0.5
        ax.imshow(mask_rgba)
        # draw bounding box (XYXY, absolute coords)
        x1, y1, x2, y2 = results["boxes"][i].cpu().tolist()
        prob = results["scores"][i].item()
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 5, f"id={i}, prob={prob:.2f}",
            color=color, weight="bold", fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.75, "pad": 2},
        )

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

# Cell 8 / 9 — Build model
bpe_path = None
model = build_sam3_image_model(bpe_path=bpe_path)

# Cell 10 — Load image and create processor
image_path = f"{sam3_root}/assets/images/test_image.jpg"
image = Image.open(image_path)
width, height = image.size
processor = Sam3Processor(model, confidence_threshold=0.5)
inference_state = processor.set_image(image)


def main():
    print(f"SAM3 loaded successfully.")
    print(f"  sam3_root : {sam3_root}")
    print(f"  image     : {image_path}  ({width}x{height})")
    print(f"  CUDA      : {torch.cuda.is_available()}")

    processor.reset_all_prompts(inference_state)
    inference_state_out = processor.set_text_prompt(state=inference_state, prompt="shoe")

    img0 = Image.open(image_path)
    plot_results(img0, inference_state_out)
    plt.savefig("sam3_baseline_output.png")
    print("Saved result to sam3_baseline_output.png")


if __name__ == "__main__":
    main()
