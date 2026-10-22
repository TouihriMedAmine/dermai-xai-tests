from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from dermai_xai.gradcam import GradCAM
from dermai_xai.localization import iou, threshold_heatmap
from dermai_xai.model import DEFAULT_MODEL_ID, HAM10000_CLASSES, load_model, load_processor
from scripts.evaluate import load_frame

LABEL_TO_INDEX = {name: index for index, name in enumerate(HAM10000_CLASSES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grad-CAM gallery: 10 random HAM10000 images, originals next to overlays")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default=None)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--target", choices=["predicted", "true"], default="predicted")
    parser.add_argument("--positive-fraction", type=float, default=0.15)
    parser.add_argument("--masks-dir", default=None)
    parser.add_argument("--output", default="outputs/gradcam_gallery.png")
    parser.add_argument("--no-open", action="store_true")
    return parser.parse_args()


def mask_overlay_values(mask_dir: Path, image_id: str, width: int, height: int):
    mask_path = mask_dir / f"{image_id}_segmentation.png"
    if not mask_path.is_file():
        return None
    mask = Image.open(mask_path).convert("L").resize((width, height), Image.NEAREST)
    return np.asarray(mask) > 0


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    frame, mapping = load_frame(data_dir)
    sampled = frame.sample(min(args.count, len(frame)), random_state=args.seed).reset_index(drop=True)

    mask_dir = Path(args.masks_dir) if args.masks_dir else data_dir / "HAM10000_segmentations_lesion_tschandl"
    model = load_model(args.model_id, args.device)
    processor = load_processor(args.model_id)
    cam_engine = GradCAM(model)
    colormap = matplotlib.colormaps["turbo"]

    records = []
    fig, axes = plt.subplots(len(sampled), 2, figsize=(5.2, 2.7 * len(sampled)))
    axes = np.atleast_2d(axes)

    for index, row in sampled.iterrows():
        image = Image.open(mapping[row.image_id]).convert("RGB")
        resized = image.resize((224, 224))
        batch = processor(resized, return_tensors="pt")["pixel_values"].to(next(model.parameters()).device)
        with torch.no_grad():
            probabilities = torch.softmax(model(pixel_values=batch).logits, dim=-1)
        predicted = int(probabilities.argmax(dim=-1))

        true_label = row.dx
        target_label = predicted if args.target == "predicted" else LABEL_TO_INDEX[true_label]
        explanation = cam_engine(batch, torch.tensor([target_label], device=batch.device))
        cam = explanation["cam"][0].detach().float().cpu().numpy()
        confidence = float(probabilities[0, predicted].cpu())

        heat_rgb = (colormap(cam)[..., :3] * 255).astype(np.uint8)
        overlay = (args.alpha * heat_rgb + (1.0 - args.alpha) * np.asarray(resized)).astype(np.uint8)

        truth = mask_overlay_values(mask_dir, row.image_id, 224, 224)
        localization_text = "no mask"
        if truth is not None:
            localization_text = f"IoU {iou(threshold_heatmap(cam, args.positive_fraction), truth):.2f}"

        row_axes = axes[0] if len(sampled) == 1 else axes[index]
        correct = predicted == LABEL_TO_INDEX.get(true_label, -1)
        label_color = "green" if correct else "red"
        row_axes[0].imshow(resized)
        row_axes[0].set_title(f"{row.image_id} | true: {true_label} | pred: {HAM10000_CLASSES[predicted]}", color=label_color, fontsize=9)
        row_axes[1].imshow(overlay)
        row_axes[1].set_title(f"Grad-CAM ({args.target}, p={confidence:.2f}) | {localization_text}", color=label_color, fontsize=9)
        row_axes[0].axis("off")
        row_axes[1].axis("off")

        records.append({
            "image_id": row.image_id,
            "true": true_label,
            "pred": HAM10000_CLASSES[predicted],
            "confidence": round(confidence, 3),
            "localization": localization_text,
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    print(pd.DataFrame(records).to_string(index=False))
    print(f"gallery saved to {output_path.resolve()}")
    if not args.no_open:
        try:
            os.startfile(output_path.resolve())
        except AttributeError:
            print(f"open {output_path.resolve()} manually")


if __name__ == "__main__":
    main()
