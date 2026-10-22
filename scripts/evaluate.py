from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from dermai_xai.faithfulness import deletion_auc, insertion_auc
from dermai_xai.gradcam import GradCAM
from dermai_xai.localization import iou, pointing_game_hit, threshold_heatmap
from dermai_xai.metrics import classification_metrics
from dermai_xai.model import DEFAULT_MODEL_ID, HAM10000_CLASSES, load_model, load_processor, module_scorer

LABEL_TO_INDEX = {name: index for index, name in enumerate(HAM10000_CLASSES)}


def load_frame(data_dir: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    metadata_path = data_dir / "HAM10000_metadata.csv"
    if not metadata_path.is_file():
        metadata_path = data_dir / "HAM10000_metadata"
    frame = pd.read_csv(metadata_path)
    mapping: dict[str, Path] = {}
    for name in ("HAM10000_images_part_1", "HAM10000_images_part_2"):
        image_dir = data_dir / name
        if image_dir.is_dir():
            mapping.update({path.stem: path for path in image_dir.glob("*.jpg")})
    mapping.update({path.stem: path for path in data_dir.glob("*.jpg")})
    frame = frame[frame.image_id.isin(mapping)].reset_index(drop=True)
    return frame, mapping


def group_aware_test_index(frame: pd.DataFrame) -> np.ndarray:
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=0)
    _, test_index = next(iter(splitter.split(frame, frame.dx, groups=frame.lesion_id)))
    return test_index


def main() -> None:
    parser = argparse.ArgumentParser(description="dermai EfficientNet-B0 evaluation with Grad-CAM and masks")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--positive-fraction", type=float, default=0.15)
    parser.add_argument("--masks-dir", default=None)
    parser.add_argument("--mask-suffix", default="_segmentation.png")
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    frame, mapping = load_frame(data_dir)
    frame = frame.loc[group_aware_test_index(frame)].reset_index(drop=True)
    if args.limit and args.limit < len(frame):
        frame = frame.sample(args.limit, random_state=0).reset_index(drop=True)

    mask_dir = Path(args.masks_dir) if args.masks_dir else data_dir / "HAM10000_segmentations_lesion_tschandl"
    model = load_model(args.model_id, args.device)
    processor = load_processor(args.model_id)
    scorer = module_scorer(model)
    cam_engine = GradCAM(model)

    labels: list[int] = []
    predicted: list[int] = []
    probas: list[np.ndarray] = []
    iou_values: list[float] = []
    pointing_hits: list[bool] = []
    deletion_values: list[float] = []
    insertion_values: list[float] = []

    for _, row in frame.iterrows():
        image = Image.open(mapping[row.image_id]).convert("RGB")
        batch = processor(image, return_tensors="pt")["pixel_values"].to(next(model.parameters()).device)
        result = torch.softmax(model(pixel_values=batch).logits, dim=-1)
        target = result.argmax(dim=-1)
        labels.append(LABEL_TO_INDEX[row.dx])
        predicted.append(int(target[0]))
        probas.append(result[0].cpu().numpy())

        explanation = cam_engine(batch)
        heatmap = explanation["cam"].detach()

        mask_path = mask_dir / f"{row.image_id}{args.mask_suffix}"
        if mask_path.is_file():
            width, height = batch.shape[-1], batch.shape[-2]
            mask = Image.open(mask_path).convert("L").resize((width, height), Image.NEAREST)
            truth = np.asarray(mask) > 0
            saliency = heatmap[0].cpu().numpy()
            pred_mask = threshold_heatmap(saliency, args.positive_fraction)
            iou_values.append(iou(pred_mask, truth))
            pointing_hits.append(pointing_game_hit(saliency, truth))

        deletion_values.append(float(deletion_auc(scorer, batch, heatmap)[0]))
        insertion_values.append(float(insertion_auc(scorer, batch, heatmap)[0]))

    report: dict = {"model_id": args.model_id, "images": len(labels)}
    report.update(classification_metrics(labels, predicted, np.stack(probas)))
    if iou_values:
        report["iou_mean"] = float(np.mean(iou_values))
        report["iou_std"] = float(np.std(iou_values))
        report["pointing_game_hit_rate"] = float(np.mean(pointing_hits))
        report["mask_pairs"] = len(iou_values)
    report["deletion_auc_mean"] = float(np.mean(deletion_values))
    report["insertion_auc_mean"] = float(np.mean(insertion_values))

    text = json.dumps(report, indent=2)
    print(text)
    if args.save:
        Path(args.save).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
