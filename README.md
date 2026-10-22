# dermai-xai-tests

Independent model and test suite for explainable AI on skin lesion classification with the Apache-2.0 checkpoint [`sgonzalez2000/dermai-efficientnet-b0`](https://huggingface.co/sgonzalez2000/dermai-efficientnet-b0) (EfficientNet-B0 fine-tuned on HAM10000, 7 diagnostic classes).

This is a clean-room implementation: no code is copied from the upstream [SebastianGonzalez2000/dermai-explainability](https://github.com/SebastianGonzalez2000/dermai-explainability) repository (which carries no code licence). The repo follows the evaluation protocol established in the skin-cancer-xai research: lesion-grouped splits, imbalance-aware metrics, Grad-CAM plus quantitative faithfulness and localization evaluation against HAM10000 masks.

## What is tested

| Suite | Scope | Network |
|-------|-------|---------|
| `test_localization.py` | IoU, thresholded heatmap masks, pointing game on synthetic masks | no |
| `test_faithfulness.py` | Deletion and insertion AUC with a mock scorer (matching heatmap must beat a mismatched one) | no |
| `test_metrics.py` | Balanced accuracy, macro F1, ROC-AUC, PR-AUC, ECE on controlled cases | no |
| `test_gradcam.py` | Grad-CAM on a tiny random EfficientNet-shaped model: hooks, shapes, normalization | no |
| `test_model.py` | Real checkpoint: 7-class logits, label mapping, probabilities, Grad-CAM and faithfulness on the trained model | yes |

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu  # or your CUDA build
pip install -r requirements.txt
```

## Run tests

```bash
pytest -m "not network"   # offline suite, seconds
pytest                    # full suite, downloads the ~16 MB checkpoint once and caches it
```

## Evaluate the model

```bash
python scripts/evaluate.py --data-dir data --limit 200                        # random subset of the test split
python scripts/evaluate.py --save report.json                                 # full 30% test split with masks
```

The script expects the HAM10000 layout produced by the upstream download:

```
data/
  HAM10000_metadata.csv
  HAM10000_images_part_1/
  HAM10000_images_part_2/
  HAM10000_segmentations_lesion_tschandl/   # {image_id}_segmentation.png each
```

Reported metrics: test-split lesion-grouped classification (balanced accuracy, macro F1, OVR ROC-AUC, PR-AUC, ECE), deletion/insertion AUC with Grad-CAM saliency, IoU and pointing-game hit rate against Tschandl masks at the model input resolution.

## Reference run (RTX 4060, 2026-10-22)

One full 1002-image lesion-grouped split with local code, mask pairing and Grad-CAM at 224x224:

| Metric | 1002-image split | 200-image subset |
|---|---:|---:|
| Balanced accuracy | 0.949 | 0.960 |
| Macro F1 | 0.928 | 0.944 |
| ECE | 0.017 | 0.031 |
| ROC-AUC (OVR macro) | 0.996 | 0.996 |
| PR-AUC (macro) | 0.984 | 0.982 |
| IoU (15% positive fraction) | 0.467 +- 0.179 | 0.490 +- 0.181 |
| Pointing game hit rate | 0.971 | 0.980 |
| Deletion AUC (zeros baseline, lower=better) | 0.691 | 0.690 |
| Insertion AUC (mean baseline, higher=better) | 0.632 | 0.629 |

Note: the split here is a StratifiedGroupKFold(10, seed 0) fold, not the authors' holdout, so classification numbers may include images the authors used for training. Faithfulness numbers differ from the authors' report (macro deletion 0.174 / insertion 0.203) because that protocol uses a blur baseline and per-ground-truth-class curves, while ours uses an argmax target with a zeros baseline.

## Licences and data

- Code in this repository: MIT.
- Model weights: Apache-2.0 (`sgonzalez2000/dermai-efficientnet-b0`).
- HAM10000 images and masks: CC BY-NC-SA 4.0, non-clinical research use only.
