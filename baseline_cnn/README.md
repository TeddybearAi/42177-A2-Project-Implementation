# Baseline CNN — new resize style (group's shared protocol)

The baseline CNN of Assessment 1, Section 3.1, retrained to follow the protocol of the shared manifest
(Teddy's ResNet50 / EfficientNet-B0 runs). Self-contained: nothing here imports the rest of `WorkPlace/`.
The first CNN (`WorkPlace/CNN`) is kept unchanged for comparison.

## What changed from the first CNN

| | First CNN (`WorkPlace/CNN`) | New resize style (this folder) |
|---|---|---|
| Evaluation pre-processing | Resize to 224 × 224 | Resize 256 (shorter side), centre-crop 224 |
| Training crop | Random resized crop 70-100% of the whole image | Resize 256, then random resized crop 60-100% |
| Training seeds | 0, 1, 2 | 42, 43, 44 |
| External set | 1,049 images (incomplete extraction, 12 removed) | 1,097 images, complete extraction, nothing removed |
| PlantVillage split | seed 42, 14 duplicates removed | identical image list (checked) |

Unchanged: the network (1.18 M parameters), flips, rotation 20°, colour jitter, AdamW (lr 1e-3, weight decay
1e-4), cosine decay, label smoothing 0.1, batch 64, up to 30 epochs, early stopping after 6 epochs without a
better PlantVillage validation macro-F1. The training budget is the CNN's own: a network trained from scratch
needs more epochs than Teddy's fine-tuning schedule.

## Files

| File | What it does |
|---|---|
| `cnn_config.py` | Paths, class mapping, protocol and hyper-parameters |
| `data.py` | Audit, split, external set, transforms, loaders |
| `model.py` | The `BaselineCNN` network |
| `train.py` | Training loop with best-validation checkpoint, early stopping and resume |
| `evaluate.py` | E1 / E2 metrics, bootstrap CI, per-class table, figures |
| `run_cnn.py` | One command: prepare data → train → evaluate → summarise |
| `report_analysis.py` | Figures and numbers for the report section (per class, confusion, Grad-CAM, old vs new) |

## Usage

```
cd "D:\IPPR\WorkPlace\CNN new resize style"
..\.venv\Scripts\python.exe run_cnn.py            # seeds 42, 43, 44 (~20 min each on an RTX 5070 laptop GPU)
..\.venv\Scripts\python.exe report_analysis.py    # report figures and numbers
```

Finished seeds are skipped and an interrupted seed resumes from its last epoch.

## Outputs (`outputs/`)

- `data/` — `plantvillage_split.csv`, `plantdoc_external.csv` (1,097 images)
- `runs/cnn_s<seed>/` — `best.pt`, `history.csv`, `preds.npz`
- `results/` — `summary.md`, per-seed and per-class tables, confusion matrix, training curves
- `report/` — numbers and figures used in the report section
- `logs/train.log` — console output of the training run

## Notes for this repository (added by the group)

- **Status in the report.** This is the baseline CNN reported in the group report: Models › Baseline CNN, Table 2, Table A6 and Listings B30–B36.
- **What is included.** The code, `requirements.txt` and `outputs/data/` (its split files). Also `outputs/runs/cnn_s<seed>/` (`history.csv` and `done.json` only), `outputs/results/` (summary and per-class tables, `numbers.json`), `outputs/figures/` and `outputs/logs/train.log`.
- **What is not included.** The checkpoints (`best.pt`), `preds.npz` and the Grad-CAM image grids. They are too large for GitHub, so running `run_cnn.py` here retrains from scratch.
- **Where the comparison predictions are.** The group-format predictions on the 1,097 shared PlantDoc images, used in the cross-model comparison, are in `../cross_model_comparison/predictions/cnn_s42`–`cnn_s44`. Their export step is not part of this package.
- **Two extra dependencies.**
  - `report_analysis.py` also reads the earlier pilot CNN's outputs (`WorkPlace/CNN/outputs/results_1097`), which are not in this repository.
  - Its PlantVillage split uses the same 17,392 images as the shared manifest but a different partition.
