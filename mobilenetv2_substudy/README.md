# MobileNetV2 — PlantVillage → Cropped PlantDoc

Self-contained code for the MobileNetV2 alternative model (Assessment Task 1, Section 4.3).
Nothing in this folder depends on the rest of `WorkPlace/`. It uses the same 12 classes,
the same data split (seed 42), the same pre-processing and the same evaluation protocol as
the baseline CNN, ResNet50 and YOLO11s-cls, so its numbers are directly comparable.

**Keep `outputs/` next to the `.py` files** — the trained models are found there. If the
folder is moved without it, the code will retrain from scratch.

## Files

| File | What it does |
|---|---|
| `mnv2_config.py` | Dataset paths, 12-class mapping, split seed, hyper-parameters |
| `data.py` | Image audit (unreadable files, duplicates, conflicting labels), 70/15/15 split, transforms, data loaders |
| `model.py` | MobileNetV2 with a new 12-class head (run it directly to check the model) |
| `train.py` | Training: 2 head-only warm-up epochs, then full fine-tuning; best-validation checkpoint; early stopping; resume |
| `evaluate.py` | E1/E2 metrics, bootstrap CI, per-class table, cost, figures, and comparison with the other models |
| `run_mobilenetv2.py` | One command: prepare data → train → evaluate → compare |

## Requirements

Python 3.10+ with `torch`, `torchvision`, `scikit-learn`, `scipy`, `pandas`, `matplotlib`,
`pillow`, `tabulate`, `thop`. A CUDA GPU is strongly recommended. On this laptop the project
environment already has everything:

```
cd D:\IPPR\WorkPlace\Extra_Experiments\MobileNetV2
..\..\.venv\Scripts\python.exe run_mobilenetv2.py
```

If the datasets are elsewhere, edit `PV_DIR` and `PD_DIR` in `mnv2_config.py`.

## Usage

```
python run_mobilenetv2.py                                  # all three seeds (~8 min per seed on an RTX 5070 laptop GPU)
python run_mobilenetv2.py --seeds 0                        # one seed
python run_mobilenetv2.py --seeds 9 --epochs 1 --limit 20  # quick check that everything runs (then delete outputs/runs/mobilenetv2_s9)
python model.py                                            # check the model builds
```

Finished seeds are skipped; an interrupted seed resumes from its last saved epoch.
To retrain a seed from scratch, delete `outputs/runs/mobilenetv2_s<seed>/`.

## Training settings

| Setting | Value |
|---|---|
| Pretrained weights | torchvision ImageNet (IMAGENET1K_V2) |
| Head | Dropout 0.3 + linear layer, 12 classes |
| Input | 224 × 224, ImageNet normalisation |
| Augmentation (training only) | Random resized crop (70–100%), horizontal/vertical flips, rotation ≤ 20°, small colour jitter |
| Optimiser | AdamW, head 1e-3, backbone 1e-4, weight decay 1e-4, cosine decay |
| Schedule | Epochs 0–1 head only, then full fine-tuning; max 12 epochs; early stop after 4 epochs without improvement |
| Loss | Cross-entropy with label smoothing 0.1; mixed precision |
| Checkpoint selection | Best PlantVillage **validation** macro-F1 (PlantDoc never used for any choice) |

## Outputs (`outputs/`)

- `data/` — split manifests (`plantvillage_split.csv`, `plantdoc_external.csv`)
- `runs/mobilenetv2_s<seed>/` — `best.pt`, `history.csv`, `preds.npz` (12 class probabilities for every test image)
- `results/summary.md` — headline table, cost, per-seed metrics, per-class PlantDoc scores
- `results/comparison.md` — MobileNetV2 vs baseline CNN, ResNet50 and YOLO11s-cls, with McNemar tests
- `results/fig_comparison.png`, `fig_confusion.png`, `fig_training.png`
- `training_log.txt` — console output of the three training runs

The comparison needs the main pipeline's predictions (`WorkPlace/runs/`); without them it is skipped.

## Results (3 seeds, mean ± sd)

| Model | E1 macro-F1 (PlantVillage test) | E2 macro-F1 (Cropped PlantDoc) | Retention | Parameters | GMACs |
|---|---|---|---|---|---|
| Baseline CNN | 0.993 ± 0.002 | 0.146 ± 0.012 | 0.147 | 1.18 M | 2.61 |
| ResNet50 (layer4) | 0.994 ± 0.001 | 0.221 ± 0.003 | 0.222 | 23.5 M | 4.13 |
| YOLO11s-cls | 0.996 ± 0.001 | 0.187 ± 0.014 | 0.187 | 5.46 M | 0.74 |
| **MobileNetV2** | **0.992 ± 0.000** | **0.197 ± 0.009** | **0.199** | **2.24 M** | **0.33** |

GMACs = multiply-accumulates per 224 × 224 image (1 MAC = 2 FLOPs).

- MobileNetV2 beats the baseline CNN on PlantDoc in every seed (McNemar p ≤ 0.0001).
- It is not significantly different from YOLO11s-cls in 2 of 3 seeds, and not significantly
  different from ResNet50 in 2 of 3 seeds (ResNet50 better in seed 2, p = 0.015).
- It reaches ~90% of ResNet50's PlantDoc macro-F1 with ~10% of its parameters and ~8% of its computation.
- Task 1 asked whether a smaller model might fit less to PlantVillage backgrounds and generalise
  better: MobileNetV2 did **not** generalise better than ResNet50; it shows the same failure pattern
  (tomato healthy, mosaic virus and leaf mould recall ≤ 0.03 on PlantDoc).

## Using this package on another computer

1. Download the datasets once: PlantVillage (Kaggle release) and Cropped PlantDoc
   (https://github.com/pratikkayal/PlantDoc-Dataset).
2. Create an environment and install the packages (see `requirements.txt`; install PyTorch first,
   with the CUDA command if you have an NVIDIA GPU):
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
   pip install -r requirements.txt
   ```
3. Edit `PV_DIR` and `PD_DIR` at the top of the config file to point to your copies of the datasets.
4. Run the one command from inside this folder (`python run_mobilenetv2.py`). Because the trained models are
   included in `outputs/runs/`, training is skipped and the result tables and figures are rebuilt;
   the numbers should match `outputs/results/summary.md`.
   To retrain from scratch instead, delete `outputs/runs/` first (a different GPU gives very slightly
   different numbers; the 3-seed means should agree closely).

## Notes for this repository (added by the group)

- **How the report treats it.** This package rebuilt its own PlantVillage split and its own PlantDoc set of 1,049 images, rather than using the shared manifest's 1,097. The group report therefore treats MobileNetV2 as a sub-study and compares it only with ResNet50 (layer4) and YOLO11s-cls re-run in this same pipeline (report Table 9, Table A5, Listings B24–B29). Its numbers are not comparable with the main tables.
- **What is included.** The `.py` files above, `requirements.txt`, `outputs/data/` (split files), `outputs/results/` (summary tables and figures) and `outputs/training_log.txt`.
- **What is not included.** Trained checkpoints and `preds.npz` (`outputs/runs/`), and the re-runs of the other models (`outputs/other_models/`). They are too large for GitHub, so running `run_mobilenetv2.py` here retrains from scratch.

### Re-running from this archive

The tracked split CSVs contain the author's original absolute Windows dataset paths. Before running on another computer, set the dataset paths in the config file and set `MNV2_OUT_DIR` to a **new, empty output directory outside this repository**. The scripts will then build fresh manifests for those paths and train new checkpoints, while leaving the archived evidence unchanged.

The earlier usage section describes the author's original local package, which included checkpoints. This GitHub archive omits those checkpoints, so it trains again rather than rebuilding results from stored models. Its comparisons with other models also require the separate same-pipeline prediction files; they are not replaced by the main 1,097-image predictions.
