# ResNet50 + EfficientNet-B0 — A2 implementation (42177, Group 2)

Cross-dataset plant leaf disease classification: train on **PlantVillage**, test on the
PlantVillage internal test set (**E1**) and, with no retraining, on **PlantDoc** (**E2**).
This package implements the ResNet50 protocol promised in A1 Sections 3.2 and 5.

| File | Purpose |
|---|---|
| `class_mapping.json` | the 12 shared classes (PlantVillage folder ↔ PlantDoc folder) + exclusions |
| `prepare_data.py` | audit, clean (unreadable + duplicate files), 70/15/15 split (seed 42), portable manifest, fixed Grad-CAM sample |
| `common.py` | transforms, dataset, model, metrics, bootstrap CI, McNemar, plotting |
| `train.py` | one run: `--arch resnet50 | efficientnet_b0`, `--regime frozen | layer4 | full`, `--seed`, `--device` |
| `run_all.py` | runs all regimes × seeds, spread across your GPUs, resumable |
| `aggregate.py` | mean ± SD tables, McNemar tests, report figures, teammate comparison |
| `gradcam.py` | Grad-CAM heat maps on the fixed sample |
| `paired_bootstrap.py` | paired bootstrap CIs for macro-F1 differences between configurations |
| `export_onnx.py`, `matlab/evaluate_in_matlab.m` | export trained models to ONNX and reproduce the evaluation in MATLAB |
| `PREDICTIONS_FORMAT.md` | the CSV format teammates export so models can be compared |

## 1. Environment (Windows, VS Code terminal)

```powershell
python -m venv .venv
.venv\Scripts\activate
# RTX 5090 (Blackwell) needs a CUDA 12.8+ build - pick the newest CUDA option on pytorch.org/get-started
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
```
Note which index is the 5090 and which is the 4080 (CUDA numbering can differ from Task Manager).

## 2. Data

* PlantVillage: Kaggle copy (`archive.zip`). `--pv-root` = the folder that **directly** contains
  `Pepper__bell___Bacterial_spot`, `Tomato_healthy`, ... (the Kaggle zip nests a `PlantVillage` folder;
  point at the innermost one).
* PlantDoc: `--pd-root` = the folder containing `train/` and `test/`.

```powershell
python prepare_data.py --pv-root "D:/data/PlantVillage" --pd-root "D:/data/PlantDoc_cropped" --out outputs
```
If a class folder name doesn't match, the error lists the available folders - fix the entry in
`class_mapping.json`. Check `outputs/split_counts.csv` and `outputs/data_checks.json`.

**Share `outputs/manifest.csv` and `outputs/gradcam_sample.csv` with the group.** Paths inside are
relative, so everyone trains and tests on exactly the same images. (If the group decides PlantDoc
`train` only, re-run with `--pd-splits train` *before* sharing.)

## 3. Train everything (both GPUs)

```powershell
python run_all.py --pv-root "D:/data/PlantVillage" --pd-root "D:/data/PlantDoc_cropped" --devices cuda:0,cuda:1
```
3 regimes × 3 seeds (42, 43, 44) = 9 runs. Each GPU takes the next job from the queue (full
fine-tunes first). Progress per run is in `runs/logs/<regime>_s<seed>.log`
(in a second terminal: `Get-Content runs/logs/full_s42.log -Wait`). Re-running skips finished runs.

Single run, e.g. to test the setup first:
```powershell
python train.py --pv-root "D:/data/PlantVillage" --pd-root "D:/data/PlantDoc_cropped" --regime layer4 --seed 42 --device cuda:0 --epochs 2
```
Defaults: batch 128, bf16 mixed precision, 8 data workers, 2 warm-up + up to 20 fine-tune epochs,
patience 5, AdamW (head 1e-3, backbone 1e-4), cosine decay, label smoothing 0.1, dropout 0.3.
If data loading hangs on Windows, add `--workers 0`; if GPU memory runs out, `--batch-size 64`.

## 3b. EfficientNet-B0 alternative (6 runs)

```powershell
python run_all.py --pv-root "<PV path>" --pd-root "<PlantDoc path>" --devices cuda:0,cuda:1 --arch efficientnet_b0 --regimes full,frozen
```
Same data, splits, augmentation, optimiser and selection rule as ResNet50. Runs go to `runs/effb0_<regime>_s<seed>/`.
(`--regimes layer4` also works for B0 and fine-tunes its last two MBConv stages + final conv.)

## 4. Results, statistics, figures

```powershell
python aggregate.py
python gradcam.py --run runs/full_s42 --pv-root "D:/data/PlantVillage" --pd-root "D:/data/PlantDoc_cropped"
python gradcam.py --run runs/full_s42 --pv-root "D:/data/PlantVillage" --pd-root "D:/data/PlantDoc_cropped" --set internal
```
`runs/summary/` then contains `model_summary.md` (paste-ready table), `mcnemar.csv`,
`fig_internal_vs_external.png`, `fig_per_class_f1.png`, `fig_external_prediction_share.png` and
confusion matrices. Use the representative run named in `model_summary.csv` for Grad-CAM.

Comparing with the baseline CNN and YOLO11s-cls (once teammates export predictions):
```powershell
python aggregate.py --extra "Baseline CNN=path/to/cnn_predictions_external.csv" --extra "YOLO11s-cls=path/to/yolo_predictions_external.csv"
```

## Per-run outputs (`runs/<regime>_s<seed>/`)
`results.json` (all metrics, 95% bootstrap CI, gap, retention, params, latency), `history.csv`,
`training_curves.png`, `predictions_{internal,external}.csv`, `per_class_*.csv`,
`confusion_*.csv/.png`, `report_*.txt`, `config.json`, `best_model.pt`.

## Protocol guarantees
* Model selection uses PlantVillage **validation** macro-F1 only; PlantDoc is evaluated once, after the
  best checkpoint is fixed.
* No augmentation on validation / test / external data.
* Frozen layers keep their ImageNet BatchNorm statistics (true linear probe).
* Data split seed fixed (42); training seeds vary (42, 43, 44).

## 5. Reproducing the evaluation in MATLAB (optional)

The models were trained in Python/PyTorch. For markers who prefer MATLAB, the trained
models can be exported to ONNX and evaluated in MATLAB with the same images and metrics.

1. In Python (once):
   ```powershell
   pip install onnx onnxruntime
   python export_onnx.py          # writes matlab\full_s44.onnx and matlab\effb0_full_s44.onnx
   ```
   Each export is checked against PyTorch (same logits on the same input).
2. In MATLAB (R2023b or later recommended; Deep Learning Toolbox, Image Processing Toolbox and the free
   add-on "Deep Learning Toolbox Converter for ONNX Model Format"): open `matlab\evaluate_in_matlab.m`
   and press Run. The dataset folders default to `PlantDoc-Dataset-master` and `PlantVillage` next to
   the project folder; if PlantDoc is not there, the script downloads it from GitHub (about 1 GB, once).
   To use copies elsewhere, edit `pdRoot` / `pvRoot` at the top of the script. Files whose names Windows
   changed during extraction (`?` → `_`) are matched automatically. Tested in MATLAB Online.

Tested result (MATLAB Online, PlantDoc external test, 1,097 images):

| Run | macro-F1 MATLAB | macro-F1 Python | Same prediction |
|---|---|---|---|
| `full_s44` (ResNet50) | 0.278 | 0.279 | 94.1% |
| `effb0_full_s44` (EfficientNet-B0) | 0.241 | 0.239 | 96.1% |

The script prints accuracy, macro-F1 and per-class F1, saves a row-normalised confusion matrix, and
reports how often its predictions agree with the Python predictions for the same images
(`matlab\matlab_summary_external.csv`). Small differences come from MATLAB's and Python's image
resizing, so agreement is expected to be very high but not always exactly 100%.
