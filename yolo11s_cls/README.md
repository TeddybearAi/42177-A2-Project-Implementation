# YOLO11s-cls

**Author:** Wei Cui

This folder trains Ultralytics YOLO11s-cls on the group's shared manifest, using seeds 42, 43 and 44, and evaluates it on:

- the PlantVillage internal test set (2,609 images);
- the Cropped PlantDoc external set (1,097 images).

It was run in Google Colab on a Tesla T4. In the report, this work appears in:

- Models › YOLO11s-cls;
- Results › YOLO11s-cls and Table 7;
- Table A3;
- Appendix B, Table B2 and Listings B9–B11.

## Files

| File | What it does |
|---|---|
| `YOLO11s.ipynb` | The Colab notebook that ran everything: environment, data preparation, training, freezing, evaluation and analysis. |
| `yolo_pipeline.py` | Loads and checks the manifest, builds the model and transforms, and trains and evaluates (`smoke`, `train` and `evaluate` modes). |
| `prepare_pv.py` | Extracts only the manifest's PlantVillage images from the Kaggle archive and checks each one by MD5. |
| `submission_analysis.py` | Audits the external set, then produces the bootstrap intervals, per-class tables and figures from the saved predictions. |
| `class_mapping.json` | The group's twelve-class order. |
| `notebook_environment.json` | GPU and package versions recorded by the notebook. |
| `audits/` | Checks run before and after evaluation. The model hashes were frozen before PlantDoc was opened (`frozen_before_external.json`). |
| `logs/` | Console logs of data preparation, the three training runs and the three evaluations. |
| `results/` | Results per seed, mean ± SD, domain gap, per-class tables, confusion matrices and training curves. |

## Usage

The shared manifest is `../resnet50_a2/outputs/manifest.csv`. Example commands for seed 42:

```
pip install ultralytics==8.4.53 pandas scikit-learn matplotlib
python prepare_pv.py --archive <PlantVillage zip> --manifest ../resnet50_a2/outputs/manifest.csv --out data/pv --full
python yolo_pipeline.py train    --manifest ../resnet50_a2/outputs/manifest.csv --mapping class_mapping.json --weights yolo11s-cls.pt --pv-root data/pv --out runs/train_s42 --seed 42 --batch 64 --warmup 2 --epochs 20
python yolo_pipeline.py evaluate --manifest ../resnet50_a2/outputs/manifest.csv --mapping class_mapping.json --weights yolo11s-cls.pt --pv-root data/pv --pd-root <Cropped PlantDoc> --checkpoint runs/train_s42/best_model.pt --out runs/eval_s42 --seed 42 --batch 64
```

These are the settings the notebook used: batch 64, 2 warm-up epochs and 20 fine-tuning epochs.

The notebook shows the exact sequence that was run, including the `smoke` check and the freezing step.

## What is not included

- The trained checkpoints (`best_model.pt`).
- The full internal and validation prediction files.
- The datasets.

The external predictions used in the cross-model comparison are in `../cross_model_comparison/predictions/yolo_s42`–`yolo_s44`.

The author's student ID has been removed from the notebook's title cell. Nothing else in the notebook was changed.
