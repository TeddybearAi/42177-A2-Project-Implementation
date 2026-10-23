# 42177 Image Processing and Pattern Recognition: A2 Project Implementation (Group 2)

Cross-dataset plant leaf disease classification. The models are trained on **PlantVillage**, which contains laboratory images. They are then tested, without retraining, on **Cropped PlantDoc**, which contains field images. Both datasets share the same 12 classes.

## Contents

Each folder holds one person's work for one model, so the authorship stays clear.

| Folder | Model | Author | Status in the report |
|---|---|---|---|
| [`resnet50_a2/`](resnet50_a2/) | ResNet50 (frozen, layer4 and full fine-tuning) and EfficientNet-B0 | Teddy Ryu | Main model family. It also holds the shared manifest, the class mapping and the statistics scripts. |
| [`yolo11s_cls/`](yolo11s_cls/) | YOLO11s-cls (Ultralytics), Colab | Wei Cui | Main model, shared protocol |
| [`baseline_cnn/`](baseline_cnn/) | Baseline CNN trained from scratch | Qinxin Ye | Main model. It uses shared pre-processing and seeds 42–44 on its own partition of the same images. |
| [`vit_s16_matlab/`](vit_s16_matlab/) | Vision Transformer ViT-S/16 in MATLAB | Yuhang (Louis) Lin | Main model, shared protocol |
| [`mobilenetv2_substudy/`](mobilenetv2_substudy/) | MobileNetV2 | Qinxin Ye | Sub-study in its own pipeline. It is compared only within that pipeline. |
| [`vgg16_separate/`](vgg16_separate/) | VGG16 | Mingzhang (Harry) Han | Separate experiment with its own split and PlantDoc folder. It is not compared with the other models. |
| [`cross_model_comparison/`](cross_model_comparison/) | Paired bootstrap and McNemar tests across the five main model families | Group | Report Table 4 |

Hangyu Hu wrote the Introduction and Datasets sections of the report. His part has no code, so it is not in this repository.

Every folder has its own README. Each README lists the files, the commands that reproduce the results and where the work appears in the report.

## The shared protocol

- **Image list and split.** `resnet50_a2/outputs/manifest.csv` lists all 18,489 images with their split, label and MD5:
  - PlantVillage train, validation and test: 12,174 / 2,609 / 2,609 images;
  - PlantDoc external set: 1,097 images.
- **Classes.** `resnet50_a2/class_mapping.json` fixes the class order.
- **Predictions.** Every main model exports `predictions_external.csv` in the format described in `resnet50_a2/PREDICTIONS_FORMAT.md`. All five model families can therefore be compared on exactly the same 1,097 PlantDoc images (see `cross_model_comparison/`).

## What is not in the repository

- **Datasets.** Download PlantVillage from Kaggle (kaggle.com/datasets/emmarex/plantdisease) and Cropped PlantDoc from GitHub (github.com/pratikkayal/PlantDoc-Dataset). Each folder's README says where to point the scripts.
- **Trained model files** (`*.pt`, `*.onnx`, `preds.npz`). These are too large for GitHub; the training scripts recreate them. Each folder keeps the summary tables, logs and figures used in the report.
- **Large intermediate files.** These include full per-image audit tables, the internal and validation prediction files, and the Grad-CAM image grids.

## Changes made when the code was collected (23 October 2026)

The code was collected from the group's shared Drive folders. It is otherwise unchanged, apart from these additions and removals:

- the README files for the YOLO, ViT, VGG16 and cross-model folders;
- a "Notes for this repository" section at the end of the baseline CNN and MobileNetV2 READMEs;
- the MATLAB licence number, removed from `vit_s16_matlab/outputs/env/matlab_env.txt`;
- a student ID, removed from the YOLO notebook's title cell;
- the doubled `.py.py` extension, dropped from one VGG16 file name.
