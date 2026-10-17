# 42177 Image Processing and Pattern Recognition: A2 Project Implementation (Group 2)

Cross-dataset plant leaf disease classification. Models are trained on **PlantVillage**
(laboratory images) and tested, without retraining, on **PlantDoc** (field images), using
12 classes shared by both datasets.

## Contents

| Folder | Model | Author |
|---|---|---|
| [`resnet50_a2/`](resnet50_a2/) | ResNet50 with transfer learning (frozen, layer4 and full fine-tuning) and EfficientNet-B0 as an alternative | Teddy Ryu |

Each folder has its own README with setup steps and the commands that reproduce its results.
All models use the same image list and data split (`resnet50_a2/outputs/manifest.csv`), so their
predictions on the 1,097 PlantDoc images can be compared directly.

## What is not in the repository

- **Datasets.** Download PlantVillage from Kaggle (kaggle.com/datasets/emmarex/plantdisease) and
  Cropped-PlantDoc from GitHub (github.com/pratikkayal/PlantDoc-Dataset); the README in each folder
  says where to point the scripts.
- **Trained model files** (`best_model.pt`, `.onnx`), which are too large for GitHub. They are
  recreated by the training and export scripts. The summary tables and figures used in the report
  are included in `resnet50_a2/runs/summary/`.
