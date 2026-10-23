# VGG16 (separate experiment)

**Author:** Mingzhang (Harry) Han

This folder holds the transfer-learning experiment with an ImageNet-pretrained VGG16 (Simonyan & Zisserman, 2015).
It ran in its own pipeline, outside the group's shared protocol:

- it used its own random 80/20 PlantVillage split;
- it used its own PlantDoc folder;
- it reports accuracy and loss rather than macro-F1.

For these reasons the report describes it as a separate experiment (Models › VGG16, Results › VGG16, Table 10, Table A7) and leaves it out of the cross-model comparisons.

The code is unchanged from the version uploaded to the group folder, which is the same code as in Appendix B, Listings B37–B43. On Drive the split script is named `VGG16_split_plantvillage.py.py`; the doubled extension is dropped here.

## Files

| File | What it does |
|---|---|
| `VGG16_split_plantvillage.py` | Copies each PlantVillage class folder into `train/` and `test/` with a random 80/20 split (no fixed seed) |
| `VGG16_data.py` | Dataset paths, transforms and data loaders for training, testing and PlantDoc verification |
| `VGG16_model.py` | `VGG16Model`: torchvision VGG16 (IMAGENET1K_V1) with the last classifier layer replaced |
| `VGG16_train.py` | Training loop: SGD (lr 0.001, momentum 0.9, weight decay 5e-4), cosine annealing, test evaluation each epoch, checkpoint at best test accuracy |
| `VGG16_test.py` | Accuracy and loss of `best_model.pt` on the PlantVillage test folder |
| `VGG16_verify.py` | Accuracy and loss of `best_model.pt` on the PlantDoc folder |
| `VGG16_main.py` | Trains (20 epochs as uploaded), then tests and verifies |
| `VGG16_Results.xlsx` | Recorded results of the two runs |
| `VGG16_terminal_logs.txt` | Console output of the runs. These logs come from an earlier layout of the same files, without the `VGG16_` prefix and with progress bars. |

## Usage

1. Edit the dataset paths, which are local Windows paths:
   - `PV_ROOT` and `MV_ROOT` in `VGG16_data.py`;
   - `SOURCE` in `VGG16_split_plantvillage.py`.
2. Run:

```
python VGG16_split_plantvillage.py   # once: creates PlantVillage/train and PlantVillage/test
python VGG16_main.py                 # train, test, verify
```

## Recorded results

| Run | Epochs (saved) | PlantVillage test accuracy | PlantDoc accuracy |
|---|---|---|---|
| 1 | 30 (26) | 0.9881 | 0.1284 |
| 2 | 20 (16) | 0.9886 | 0.1275 |

These numbers need to be read with some care:

- **Checkpoint selection.** The checkpoint is the epoch with the highest test accuracy, so the PlantVillage figure is slightly optimistic.
- **PlantDoc labels.** `ImageFolder` assigns class indices from the alphabetical order of the folder names. The PlantDoc accuracy is therefore valid only if the PlantDoc folder holds exactly the same class names as the training folder; this has not been confirmed.
