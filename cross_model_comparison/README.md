# Cross-model paired comparisons

This folder compares the five model families on the same 1,097 Cropped PlantDoc images. It produces the cross-model paragraphs in the report section "Paired statistical comparisons" and **Table 4**.

## Contents

| Path | What it is |
|---|---|
| `predictions/<run>/predictions_external.csv` | Group-format predictions for 15 runs: `image_id`, `y_true`, `y_pred`, `p_0`…`p_11` (see `../resnet50_a2/PREDICTIONS_FORMAT.md`). |
| `cross_model.py` | Runs the paired bootstrap and the McNemar tests. |
| `results/paired_bootstrap_external_cross_model.csv` | Macro-F1 differences per seed and averaged over seeds, with 95% intervals. |
| `results/mcnemar_cross_model.csv` | Accuracy per seed, discordant counts and exact two-sided p-values. |

### Runs in `predictions/`

| Folder | Model | Source |
|---|---|---|
| `full_s42`–`full_s44` | ResNet50, full fine-tuning | Teddy Ryu (`../resnet50_a2`) |
| `effb0_full_s42`–`effb0_full_s44` | EfficientNet-B0, full fine-tuning | Teddy Ryu |
| `yolo_s42`–`yolo_s44` | YOLO11s-cls | Wei Cui (`../yolo11s_cls`) |
| `vit_s42`–`vit_s44` | ViT-S/16, attention regime; runs R1, R4 and R5 | Yuhang (Louis) Lin (`../vit_s16_matlab`) |
| `cnn_s42`–`cnn_s44` | Baseline CNN | Qinxin Ye (`../baseline_cnn`) |

MobileNetV2 and VGG16 are not included. Both used their own splits and PlantDoc sets; see their folders.

## Method

- **Bootstrap.** `cross_model.py` imports `../resnet50_a2/paired_bootstrap.py` unchanged and replaces only its list of pairs.
  - 2,000 resamples, random seed 0, unstratified, percentile 95% interval.
  - The same resampled images are used for both models and all three seeds.
  - The headline figure is the difference averaged over seeds.
- **McNemar.** An exact McNemar test is run for each matched seed. A difference counts as consistent when p < 0.05 in at least 2 of 3 seeds.
- **Multiplicity.** Ten pairs are compared. Every seed that counts towards a consistent difference has p ≤ 0.0022, so the conclusions are unchanged at a Bonferroni threshold of 0.005.
- **Cross-check.** ResNet50 full against EfficientNet-B0 full reproduces the report's Table 3 exactly: +0.042 [+0.019, +0.064], McNemar 2 of 3 seeds.

## Checks on the inputs

- All 15 files contain the same 1,097 `image_id` values as the external rows of `../resnet50_a2/outputs/manifest.csv`.
- `y_true` is identical for every image in every file.
- In every row, `y_pred` equals the argmax of `p_0`…`p_11`.

## Reproduce

Run from this folder:

```
python cross_model.py ../resnet50_a2 predictions
```

Requirements: `numpy`, `pandas` and `scipy`. The script writes to `predictions/summary/`. Those files match `results/` exactly; this was checked on 23 October 2026.
