# ViT-S/16 (MATLAB)

**Author:** Yuhang (Louis) Lin

Vision Transformer `small-16-imagenet-384`, fine-tuned in MATLAB R2026b (Deep Learning Toolbox and Computer Vision Toolbox) on the group's shared manifest.

Five runs were specified in advance; two more were exploratory.

| Run | Regime | Seed |
|---|---|---|
| R1 | attention fine-tuning | 42 |
| R2 | frozen backbone | 42 |
| R3 | attention fine-tuning with grey-world white balance | 42 |
| R4 | attention fine-tuning | 43 |
| R5 | attention fine-tuning | 44 |
| R2x | frozen backbone (exploratory) | 42 |
| R2p | linear probe (exploratory, `linearProbe.m`) | 42 |

In the report, this work appears in:

- Models › ViT-S/16 and Results › ViT-S/16, with Table 8;
- Table A4;
- Appendix B, Table B3 and Listings B12–B23.

## Files

| File | What it does |
|---|---|
| `src/vitConfig.m` | Paths, preprocessing, augmentation, training recipe and run definitions |
| `src/loadManifest.m` | Reads the shared manifest and class mapping, keeping the explicit class order |
| `src/auditDataset.m` | Independent check of every manifest file: presence, MD5, decodability, counts and duplicates |
| `src/preprocessImage.m` | Decodes an image and applies the evaluation or training transform, with optional grey-world white balance |
| `src/trainRun.m` | Trains, selects and evaluates one run (R1–R5, R2x) |
| `src/linearProbe.m` | Exploratory run R2p: a linear layer on frozen ViT features |
| `src/predictScores.m` | Class probabilities for a list of images |
| `src/evaluateSplit.m` | E1 and E2 evaluation; writes predictions in the group's shared format |
| `src/classificationMetrics.m`, `src/bootstrapF1.m` | Metrics and the 95% bootstrap interval for macro-F1 |
| `src/summarizeRuns.m` | Collects finished runs into the summary tables and figures |
| `src/selfTest.m` | Self-check of the loader, preprocessing and metrics |
| `outputs/summary/` | `vit_runs.csv` (one row per run), the comparison table and two figures |
| `outputs/logs/` | Training logs for each run, plus a smoke-test log |
| `outputs/audit/` | Dataset audit summary, counts and log |
| `outputs/env/matlab_env.txt` | MATLAB and toolbox versions. The licence-number line has been removed. |

## Setup and usage

1. Place the datasets next to this folder, or edit `pvRoot` and `pdRoot` in `src/vitConfig.m`.
2. Create `shared/` inside this folder and copy in the group files from `../resnet50_a2/`:
   - `outputs/manifest.csv` → `shared/manifest.csv`;
   - `class_mapping.json` → `shared/class_mapping.json`;
   - `outputs/gradcam_sample.csv` → `shared/gradcam_sample.csv`;
   - `runs/summary/model_summary.csv` → `shared/results/model_summary.csv`;
   - `runs/summary/per_class_f1.csv` → `shared/results/per_class_f1.csv`.
3. From MATLAB, inside `src/`:

```
selfTest                       % checks
auditDataset(vitConfig());     % dataset audit
trainRun("R1");                % repeat for R2, R3, R4, R5 and R2x
linearProbe(vitConfig());      % exploratory R2p
summarizeRuns(vitConfig());    % summary tables and figures
```

The code comments refer to the author's protocol notes (`docs/adr/0002`). Those notes are not part of this repository; the settings they describe are in `src/vitConfig.m` and in report Table A4.

## What is not included

- Checkpoints.
- The per-run output folders (`outputs/runs/`).
- The full per-image audit table.
- The large example figure.

The external predictions of R1, R4 and R5, which the cross-model comparison uses as seeds 42, 43 and 44, are in `../cross_model_comparison/predictions/vit_s42`–`vit_s44`.
