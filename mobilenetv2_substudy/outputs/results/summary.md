# MobileNetV2 results

|   Seeds | Val macro-F1   | E1 accuracy   | E1 macro-F1   | E2 accuracy   | E2 macro-F1   | E2 95% CI (mean)   | ΔF1 (E1−E2)   | Retention     |
|--------:|:---------------|:--------------|:--------------|:--------------|:--------------|:-------------------|:--------------|:--------------|
|       3 | 0.994 ± 0.001  | 0.991 ± 0.000 | 0.992 ± 0.000 | 0.250 ± 0.010 | 0.197 ± 0.009 | [0.174, 0.220]     | 0.795 ± 0.010 | 0.199 ± 0.009 |

## Cost

| model       |   params_M |   GMACs |   GFLOPs |   gpu_latency_ms |
|:------------|-----------:|--------:|---------:|-----------------:|
| MobileNetV2 |       2.24 |    0.33 |     0.65 |             8.53 |

## Per seed

|   seed | split    |    n |   accuracy |   macro_precision |   macro_recall |   macro_f1 |   f1_ci_low |   f1_ci_high |
|-------:|:---------|-----:|-----------:|------------------:|---------------:|-----------:|------------:|-------------:|
|      0 | val      | 2609 |      0.994 |             0.994 |          0.993 |      0.994 |     nan     |      nan     |
|      0 | test     | 2609 |      0.991 |             0.991 |          0.992 |      0.991 |     nan     |      nan     |
|      0 | external | 1049 |      0.262 |             0.337 |          0.253 |      0.208 |       0.186 |        0.231 |
|      1 | val      | 2609 |      0.995 |             0.995 |          0.995 |      0.995 |     nan     |      nan     |
|      1 | test     | 2609 |      0.991 |             0.990 |          0.992 |      0.991 |     nan     |      nan     |
|      1 | external | 1049 |      0.246 |             0.317 |          0.225 |      0.192 |       0.169 |        0.215 |
|      2 | val      | 2609 |      0.994 |             0.994 |          0.994 |      0.994 |     nan     |      nan     |
|      2 | test     | 2609 |      0.991 |             0.991 |          0.994 |      0.992 |     nan     |      nan     |
|      2 | external | 1049 |      0.243 |             0.318 |          0.235 |      0.191 |       0.169 |        0.215 |

## Per class on Cropped PlantDoc (mean over seeds)

| class                 |   precision |   recall |   f1 |
|:----------------------|------------:|---------:|-----:|
| Pepper_Bacterial_spot |        0.28 |     0.24 | 0.25 |
| Pepper_healthy        |        0.32 |     0.55 | 0.39 |
| Potato_Early_blight   |        0.50 |     0.10 | 0.16 |
| Potato_Late_blight    |        0.30 |     0.09 | 0.14 |
| Tomato_Bacterial_spot |        0.16 |     0.03 | 0.05 |
| Tomato_Early_blight   |        0.16 |     0.52 | 0.25 |
| Tomato_healthy        |        0.03 |     0.01 | 0.01 |
| Tomato_Late_blight    |        0.27 |     0.48 | 0.34 |
| Tomato_Leaf_Mold      |        0.23 |     0.03 | 0.06 |
| Tomato_Septoria       |        0.27 |     0.48 | 0.34 |
| Tomato_Mosaic_virus   |        1.00 |     0.02 | 0.04 |
| Tomato_YLCV           |        0.38 |     0.31 | 0.34 |