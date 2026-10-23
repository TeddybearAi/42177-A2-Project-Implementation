# Baseline CNN results

|   Seeds | Val macro-F1   | E1 accuracy   | E1 macro-F1   | E2 accuracy   | E2 macro-F1   | E2 95% CI (mean)   | ΔF1 (E1−E2)   | Retention     |
|--------:|:---------------|:--------------|:--------------|:--------------|:--------------|:-------------------|:--------------|:--------------|
|       3 | 0.989 ± 0.001  | 0.989 ± 0.001 | 0.990 ± 0.001 | 0.191 ± 0.005 | 0.171 ± 0.006 | [0.147, 0.192]     | 0.819 ± 0.007 | 0.172 ± 0.006 |

## Per seed

|   seed | split    |    n |   accuracy |   macro_precision |   macro_recall |   macro_f1 |   f1_ci_low |   f1_ci_high |
|-------:|:---------|-----:|-----------:|------------------:|---------------:|-----------:|------------:|-------------:|
|     42 | val      | 2609 |      0.990 |             0.991 |          0.988 |      0.989 |     nan     |      nan     |
|     42 | test     | 2609 |      0.990 |             0.991 |          0.989 |      0.990 |     nan     |      nan     |
|     42 | external | 1097 |      0.197 |             0.236 |          0.188 |      0.175 |       0.154 |        0.198 |
|     43 | val      | 2609 |      0.990 |             0.990 |          0.989 |      0.989 |     nan     |      nan     |
|     43 | test     | 2609 |      0.989 |             0.990 |          0.987 |      0.989 |     nan     |      nan     |
|     43 | external | 1097 |      0.190 |             0.267 |          0.188 |      0.172 |       0.148 |        0.193 |
|     44 | val      | 2609 |      0.991 |             0.990 |          0.990 |      0.990 |     nan     |      nan     |
|     44 | test     | 2609 |      0.990 |             0.992 |          0.991 |      0.991 |     nan     |      nan     |
|     44 | external | 1097 |      0.188 |             0.213 |          0.178 |      0.164 |       0.139 |        0.185 |

## Per class on Cropped PlantDoc (mean over seeds)

| class                 |   precision |   recall |   f1 |
|:----------------------|------------:|---------:|-----:|
| Pepper_Bacterial_spot |        0.15 |     0.33 | 0.21 |
| Pepper_healthy        |        0.17 |     0.21 | 0.19 |
| Potato_Early_blight   |        0.19 |     0.15 | 0.17 |
| Potato_Late_blight    |        0.19 |     0.03 | 0.06 |
| Tomato_Bacterial_spot |        0.13 |     0.20 | 0.16 |
| Tomato_Early_blight   |        0.14 |     0.30 | 0.19 |
| Tomato_healthy        |        0.32 |     0.03 | 0.06 |
| Tomato_Late_blight    |        0.19 |     0.38 | 0.26 |
| Tomato_Leaf_Mold      |        0.25 |     0.11 | 0.16 |
| Tomato_Septoria       |        0.38 |     0.21 | 0.26 |
| Tomato_Mosaic_virus   |        0.35 |     0.03 | 0.06 |
| Tomato_YLCV           |        0.40 |     0.24 | 0.29 |