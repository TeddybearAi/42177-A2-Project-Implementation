# MobileNetV2 vs the group's other models (mean ± sd over seeds)

| model             |   seeds |   E1_mean |   E1_sd |   E2_mean |   E2_sd |   retention |
|:------------------|--------:|----------:|--------:|----------:|--------:|------------:|
| Baseline CNN      |       3 |     0.993 |   0.002 |     0.146 |   0.012 |       0.147 |
| ResNet50 (layer4) |       3 |     0.994 |   0.001 |     0.221 |   0.003 |       0.222 |
| YOLO11s-cls       |       3 |     0.996 |   0.001 |     0.187 |   0.014 |       0.187 |
| MobileNetV2       |       3 |     0.992 |   0.000 |     0.197 |   0.009 |       0.199 |

## McNemar tests on Cropped PlantDoc (MobileNetV2 vs each model, same seed)

|   seed | other             |   mnv2_right_other_wrong |   other_right_mnv2_wrong |   p_value |
|-------:|:------------------|-------------------------:|-------------------------:|----------:|
|      0 | Baseline CNN      |                      189 |                       81 |    0.0000 |
|      0 | ResNet50 (layer4) |                       94 |                       98 |    0.8287 |
|      0 | YOLO11s-cls       |                      142 |                      104 |    0.0181 |
|      1 | Baseline CNN      |                      169 |                      104 |    0.0001 |
|      1 | ResNet50 (layer4) |                       87 |                      103 |    0.2765 |
|      1 | YOLO11s-cls       |                      127 |                      121 |    0.7509 |
|      2 | Baseline CNN      |                      171 |                       94 |    0.0000 |
|      2 | ResNet50 (layer4) |                       70 |                      103 |    0.0147 |
|      2 | YOLO11s-cls       |                      106 |                      122 |    0.3205 |