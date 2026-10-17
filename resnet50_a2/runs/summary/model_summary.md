| Model | Seeds | E1 accuracy | E1 macro-F1 | E2 accuracy | E2 macro-F1 | Gap (E1−E2) | Retention | Params (M) | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|
| ResNet50 frozen backbone | 3 | 0.929 ± 0.002 | 0.915 ± 0.003 | 0.264 ± 0.002 | 0.223 ± 0.003 | 0.692 ± 0.004 | 0.244 ± 0.003 | 23.5 | 5.9 |
| ResNet50 layer4 fine-tune | 3 | 0.993 ± 0.002 | 0.992 ± 0.002 | 0.305 ± 0.001 | 0.268 ± 0.004 | 0.724 ± 0.006 | 0.270 ± 0.004 | 23.5 | 8.0 |
| ResNet50 full fine-tune | 3 | 0.996 ± 0.000 | 0.995 ± 0.001 | 0.328 ± 0.007 | 0.279 ± 0.004 | 0.716 ± 0.004 | 0.281 ± 0.004 | 23.5 | 5.5 |
| EffNet-B0 frozen backbone | 3 | 0.941 ± 0.002 | 0.931 ± 0.003 | 0.283 ± 0.014 | 0.244 ± 0.013 | 0.687 ± 0.014 | 0.262 ± 0.014 | 4.0 | 6.7 |
| EffNet-B0 full fine-tune | 3 | 0.993 ± 0.002 | 0.993 ± 0.002 | 0.284 ± 0.014 | 0.238 ± 0.015 | 0.755 ± 0.016 | 0.240 ± 0.015 | 4.0 | 5.5 |

Values are mean ± SD over training seeds. E1 = PlantVillage internal test; E2 = PlantDoc external test (never used for training or selection). Latency = single-image GPU inference time.