| Model | Seeds | E1 accuracy | E1 macro-F1 | E2 accuracy | E2 macro-F1 | Gap (E1−E2) | Retention |
|---|---|---|---|---|---|---|---|
| ResNet50 frozen backbone | 3 | 0.929 ± 0.002 | 0.915 ± 0.003 | 0.264 ± 0.002 | 0.223 ± 0.003 | 0.692 ± 0.004 | 0.244 ± 0.003 |
| ResNet50 layer4 fine-tune | 3 | 0.993 ± 0.002 | 0.992 ± 0.002 | 0.305 ± 0.001 | 0.268 ± 0.004 | 0.724 ± 0.006 | 0.270 ± 0.004 |
| ResNet50 full fine-tune | 3 | 0.996 ± 0.000 | 0.995 ± 0.001 | 0.328 ± 0.007 | 0.279 ± 0.004 | 0.716 ± 0.004 | 0.281 ± 0.004 |
| EffNet-B0 frozen backbone | 3 | 0.941 ± 0.002 | 0.931 ± 0.003 | 0.283 ± 0.014 | 0.244 ± 0.013 | 0.687 ± 0.014 | 0.262 ± 0.014 |
| EffNet-B0 full fine-tune | 3 | 0.993 ± 0.002 | 0.993 ± 0.002 | 0.284 ± 0.014 | 0.238 ± 0.015 | 0.755 ± 0.016 | 0.240 ± 0.015 |
| ViT-S/16 attention fine-tune | 3 | 0.904 ± 0.034 | 0.888 ± 0.042 | 0.178 ± 0.014 | 0.135 ± 0.014 | 0.754 ± 0.043 | 0.152 ± 0.017 |
| ViT-S/16 frozen backbone | 1 | 0.341 | 0.296 | 0.205 | 0.160 | 0.136 | 0.542 |
| ViT-S/16 attention + gray-world (ablation) | 1 | 0.912 | 0.892 | 0.199 | 0.150 | 0.742 | 0.168 |
| ViT-S/16 frozen backbone, LR 1e-3 (exploratory) | 1 | 0.531 | 0.447 | 0.224 | 0.200 | 0.247 | 0.447 |

Mean ± SD over training seeds (single-seed rows: value only). CNN rows: shared/results/model_summary.csv (Teddy Ryu), unchanged.
