"""
MobileNetV2 with transfer learning (Sandler et al., 2018).

* torchvision ImageNet-pretrained weights (IMAGENET1K_V2).
* The original 1000-way classifier is replaced by dropout (0.3) + a 12-way linear layer.
* 19 inverted-residual blocks: 1x1 expansion -> 3x3 depthwise conv -> 1x1 linear
  projection, with a shortcut between the narrow ends. About 2.2 M parameters
  with the new head and ~0.3 GMACs per 224 x 224 image.
"""
import torch.nn as nn
from torchvision import models

import mnv2_config as C


def build_mobilenetv2(pretrained=True):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    m = models.mobilenet_v2(weights=weights)
    m.classifier = nn.Sequential(nn.Dropout(C.DROPOUT), nn.Linear(m.last_channel, C.NUM_CLASSES))
    return m


def set_trainable(model, head_only):
    """Head-only warm-up: feature extractor frozen and in eval mode, so its batch-norm
    layers keep their ImageNet statistics. Afterwards everything is fine-tuned."""
    model.train()
    for p in model.features.parameters():
        p.requires_grad = not head_only
    if head_only:
        model.features.eval()


if __name__ == "__main__":
    import torch
    m = build_mobilenetv2(pretrained=False)
    print(f"parameters: {sum(p.numel() for p in m.parameters()) / 1e6:.2f} M")
    print("output shape:", tuple(m(torch.randn(2, 3, C.IMG_SIZE, C.IMG_SIZE)).shape))
