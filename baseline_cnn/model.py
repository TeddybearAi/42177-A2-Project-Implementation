"""
Baseline CNN trained from scratch (Assessment Task 1, Section 3.1).

    input 3 x 224 x 224
    block 1: [conv3x3 -> BN -> ReLU] x 2, max-pool   ->  32 x 112 x 112
    block 2: same                                    ->  64 x  56 x  56
    block 3: same                                    -> 128 x  28 x  28
    block 4: same                                    -> 256 x  14 x  14
    global average pooling                           -> 256
    dropout (0.3) + linear                           ->  12 class scores

1.18 M parameters, all randomly initialised (no pretraining).
"""
import torch.nn as nn

import cnn_config as C


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=C.NUM_CLASSES, dropout=C.DROPOUT):
        super().__init__()

        def block(cin, cout):
            """Two 3x3 convolutions (each with batch norm and ReLU), then 2x2 max-pooling."""
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2))

        # filters double while the spatial size halves
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


if __name__ == "__main__":
    import torch
    m = BaselineCNN()
    print(m)
    print(f"parameters: {sum(p.numel() for p in m.parameters()) / 1e6:.2f} M")
    print("output shape:", tuple(m(torch.randn(2, 3, C.IMG_SIZE, C.IMG_SIZE)).shape))
