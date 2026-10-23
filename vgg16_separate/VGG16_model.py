# This section imports the neural network components. 
# PyTorch's neural network module provides the layers used for classification
# torchvision.models provides pretrained computer vision architectures, including VGG16.
import torch.nn as nn
import torchvision.models as models

# Initializes the model and receives the total number of classes
class VGG16Model(nn.Module):
    def __init__(self, n):
        super().__init__()

        # Load pretrained VGG16 with previously learned features such as edges, textures, shapes, and patterns. Improves training efficiency.
        self.backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # Replace final classifier layer for the purpose of this project.
        self.backbone.classifier[6] = nn.Linear(4096, n)

# The forward function defines how images pass through the network. 
# Images are fed into the pretrained VGG16 backbone
# extracts features and produces a class prediction. 
    def forward(self, x):
        return self.backbone(x)
