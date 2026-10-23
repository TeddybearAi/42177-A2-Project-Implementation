# Imports the libraries required for dataset loading,
# image preprocessing, data augmentation, and batch creation.
from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    Compose, Resize, RandomCrop, ToTensor, Normalize,
    RandomHorizontalFlip, RandomVerticalFlip,
    RandomRotation, ColorJitter
)
from torch.utils.data import DataLoader
import torch


# Define the image dimensions expected by the VGG16 model
# and the number of images processed in each batch.
IMG_SIZE = 224
BATCH_SIZE = 32


# Define the root directories for the PlantVillage dataset
# used for training/testing and the PlantDoc dataset used
# for external verification.
PV_ROOT = Path(r"C:\VGG 16\PlantVillage")
MV_ROOT = Path(r"C:\VGG 16\Matlab One")


# Training transformations perform image augmentation
# to improve model robustness and reduce overfitting.
# Images are resized, randomly cropped, flipped,
# rotated and colour-adjusted before normalisation.
train_transform = Compose([
    Resize((256, 256)),              # Faster than RandomResizedCrop
    RandomCrop(IMG_SIZE),            # Provides augmentation
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    RandomRotation(20),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])
])


# Testing and verification datasets are only resized
# and normalised. No augmentation is applied to ensure
# evaluation is performed consistently.
test_transform = Compose([
    Resize((IMG_SIZE, IMG_SIZE)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])
])

verify_transform = test_transform


# Creates a DataLoader for the PlantVillage training set.
# Images are shuffled each epoch to improve learning,
# while worker processes and memory pinning are used
# to accelerate data loading.
def training_data_loader():
    train_dataset = ImageFolder(PV_ROOT / "train", transform=train_transform)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,          # Parallel loading
        pin_memory=True,        # Faster GPU transfer
        persistent_workers=True # Stable workers
    )
    return train_loader, train_dataset.classes


# Creates a DataLoader for the PlantVillage testing set.
# Shuffling is disabled to ensure reproducible evaluation.
def testing_data_loader():
    test_dataset = ImageFolder(PV_ROOT / "test", transform=test_transform)
    test_loader  = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    return test_loader, test_dataset.classes


# Creates a DataLoader for the PlantDoc verification set.
# This dataset is used to evaluate model performance on
# images collected under different conditions from training.
def verification_data_loader():
    verify_dataset = ImageFolder(MV_ROOT, transform=verify_transform)
    verify_loader  = DataLoader(
        verify_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    return verify_loader, verify_dataset.classes