#This section imports the libraries 
# os is used to access directories, 
# shutil is used to copy image files, 
# random is used to shuffle images before splitting, 
# Path provides an easier and more reliable way to work with file paths.

import os
import shutil
import random
from pathlib import Path

# defines the source dataset location and creates references for the training and testing folders. 
# The PlantVillage dataset originally contains only class folders, dedicated train and test directories must be generated.

SOURCE = Path(r"C:\VGG 16\PlantVillage")
TRAIN  = SOURCE / "train"
TEST   = SOURCE / "test"

# Creates the training and testing directories if they do not already exist.
TRAIN.mkdir(exist_ok=True)
TEST.mkdir(exist_ok=True)

split_ratio = 0.8  # 80% train, 20% test

IGNORE = {"train", "test"}  # ignore leftover folders

#This loop iterates through every disease class in the PlantVillage dataset. 
# Non-class folders and unrelated files are skipped.

for cls in os.listdir(SOURCE):
    if cls in IGNORE:
        continue


    src_cls = SOURCE / cls
    if not src_cls.is_dir():
        continue

# All images within a class are collected and randomly shuffled before splitting.
    imgs = [f for f in os.listdir(src_cls) if (src_cls / f).is_file()]
    random.shuffle(imgs)

# The dataset is shuffled and divided according to 80-20 split ratio.
    split = int(len(imgs) * split_ratio)
    train_imgs = imgs[:split]
    test_imgs  = imgs[split:]

# The train and test directories folders are created for each class.
    (TRAIN / cls).mkdir(exist_ok=True)
    (TEST / cls).mkdir(exist_ok=True)

#Training and testing images are copied into their respective directories while maintaining the same class structure.

    for img in train_imgs:
        shutil.copy(src_cls / img, TRAIN / cls / img)

    for img in test_imgs:
        shutil.copy(src_cls / img, TEST / cls / img)

print("PlantVillage split complete.")
