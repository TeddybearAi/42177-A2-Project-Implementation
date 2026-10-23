"""
Configuration for the baseline CNN, "new resize style" (the group's shared protocol).

Differences from the first CNN (WorkPlace/CNN):
  * every image is resized to 256 on the shorter side and centre-cropped to 224
    (training images: resize 256, then random resized crop 60-100% of the area)
  * training seeds 42, 43, 44
  * external set: the complete Cropped PlantDoc extraction, 1,097 images in the twelve
    shared classes, nothing removed (as in the shared manifest)
The PlantVillage split (seed 42, 14 duplicates removed) and the CNN itself are unchanged.

This folder is self-contained. Change PV_DIR / PD_DIR below if the datasets live somewhere else.
"""
import os
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent            # .../WorkPlace/CNN new resize style
PV_DIR = Path(r"D:\IPPR\PlantVillage")            # PlantVillage class folders
PD_DIR = Path(r"D:\IPPR\PlantDoc-Dataset-full")   # complete Cropped PlantDoc extraction (train/ and test/)
# Two different zip files differ only in upper/lower case; Windows keeps one, so the shared
# manifest has 1,097 images. Our extraction kept both (one renamed with _1): drop the renamed copy.
PD_EXCLUDE = {"potato-blight-phytophora-infestans-close-up-of-infected-leaf-showing-a60hxn_1.jpg"}

OUT_DIR = Path(os.environ.get("CNN_OUT_DIR", HERE / "outputs"))   # env var lets a test run write elsewhere
DATA_DIR = OUT_DIR / "data"                       # split manifests (CSV)
RUNS_DIR = OUT_DIR / "runs"                       # one folder per seed
RESULTS_DIR = OUT_DIR / "results"                 # metrics, tables, figures

PV_MANIFEST = DATA_DIR / "plantvillage_split.csv"
PD_MANIFEST = DATA_DIR / "plantdoc_external.csv"

# ----------------------------------------------------------------------------
# Twelve shared classes: (short name, PlantVillage folder, PlantDoc folder)
# ----------------------------------------------------------------------------
CLASS_MAP = [
    ("Pepper_Bacterial_spot", "Pepper__bell___Bacterial_spot",         "Bell_pepper leaf spot"),
    ("Pepper_healthy",        "Pepper__bell___healthy",                "Bell_pepper leaf"),
    ("Potato_Early_blight",   "Potato___Early_blight",                 "Potato leaf early blight"),
    ("Potato_Late_blight",    "Potato___Late_blight",                  "Potato leaf late blight"),
    ("Tomato_Bacterial_spot", "Tomato_Bacterial_spot",                 "Tomato leaf bacterial spot"),
    ("Tomato_Early_blight",   "Tomato_Early_blight",                   "Tomato Early blight leaf"),
    ("Tomato_healthy",        "Tomato_healthy",                        "Tomato leaf"),
    ("Tomato_Late_blight",    "Tomato_Late_blight",                    "Tomato leaf late blight"),
    ("Tomato_Leaf_Mold",      "Tomato_Leaf_Mold",                      "Tomato mold leaf"),
    ("Tomato_Septoria",       "Tomato_Septoria_leaf_spot",             "Tomato Septoria leaf spot"),
    ("Tomato_Mosaic_virus",   "Tomato__Tomato_mosaic_virus",           "Tomato leaf mosaic virus"),
    ("Tomato_YLCV",           "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato leaf yellow virus"),
]
CLASS_NAMES = [c[0] for c in CLASS_MAP]
NUM_CLASSES = len(CLASS_NAMES)

# ----------------------------------------------------------------------------
# Protocol and hyper-parameters
# ----------------------------------------------------------------------------
SPLIT_SEED = 42                 # fixed 70/15/15 PlantVillage split
SEEDS = [42, 43, 44]            # training seeds (as in the shared protocol)
RESIZE = 256                    # shorter side before cropping
IMG_SIZE = 224                  # centre crop (evaluation) / random resized crop (training)
CROP_SCALE = (0.6, 1.0)         # area range of the training random resized crop
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BATCH_SIZE = 64
NUM_WORKERS = 2                 # Windows: each worker reserves ~2 GB of virtual memory

EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 6                    # early stopping on validation macro-F1
LABEL_SMOOTHING = 0.1
DROPOUT = 0.3
N_BOOTSTRAP = 1000

for d in (DATA_DIR, RUNS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
