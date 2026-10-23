"""
Configuration for the stand-alone MobileNetV2 experiment (Task 1, Section 4.3).

This folder is self-contained: nothing here imports the group's main pipeline. The
class mapping, split seed, pre-processing and evaluation protocol are the same as for
the baseline CNN, ResNet50 and YOLO11s-cls, so results are directly comparable.

Change PV_DIR / PD_DIR below if the datasets live somewhere else.
Keep the outputs/ folder next to these .py files (trained models are found there).
"""
import os
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent            # this folder
PV_DIR = Path(r"D:\IPPR\PlantVillage")            # PlantVillage class folders
PD_DIR = Path(r"D:\IPPR\PlantDoc-Dataset")        # Cropped PlantDoc (train/ and test/)

OUT_DIR = Path(os.environ.get("MNV2_OUT_DIR", HERE / "outputs"))
DATA_DIR = OUT_DIR / "data"                       # split manifests (CSV)
RUNS_DIR = OUT_DIR / "runs"                       # one folder per seed
RESULTS_DIR = OUT_DIR / "results"                 # metrics, tables, figures
# pretrained weights: the project's cache on the original machine, otherwise a folder here
_project_cache = HERE.parent.parent / ".cache" / "torch"
TORCH_HOME = _project_cache if _project_cache.is_dir() else HERE / ".torch_cache"

PV_MANIFEST = DATA_DIR / "plantvillage_split.csv"
PD_MANIFEST = DATA_DIR / "plantdoc_external.csv"

# Predictions of the group's other models (CNN, ResNet50, YOLO11s-cls), used only for the
# comparison table. A copy ships in outputs/other_models/; the main pipeline's runs/ folder
# is used instead when present.
_main_runs = HERE.parent.parent / "runs"
MAIN_RUNS_DIR = _main_runs if _main_runs.is_dir() else OUT_DIR / "other_models"

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
# Protocol and hyper-parameters (same as ResNet50 in the main experiment)
# ----------------------------------------------------------------------------
SPLIT_SEED = 42
SEEDS = [0, 1, 2]
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BATCH_SIZE = 64
NUM_WORKERS = 2                 # Windows: each worker reserves ~2 GB of virtual memory

EPOCHS = 12
WARMUP_HEAD_EPOCHS = 2          # epochs 0-1 train only the new classifier
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 4                    # early stopping on validation macro-F1
LABEL_SMOOTHING = 0.1
DROPOUT = 0.3
N_BOOTSTRAP = 1000

os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))
for d in (DATA_DIR, RUNS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
