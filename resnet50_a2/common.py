"""
common.py - shared data, model, evaluation and statistics utilities for the
cross-dataset experiment (PlantVillage -> PlantDoc).
42177 Image Processing and Pattern Recognition, A2, Group 2. Author: Teddy Ryu.

Imported by train.py, aggregate.py, gradcam.py and export_onnx.py, so every script
uses the same preprocessing, model definition and metric definitions.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # headless plotting (no window needed)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from torch.utils.data import Dataset
from torchvision import models, transforms

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)     # ImageNet channel statistics

# Report figure palette (validated categorical slots 1-3, light surface).
REGIME_COLORS = {"frozen": "#2a78d6", "layer4": "#eb6834", "full": "#1baf7a"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# =========================================================================== data
def train_tf(size=224, augment=True):
    """Training transform. Augmentation simulates framing/illumination changes
    that are absent from PlantVillage's laboratory images (A1 Section 2.3)."""
    if not augment:
        return eval_tf(size)
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_tf(size=224):
    """Deterministic transform for validation, internal test and external test."""
    return transforms.Compose([
        transforms.Resize(int(size * 256 / 224)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def resolve_paths(df: pd.DataFrame, pv_root: str, pd_root: str) -> pd.DataFrame:
    """Turn the portable (root, relpath) columns of the manifest into absolute paths."""
    roots = {"pv": Path(pv_root), "pd": Path(pd_root)}
    df = df.copy()
    df["path"] = [str(roots[r] / p) for r, p in zip(df["root"], df["relpath"])]
    return df


class ManifestDataset(Dataset):
    """Dataset over manifest rows; returns (image tensor, class index, row index)."""

    def __init__(self, df: pd.DataFrame, tf):
        self.paths, self.ys, self.tf = df["path"].tolist(), df["y"].tolist(), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), self.ys[i], i


def load_labels(mapping_path):
    """Class names in training order (index 0..11) from class_mapping.json."""
    return [c["label"] for c in json.load(open(mapping_path, encoding="utf-8"))["classes"]]


# =========================================================================== model
# Per architecture: name of the classification head, the blocks trained in the
# partial ("layer4") regime, the Grad-CAM target block, and a display name.
ARCH = {
    "resnet50": {"head": "fc", "partial": ["layer4"], "cam": "layer4", "name": "ResNet50",
                 "partial_name": "layer4 fine-tune"},
    # EfficientNet-B0: features.6-8 = last two MBConv stages + final 1x1 conv (the layer4 analogue)
    "efficientnet_b0": {"head": "classifier", "partial": ["features.6", "features.7", "features.8"],
                        "cam": "features.8", "name": "EfficientNet-B0", "partial_name": "last-stage fine-tune"},
}


def build_model(arch: str, num_classes: int, pretrained=True, dropout=0.3):
    """ImageNet-pretrained backbone with a new head: Dropout(p) -> Linear(features, num_classes).
    resnet50        : torchvision IMAGENET1K_V2 weights, 2048-d features (He et al. 2016)
    efficientnet_b0 : torchvision IMAGENET1K_V1 weights, 1280-d features (Tan & Le 2019)"""
    if arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[-1].in_features, num_classes))
    else:
        raise ValueError(f"unknown arch {arch}")
    return m


def build_resnet50(num_classes: int, pretrained=True, dropout=0.3):
    """Backward-compatible shortcut."""
    return build_model("resnet50", num_classes, pretrained, dropout)


def trainable_modules(arch: str, regime: str, stage: str):
    """Module-name prefixes that are updated.
    regime: frozen (head only) | layer4 (last stage(s) + head) | full (everything = [""])
    stage : warmup (always head only, protects pretrained features) | finetune"""
    head = ARCH[arch]["head"]
    if stage == "warmup" or regime == "frozen":
        return [head]
    if regime == "layer4":
        return ARCH[arch]["partial"] + [head]
    return [""]                                  # empty prefix = every parameter


def _matches(name, prefixes):
    """True if parameter/module `name` lies inside one of the blocks named in `prefixes`."""
    return any(p == "" or name == p or name.startswith(p + ".") for p in prefixes)


def set_trainable(model, prefixes):
    """Freeze every parameter except those inside the named blocks."""
    for n, p in model.named_parameters():
        p.requires_grad = _matches(n, prefixes)


def set_train_mode(model, prefixes):
    """model.train() for trainable blocks only. Frozen blocks stay in eval mode so
    their BatchNorm running statistics remain the ImageNet ones (a true linear probe)."""
    if prefixes == [""]:
        model.train(); return
    model.eval()
    for p in prefixes:
        model.get_submodule(p).train()


def count_params(model):
    """Total number of parameters (trainable or not)."""
    return sum(p.numel() for p in model.parameters())


# =========================================================================== metrics
def compute_metrics(y_true, y_pred, n_classes):
    """Accuracy and macro precision/recall/F1, per-class scores and the confusion matrix.
    All 12 classes are always included (labels=0..n-1), so a class that is never predicted
    on PlantDoc counts as F1 = 0 instead of being silently dropped from the macro average."""
    idx = list(range(n_classes))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=idx, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, labels=idx, average="macro", zero_division=0)
    summary = {"n": int(len(y_true)), "accuracy": float(accuracy_score(y_true, y_pred)),
               "precision_macro": float(pm), "recall_macro": float(rm), "f1_macro": float(fm)}
    per_class = {"precision": p, "recall": r, "f1": f, "support": s}
    return summary, per_class, confusion_matrix(y_true, y_pred, labels=idx)


def macro_f1(y_true, y_pred, n_classes):
    """Unweighted mean of the per-class F1 scores (primary metric, robust to class imbalance)."""
    return f1_score(y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0)


def bootstrap_ci(y_true, y_pred, n_classes, n_boot=1000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for macro-F1: resample test images with replacement
    (Efron & Tibshirani 1993). Returns (low, high)."""
    rng = np.random.default_rng(seed)
    y_true, y_pred, n = np.asarray(y_true), np.asarray(y_pred), len(y_true)
    # each replicate draws n image indices with replacement and recomputes macro-F1
    stats = [macro_f1(y_true[i], y_pred[i], n_classes)
             for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    # the 2.5th and 97.5th percentiles of the replicates give the 95% interval
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def mcnemar(y_true, pred_a, pred_b):
    """Exact McNemar test on the same test images (Dietterich 1998).
    b = A right & B wrong, c = A wrong & B right; p from a two-sided binomial test."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    b, c = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    # Images that both models get right (or both get wrong) say nothing about which is better.
    # Under H0 (equal error rates) each discordant image is a fair coin flip:
    # b ~ Binomial(b + c, 0.5).
    p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "p_value": p}


# =========================================================================== saving
def save_eval(y_true, y_pred, probs, ids, labels, name, out: Path, title):
    """Write predictions, per-class table, confusion matrix (csv+png) and report for one test set."""
    n = len(labels)
    summary, per_class, cm = compute_metrics(y_true, y_pred, n)
    pred = pd.DataFrame({"image_id": ids, "y_true": y_true, "y_pred": y_pred,
                         "label_true": [labels[i] for i in y_true], "label_pred": [labels[i] for i in y_pred]})
    pred = pd.concat([pred, pd.DataFrame(probs, columns=[f"p_{i}" for i in range(n)])], axis=1)
    pred.to_csv(out / f"predictions_{name}.csv", index=False)
    pd.DataFrame(per_class, index=labels).to_csv(out / f"per_class_{name}.csv")
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out / f"confusion_{name}.csv")
    plot_cm(cm, labels, title, out / f"confusion_{name}.png")
    (out / f"report_{name}.txt").write_text(classification_report(
        y_true, y_pred, labels=list(range(n)), target_names=labels, digits=4, zero_division=0))
    return summary


# =========================================================================== plots
def _style(ax):
    """Shared minimal axis style for report figures (no top/right frame, light horizontal grid)."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def short(label: str) -> str:
    """Compact class names for axis labels."""
    return (label.replace("Pepper_bell_", "Pepper ").replace("Potato_", "Potato ")
                 .replace("Tomato_", "Tomato ").replace("_", " ")
                 .replace("yellow leaf curl virus", "YLCV").replace("septoria leaf spot", "Septoria"))


def plot_cm(cm, labels, title, path, normalize=True):
    """Row-normalised confusion matrix (each row = recall distribution of a true class)."""
    m = cm.astype(float)
    if normalize:
        m = m / np.clip(m.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    names = [short(l) for l in labels]
    ax.set_xticks(range(len(labels)), names, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), names, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if m[i, j] > 0.5 else INK)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class"); ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Proportion of true class")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_curves(hist, path, title):
    """Training curves (loss; train accuracy vs validation macro-F1). Warm-up epochs are shaded."""
    h = pd.DataFrame(hist)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(h.epoch, h.train_loss, color=REGIME_COLORS["frozen"], lw=2, label="Train")
    ax[0].plot(h.epoch, h.val_loss, color=REGIME_COLORS["layer4"], lw=2, label="Validation")
    ax[0].set_title("Loss", fontsize=10)
    ax[1].plot(h.epoch, h.train_acc, color=REGIME_COLORS["frozen"], lw=2, label="Train accuracy")
    ax[1].plot(h.epoch, h.val_f1_macro, color=REGIME_COLORS["layer4"], lw=2, label="Validation macro-F1")
    ax[1].set_title("PlantVillage only (PlantDoc never used)", fontsize=10)
    warm = h[h.stage == "warmup"].epoch
    for a in ax:
        _style(a); a.set_xlabel("Epoch"); a.legend(frameon=False, fontsize=8)
        if len(warm):
            a.axvspan(warm.min() - 0.5, warm.max() + 0.5, color=GRID, alpha=0.6, lw=0)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
