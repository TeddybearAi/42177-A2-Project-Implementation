"""
Evaluate trained CNN seeds.

E1 = PlantVillage test split (in-domain), E2 = Cropped PlantDoc (cross-domain).
Each final checkpoint sees PlantDoc once, after training is complete.

Outputs (outputs/results/):
  metrics_per_seed.csv    accuracy, macro P/R/F1 per seed and split (+ bootstrap CI for E2)
  summary.csv / .md       mean +- sd over seeds, domain gap and retention ratio
  per_class_e2.csv        per-class precision / recall / F1 on PlantDoc (mean over seeds)
  fig_confusion.png       row-normalised confusion matrices, E1 and E2 (first seed)
  fig_training.png        training loss and validation macro-F1 per epoch
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, precision_score, recall_score)

import cnn_config as C
from data import make_loader
from model import BaselineCNN
from train import predict

LABELS = list(range(C.NUM_CLASSES))


def evaluate_seed(seed):
    """Run the best checkpoint of one seed on val, test (E1) and PlantDoc (E2)."""
    run = C.RUNS_DIR / f"cnn_s{seed}"
    if (run / "preds.npz").exists():
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineCNN().to(device)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    out = {}
    for split in ["val", "test", "external"]:
        p, y = predict(model, make_loader(split), device)
        out[f"{split}_probs"], out[f"{split}_labels"] = p, y
        print(f"cnn_s{seed} {split:9s} acc {(p.argmax(1) == y).mean():.4f}")
    np.savez_compressed(run / "preds.npz", **out)


def metrics(y, p):
    return dict(accuracy=accuracy_score(y, p),
                macro_precision=precision_score(y, p, labels=LABELS, average="macro", zero_division=0),
                macro_recall=recall_score(y, p, labels=LABELS, average="macro", zero_division=0),
                macro_f1=f1_score(y, p, labels=LABELS, average="macro", zero_division=0))


def bootstrap_ci(y, p, n=C.N_BOOTSTRAP, seed=0):
    """95% percentile bootstrap interval of macro-F1 over resampled test images."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    s = [f1_score(y[i], p[i], labels=LABELS, average="macro", zero_division=0) for i in idx]
    return np.percentile(s, [2.5, 97.5])


def summarize(seeds):
    runs = {s: dict(np.load(C.RUNS_DIR / f"cnn_s{s}" / "preds.npz")) for s in seeds
            if (C.RUNS_DIR / f"cnn_s{s}" / "preds.npz").exists()}
    if not runs:
        print("nothing to summarise")
        return

    # --- per seed
    rows = []
    for s, r in runs.items():
        for split in ["val", "test", "external"]:
            y, p = r[f"{split}_labels"], r[f"{split}_probs"].argmax(1)
            row = dict(seed=s, split=split, n=len(y), **metrics(y, p))
            if split == "external":
                row["f1_ci_low"], row["f1_ci_high"] = bootstrap_ci(y, p)
            rows.append(row)
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(C.RESULTS_DIR / "metrics_per_seed.csv", index=False)

    # --- mean +- sd, gap and retention
    f1 = per_seed.pivot(index="seed", columns="split", values="macro_f1")
    acc = per_seed.pivot(index="seed", columns="split", values="accuracy")
    gap, ret = f1["test"] - f1["external"], f1["external"] / f1["test"]
    ext = per_seed[per_seed.split == "external"]
    ms = lambda v: f"{v.mean():.3f} ± {v.std():.3f}" if len(v) > 1 else f"{v.mean():.3f}"
    summary = pd.DataFrame([{
        "Seeds": len(runs),
        "Val macro-F1": ms(f1["val"]),
        "E1 accuracy": ms(acc["test"]), "E1 macro-F1": ms(f1["test"]),
        "E2 accuracy": ms(acc["external"]), "E2 macro-F1": ms(f1["external"]),
        "E2 95% CI (mean)": f"[{ext.f1_ci_low.mean():.3f}, {ext.f1_ci_high.mean():.3f}]",
        "ΔF1 (E1−E2)": ms(gap), "Retention": ms(ret),
    }])
    summary.to_csv(C.RESULTS_DIR / "summary.csv", index=False)

    # --- per-class on PlantDoc, mean over seeds
    prf = np.mean([precision_recall_fscore_support(r["external_labels"], r["external_probs"].argmax(1),
                                                   labels=LABELS, zero_division=0)[:3] for r in runs.values()], axis=0)
    per_class = pd.DataFrame({"class": C.CLASS_NAMES, "precision": prf[0], "recall": prf[1], "f1": prf[2]})
    per_class.to_csv(C.RESULTS_DIR / "per_class_e2.csv", index=False)

    # --- confusion matrices (first seed)
    first = min(runs)
    short = [n.replace("Tomato_", "T.").replace("Potato_", "Po.").replace("Pepper_", "Pe.") for n in C.CLASS_NAMES]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, (split, title) in zip(axes, [("test", "E1: PlantVillage test"), ("external", "E2: Cropped PlantDoc")]):
        r = runs[first]
        cm = confusion_matrix(r[f"{split}_labels"], r[f"{split}_probs"].argmax(1), labels=LABELS, normalize="true")
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for (a, b), v in np.ndenumerate(cm):
            if v >= 0.05:
                ax.text(b, a, f"{v:.2f}"[1:] if round(v, 2) < 1 else "1", ha="center", va="center",
                        fontsize=6, color="white" if v > 0.5 else "black")
        ax.set_xticks(LABELS, short, rotation=90, fontsize=7); ax.set_yticks(LABELS, short, fontsize=7)
        ax.set_title(f"Baseline CNN, {title} (seed {first})", fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout(); fig.savefig(C.RESULTS_DIR / "fig_confusion.png", dpi=200); plt.close(fig)

    # --- training curves
    fig, ax1 = plt.subplots(figsize=(6.5, 3.5))
    ax2 = ax1.twinx()
    for s in runs:
        h = pd.read_csv(C.RUNS_DIR / f"cnn_s{s}" / "history.csv")
        ax1.plot(h.epoch + 1, h.train_loss, ls="--", lw=1, label=f"train loss, seed {s}")
        ax2.plot(h.epoch + 1, h.val_macro_f1, marker="o", ms=2.5, label=f"val macro-F1, seed {s}")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Training loss"); ax2.set_ylabel("Validation macro-F1")
    fig.legend(frameon=False, fontsize=7, loc="center right", bbox_to_anchor=(0.88, 0.5))
    fig.tight_layout(); fig.savefig(C.RESULTS_DIR / "fig_training.png", dpi=200); plt.close(fig)

    md = ["# Baseline CNN results", "", summary.to_markdown(index=False), "",
          "## Per seed", "", per_seed.to_markdown(index=False, floatfmt=".3f"), "",
          "## Per class on Cropped PlantDoc (mean over seeds)", "",
          per_class.to_markdown(index=False, floatfmt=".2f")]
    (C.RESULTS_DIR / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:3]))
