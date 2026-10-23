"""
Numbers and figures for the A2 baseline CNN report section (new resize style).

Reads the predictions written by run_cnn.py (runs/cnn_s<seed>/preds.npz: PlantVillage val/test and
the 1,097 PlantDoc images, all with resize 256 + centre-crop 224) and the earlier CNN's 1,097-image
re-test (WorkPlace/CNN/outputs/results_1097) for the old-vs-new comparison. Nothing is retrained.

Outputs (outputs/report/):
  numbers.json                 every figure quoted in the text
  per_class_f1.csv             per-class precision / recall / F1 on E1 and E2 (mean of 3 seeds)
  prediction_share.csv         share of true vs predicted labels on PlantDoc (representative seed)
  top_confusions.csv           largest off-diagonal cells of the row-normalised confusion matrix
  old_vs_new.csv               earlier CNN vs this one, same 1,097 images and evaluation pre-processing
  predictions_1097.csv         per-image prediction and confidence for each seed
  fig_training_curves.png, fig_per_class_f1.png, fig_prediction_share.png, fig_confusion_e2.png,
  fig_gradcam_plantdoc.png, fig_gradcam_plantvillage.png

    python report_analysis.py
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

import cnn_config as C
from data import eval_transform, load_split
from evaluate import bootstrap_ci, metrics
from model import BaselineCNN

OLD = Path(r"D:\IPPR\WorkPlace\CNN\outputs\results_1097")
REP = C.OUT_DIR / "report"
REP.mkdir(exist_ok=True)
LABELS = list(range(C.NUM_CLASSES))
SHORT = [n.replace("_", " ").replace("Tomato ", "T. ").replace("Potato ", "Po. ").replace("Pepper ", "Pe. ")
         for n in C.CLASS_NAMES]
BLUE, ORANGE, GREEN, RED = "#3a6ea5", "#e07a1f", "#2d6a4f", "#bc4749"


def main():
    num = {}
    runs = {s: np.load(C.RUNS_DIR / f"cnn_s{s}" / "preds.npz") for s in C.SEEDS}
    e2 = {s: (r["external_labels"], r["external_probs"]) for s, r in runs.items()}
    e1 = {s: (r["test_labels"], r["test_probs"]) for s, r in runs.items()}
    y = e2[C.SEEDS[0]][0]
    f1s = {s: f1_score(y, e2[s][1].argmax(1), labels=LABELS, average="macro") for s in C.SEEDS}
    rep = sorted(f1s, key=f1s.get)[1]                       # median external macro-F1
    num["representative_seed"] = rep

    # per-seed metrics with bootstrap CI
    rows = []
    for s in C.SEEDS:
        m2, m1 = metrics(y, e2[s][1].argmax(1)), metrics(e1[s][0], e1[s][1].argmax(1))
        lo, hi = bootstrap_ci(y, e2[s][1].argmax(1))
        rows.append(dict(seed=s, e1_accuracy=m1["accuracy"], e1_macro_f1=m1["macro_f1"], **m2, ci_low=lo, ci_high=hi,
                         retention=m2["macro_f1"] / m1["macro_f1"]))
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(REP / "metrics_per_seed.csv", index=False)

    rng = np.random.default_rng(0)
    num["random_macro_f1"] = float(np.mean([f1_score(y, rng.integers(0, 12, len(y)), labels=LABELS, average="macro")
                                            for _ in range(1000)]))

    # old (first CNN, re-tested with the same pre-processing) vs new
    old = pd.read_csv(OLD / "metrics_per_seed.csv")
    old = old[old.prep == "resize256_crop224"]
    cmp = pd.DataFrame([dict(model="Earlier CNN (resize-224 training, seeds 0-2)", e1=old.e1_macro_f1.mean(),
                             e1_sd=old.e1_macro_f1.std(), e2=old.macro_f1.mean(), e2_sd=old.macro_f1.std(),
                             acc=old.accuracy.mean(), acc_sd=old.accuracy.std(), ret=old.retention.mean()),
                        dict(model="New resize style (seeds 42-44)", e1=per_seed.e1_macro_f1.mean(),
                             e1_sd=per_seed.e1_macro_f1.std(), e2=per_seed.macro_f1.mean(),
                             e2_sd=per_seed.macro_f1.std(), acc=per_seed.accuracy.mean(),
                             acc_sd=per_seed.accuracy.std(), ret=per_seed.retention.mean())])
    cmp.to_csv(REP / "old_vs_new.csv", index=False)
    # paired bootstrap of the seed-averaged macro-F1 difference (new - old), same resampled images
    old_p = [np.load(OLD / f"preds_resize256_crop224_s{s}.npz")["probs"].argmax(1) for s in (0, 1, 2)]
    new_p = [e2[s][1].argmax(1) for s in C.SEEDS]
    idx = np.random.default_rng(0).integers(0, len(y), size=(2000, len(y)))
    mf = lambda yy, pp: f1_score(yy, pp, labels=LABELS, average="macro", zero_division=0)
    diffs = [np.mean([mf(y[i], p[i]) for p in new_p]) - np.mean([mf(y[i], p[i]) for p in old_p]) for i in idx]
    num["new_minus_old"] = dict(diff=float(np.mean([mf(y, p) for p in new_p]) - np.mean([mf(y, p) for p in old_p])),
                                ci=[float(v) for v in np.percentile(diffs, [2.5, 97.5])])

    # per-class precision / recall / F1
    rows = []
    for name, d in [("E1", e1), ("E2", e2)]:
        prf = np.mean([precision_recall_fscore_support(d[s][0], d[s][1].argmax(1), labels=LABELS, zero_division=0)[:3]
                       for s in C.SEEDS], axis=0)
        for c in LABELS:
            rows.append(dict(test=name, cls=C.CLASS_NAMES[c], precision=prf[0][c], recall=prf[1][c], f1=prf[2][c]))
    pc = pd.DataFrame(rows)
    pc.to_csv(REP / "per_class_f1.csv", index=False)
    wide = pc.pivot(index="cls", columns="test", values="f1").loc[C.CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(12); w = 0.38
    ax.bar(x - w / 2, wide.E1, w, color=BLUE, label="PlantVillage test (E1)")
    b = ax.bar(x + w / 2, wide.E2, w, color=ORANGE, label="PlantDoc (E2)")
    ax.bar_label(b, fmt="%.2f", fontsize=6.5, padding=1)
    ax.set_xticks(x, SHORT, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("F1 (mean of 3 seeds)"); ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); fig.savefig(REP / "fig_per_class_f1.png", dpi=200); plt.close(fig)

    # prediction share and confusion matrix for the representative run
    p_rep = e2[rep][1].argmax(1)
    share = pd.DataFrame({"true_share": np.bincount(y, minlength=12) / len(y),
                          "pred_share": np.bincount(p_rep, minlength=12) / len(y)}, index=C.CLASS_NAMES)
    share["pred_share_mean_seeds"] = np.mean([np.bincount(e2[s][1].argmax(1), minlength=12) / len(y) for s in C.SEEDS], 0)
    share.to_csv(REP / "prediction_share.csv")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(x - w / 2, share.true_share, w, color=BLUE, label="True class")
    ax.bar(x + w / 2, share.pred_share, w, color=ORANGE, label="Predicted class")
    ax.set_xticks(x, SHORT, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("Share of PlantDoc images"); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(REP / "fig_prediction_share.png", dpi=200); plt.close(fig)

    cm = confusion_matrix(y, p_rep, labels=LABELS, normalize="true")
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cm, cmap="Oranges", vmin=0, vmax=1)
    for i in LABELS:
        for j in LABELS:
            if cm[i, j] >= 0.005:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xticks(LABELS, SHORT, rotation=45, ha="right", fontsize=7.5); ax.set_yticks(LABELS, SHORT, fontsize=7.5)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    fig.colorbar(im, fraction=0.046, pad=0.03)
    fig.tight_layout(); fig.savefig(REP / "fig_confusion_e2.png", dpi=200); plt.close(fig)
    off = [(C.CLASS_NAMES[i], C.CLASS_NAMES[j], cm[i, j], int((y == i).sum())) for i in LABELS for j in LABELS if i != j]
    pd.DataFrame(off, columns=["true", "predicted", "row_share", "true_images"]).nlargest(10, "row_share") \
        .to_csv(REP / "top_confusions.csv", index=False)

    # per-image predictions (relative paths)
    ext = pd.read_csv(C.PD_MANIFEST)
    pred = pd.DataFrame({"image": ext.path.map(lambda p: Path(p).relative_to(C.PD_DIR).as_posix()),
                         "true_label": [C.CLASS_NAMES[i] for i in y]})
    for s in C.SEEDS:
        pred[f"pred_s{s}"] = [C.CLASS_NAMES[i] for i in e2[s][1].argmax(1)]
        pred[f"conf_s{s}"] = e2[s][1].max(1).round(4)
    pred.to_csv(REP / "predictions_1097.csv", index=False)

    # training curves
    hist = {s: pd.read_csv(C.RUNS_DIR / f"cnn_s{s}" / "history.csv") for s in C.SEEDS}
    num["training"] = {s: dict(epochs=len(h), best_epoch=int(h.val_macro_f1.idxmax()) + 1,
                               best_val_macro_f1=float(h.val_macro_f1.max()), minutes=float(h.seconds.sum() / 60),
                               first_epoch_above_095=int((h.val_macro_f1 > 0.95).idxmax()) + 1)
                       for s, h in hist.items()}
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.9))
    for s, h in hist.items():
        axes[0].plot(h.epoch + 1, h.train_loss, label=f"seed {s}")
        axes[1].plot(h.epoch + 1, h.val_macro_f1, label=f"seed {s}")
    axes[0].set_ylabel("Training loss"); axes[1].set_ylabel("Validation macro-F1 (PlantVillage)")
    for a in axes:
        a.set_xlabel("Epoch"); a.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(REP / "fig_training_curves.png", dpi=200); plt.close(fig)

    # model size and indicative latency (batch 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = BaselineCNN()
    net.load_state_dict(torch.load(C.RUNS_DIR / f"cnn_s{rep}" / "best.pt", map_location="cpu"))
    net.eval()
    num["parameters"] = sum(p.numel() for p in net.parameters())
    xb = torch.randn(1, 3, C.IMG_SIZE, C.IMG_SIZE)
    for dev in ["cpu", "cuda"] if device.type == "cuda" else ["cpu"]:
        m, xd = net.to(dev), xb.to(dev)
        with torch.no_grad():
            for _ in range(20):
                m(xd)
            if dev == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(200):
                m(xd)
            if dev == "cuda":
                torch.cuda.synchronize()
        num[f"latency_ms_{dev}"] = (time.perf_counter() - t) / 200 * 1000

    gradcam(net.to(device), device, num)
    (REP / "numbers.json").write_text(json.dumps(num, indent=2), encoding="utf-8")
    print(json.dumps(num, indent=2))
    print(per_seed.round(3).to_string())


def gradcam(net, device, num):
    """Fixed sample: two images per class (seed 42), chosen without looking at predictions."""
    tf = eval_transform()
    for tag, df in [("plantdoc", load_split("external")), ("plantvillage", load_split("test"))]:
        sample = df.groupby("label", group_keys=False).sample(2, random_state=42).sort_values("label")
        fig, axes = plt.subplots(4, 6, figsize=(12, 9.2))
        right = 0
        for ax, (_, r) in zip(axes.flat, sample.iterrows()):
            x = tf(Image.open(r.path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = torch.softmax(net(x)[0], 0).cpu().numpy()
            pred = int(prob.argmax())
            with GradCAM(model=net, target_layers=[net.features[-1][-2]]) as cam:
                heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred)])[0]
            rgb = x[0].cpu().numpy().transpose(1, 2, 0) * np.array(C.IMAGENET_STD) + np.array(C.IMAGENET_MEAN)
            ax.imshow(show_cam_on_image(np.clip(rgb, 0, 1).astype(np.float32), heat, use_rgb=True))
            ok = pred == r.label
            right += ok
            ax.set_title(f"T: {SHORT[r.label]}\nP: {SHORT[pred]} ({prob[pred]:.2f})", fontsize=8,
                         color=GREEN if ok else RED)
            ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(); fig.savefig(REP / f"fig_gradcam_{tag}.png", dpi=150); plt.close(fig)
        num[f"gradcam_{tag}_correct"] = int(right)


if __name__ == "__main__":
    main()
