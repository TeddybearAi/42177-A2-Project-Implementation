"""
Evaluate trained MobileNetV2 seeds and compare them with the group's other models.

E1 = PlantVillage test split (in-domain), E2 = Cropped PlantDoc (cross-domain).
Each final checkpoint sees PlantDoc once, after training is complete.

Outputs (outputs/results/):
  metrics_per_seed.csv     accuracy, macro P/R/F1 per seed and split (+ bootstrap CI for E2)
  summary.md               headline table, per-seed metrics, per-class PlantDoc scores, cost
  per_class_e2.csv         per-class precision / recall / F1 on PlantDoc (mean over seeds)
  efficiency.csv           parameters, GMACs, GPU latency per image
  comparison.csv/.md       MobileNetV2 vs baseline CNN, ResNet50 (layer4), YOLO11s-cls
                           (only if the main pipeline's predictions are available)
  fig_confusion.png        row-normalised confusion matrices, E1 and E2 (first seed)
  fig_training.png         training loss and validation macro-F1 per epoch
  fig_comparison.png       E1 vs E2 macro-F1 against the other models
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, precision_score, recall_score)

import mnv2_config as C
from data import make_loader
from model import build_mobilenetv2
from train import predict

LABELS = list(range(C.NUM_CLASSES))
OTHERS = {"cnn": "Baseline CNN", "resnet50-layer4": "ResNet50 (layer4)", "yolo11s": "YOLO11s-cls"}


def evaluate_seed(seed):
    """Run the best checkpoint of one seed on val, test (E1) and PlantDoc (E2)."""
    run = C.RUNS_DIR / f"mobilenetv2_s{seed}"
    if (run / "preds.npz").exists():
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_mobilenetv2(pretrained=False).to(device)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    out = {}
    for split in ["val", "test", "external"]:
        p, y = predict(model, make_loader(split), device)
        out[f"{split}_probs"], out[f"{split}_labels"] = p, y
        print(f"{run.name} {split:9s} acc {(p.argmax(1) == y).mean():.4f}")
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


def efficiency():
    """Parameters, GMACs (thop; 1 MAC = 2 FLOPs) and GPU latency for one 224x224 image."""
    from thop import profile
    net = build_mobilenetv2(pretrained=False).eval()
    params = sum(p.numel() for p in net.parameters()) / 1e6
    macs, _ = profile(net, inputs=(torch.randn(1, 3, C.IMG_SIZE, C.IMG_SIZE),), verbose=False)
    row = dict(model="MobileNetV2", params_M=params, GMACs=macs / 1e9, GFLOPs=2 * macs / 1e9)
    if torch.cuda.is_available():
        net = build_mobilenetv2(pretrained=False).cuda().eval()
        x = torch.randn(1, 3, C.IMG_SIZE, C.IMG_SIZE, device="cuda")
        with torch.no_grad():
            for _ in range(20):
                net(x)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(200):
                net(x)
            torch.cuda.synchronize()
        row["gpu_latency_ms"] = (time.perf_counter() - t0) / 200 * 1000
    return pd.DataFrame([row])


def load_main_runs(seeds):
    """Predictions of the main models (same split files), if available."""
    runs = {}
    for cfg in OTHERS:
        for s in seeds:
            f = C.MAIN_RUNS_DIR / f"{cfg}_s{s}" / "preds.npz"
            if f.exists():
                runs[(cfg, s)] = dict(np.load(f))
    return runs


def compare(runs, seeds):
    """MobileNetV2 vs the main models: macro-F1 table and paired McNemar tests on E2."""
    main = load_main_runs(seeds)
    if not main:
        print("main pipeline predictions not found; skipping comparison")
        return
    allruns = {("mobilenetv2", s): r for s, r in runs.items()} | main
    names = {**OTHERS, "mobilenetv2": "MobileNetV2"}
    y_ext = next(iter(runs.values()))["external_labels"]
    for (cfg, s), r in main.items():                      # same images in the same order
        assert np.array_equal(r["external_labels"], y_ext), f"{cfg}_s{s}: different PlantDoc order"

    tab = []
    for cfg in names:
        f = [(f1_score(r["test_labels"], r["test_probs"].argmax(1), average="macro"),
              f1_score(r["external_labels"], r["external_probs"].argmax(1), average="macro"))
             for (c, s), r in allruns.items() if c == cfg]
        if f:
            f = np.array(f)
            tab.append(dict(model=names[cfg], seeds=len(f), E1_mean=f[:, 0].mean(), E1_sd=f[:, 0].std(ddof=1),
                            E2_mean=f[:, 1].mean(), E2_sd=f[:, 1].std(ddof=1),
                            retention=(f[:, 1] / f[:, 0]).mean()))
    tab = pd.DataFrame(tab)
    tab.to_csv(C.RESULTS_DIR / "comparison.csv", index=False)

    mc = []
    for s in seeds:
        if s not in runs:
            continue
        pm = runs[s]["external_probs"].argmax(1)
        for cfg in OTHERS:
            if (cfg, s) in main:
                po = main[(cfg, s)]["external_probs"].argmax(1)
                a = int(((pm == y_ext) & (po != y_ext)).sum()); b = int(((pm != y_ext) & (po == y_ext)).sum())
                mc.append(dict(seed=s, other=OTHERS[cfg], mnv2_right_other_wrong=a, other_right_mnv2_wrong=b,
                               p_value=binomtest(a, a + b, 0.5).pvalue if a + b else 1.0))
    mc = pd.DataFrame(mc)
    mc.to_csv(C.RESULTS_DIR / "mcnemar_vs_others.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(tab)); w = 0.38
    for off, col, colour, lab in [(-w / 2, "E1", "#2a6f97", "E1: PlantVillage test"),
                                  (w / 2, "E2", "#bc4749", "E2: Cropped PlantDoc")]:
        b = ax.bar(x + off, tab[f"{col}_mean"], w, yerr=tab[f"{col}_sd"], capsize=3, color=colour, label=lab)
        ax.bar_label(b, fmt="%.3f", fontsize=7, padding=2)
    ax.axhline(0.081, ls="--", c="grey", lw=1, label="Random guess on E2 (0.08)")
    ax.set_xticks(x, tab.model); ax.set_ylabel("Macro-F1"); ax.set_ylim(0, 1.1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout(); fig.savefig(C.RESULTS_DIR / "fig_comparison.png", dpi=200); plt.close(fig)

    md = ["# MobileNetV2 vs the group's other models (mean ± sd over seeds)", "",
          tab.to_markdown(index=False, floatfmt=".3f"), "",
          "## McNemar tests on Cropped PlantDoc (MobileNetV2 vs each model, same seed)", "",
          mc.to_markdown(index=False, floatfmt=".4f")]
    (C.RESULTS_DIR / "comparison.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


def summarize(seeds):
    runs = {s: dict(np.load(C.RUNS_DIR / f"mobilenetv2_s{s}" / "preds.npz")) for s in seeds
            if (C.RUNS_DIR / f"mobilenetv2_s{s}" / "preds.npz").exists()}
    if not runs:
        print("nothing to summarise")
        return

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

    f1 = per_seed.pivot(index="seed", columns="split", values="macro_f1")
    acc = per_seed.pivot(index="seed", columns="split", values="accuracy")
    ext = per_seed[per_seed.split == "external"]
    ms = lambda v: f"{v.mean():.3f} ± {v.std():.3f}" if len(v) > 1 else f"{v.mean():.3f}"
    summary = pd.DataFrame([{
        "Seeds": len(runs), "Val macro-F1": ms(f1["val"]),
        "E1 accuracy": ms(acc["test"]), "E1 macro-F1": ms(f1["test"]),
        "E2 accuracy": ms(acc["external"]), "E2 macro-F1": ms(f1["external"]),
        "E2 95% CI (mean)": f"[{ext.f1_ci_low.mean():.3f}, {ext.f1_ci_high.mean():.3f}]",
        "ΔF1 (E1−E2)": ms(f1["test"] - f1["external"]), "Retention": ms(f1["external"] / f1["test"]),
    }])
    summary.to_csv(C.RESULTS_DIR / "summary.csv", index=False)

    prf = np.mean([precision_recall_fscore_support(r["external_labels"], r["external_probs"].argmax(1),
                                                   labels=LABELS, zero_division=0)[:3] for r in runs.values()], axis=0)
    per_class = pd.DataFrame({"class": C.CLASS_NAMES, "precision": prf[0], "recall": prf[1], "f1": prf[2]})
    per_class.to_csv(C.RESULTS_DIR / "per_class_e2.csv", index=False)
    eff = efficiency()
    eff.to_csv(C.RESULTS_DIR / "efficiency.csv", index=False)

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
        ax.set_title(f"MobileNetV2, {title} (seed {first})", fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout(); fig.savefig(C.RESULTS_DIR / "fig_confusion.png", dpi=200); plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(6.5, 3.5))
    ax2 = ax1.twinx()
    for s in runs:
        h = pd.read_csv(C.RUNS_DIR / f"mobilenetv2_s{s}" / "history.csv")
        ax1.plot(h.epoch + 1, h.train_loss, ls="--", lw=1, label=f"train loss, seed {s}")
        ax2.plot(h.epoch + 1, h.val_macro_f1, marker="o", ms=2.5, label=f"val macro-F1, seed {s}")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Training loss"); ax2.set_ylabel("Validation macro-F1")
    fig.legend(frameon=False, fontsize=7, loc="center right", bbox_to_anchor=(0.88, 0.5))
    fig.tight_layout(); fig.savefig(C.RESULTS_DIR / "fig_training.png", dpi=200); plt.close(fig)

    md = ["# MobileNetV2 results", "", summary.to_markdown(index=False), "",
          "## Cost", "", eff.to_markdown(index=False, floatfmt=".2f"), "",
          "## Per seed", "", per_seed.to_markdown(index=False, floatfmt=".3f"), "",
          "## Per class on Cropped PlantDoc (mean over seeds)", "",
          per_class.to_markdown(index=False, floatfmt=".2f")]
    (C.RESULTS_DIR / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:3]))
    compare(runs, seeds)
