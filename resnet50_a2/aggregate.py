"""
aggregate.py - collect every finished run (ResNet50 and, if trained, EfficientNet-B0)
and produce the tables, statistics and figures used in the A2 Results section.
42177 Image Processing and Pattern Recognition, A2, Group 2. Author: Teddy Ryu.

Outputs (in runs/summary/ by default)
-------------------------------------
runs_table.csv            one row per run (model x regime x seed)
model_summary.csv / .md   mean +/- SD over seeds per model/regime (paste the .md into the report)
mcnemar.csv               paired McNemar tests on identical test images
extra_models.csv          macro-F1 + bootstrap CI of teammates' models (if --extra given)
fig_internal_vs_external.png       E1 vs E2 macro-F1 per model/regime (mean, SD, individual seeds)
fig_per_class_f1.png               per-class F1, internal vs external
fig_external_prediction_share.png  true vs predicted class share on PlantDoc (collapse check)
fig_confusion_<set>_<model>.png    confusion matrices of each representative run

Comparing with teammates' models (same shared manifest, same external images)
-----------------------------------------------------------------------------
python aggregate.py --extra "Baseline CNN=../cnn/predictions_external.csv" --extra "YOLO11s-cls=../yolo/predictions_external.csv"
Each file needs at least the columns image_id, y_true, y_pred (see PREDICTIONS_FORMAT.md).
"""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ARCH, GRID, INK, INK_2, _style, bootstrap_ci, load_labels, macro_f1, mcnemar, short

ARCH_ORDER = ["resnet50", "efficientnet_b0"]
REGIMES = ["frozen", "layer4", "full"]
REGIME_NAMES = {"frozen": "frozen backbone", "full": "full fine-tune"}
SHORT_ARCH = {"resnet50": "ResNet50", "efficientnet_b0": "EffNet-B0"}
E1_COLOR, E2_COLOR = "#2a78d6", "#eb6834"      # validated categorical slots 1-2


def model_key(arch, regime):
    """Internal key for one configuration, e.g. 'resnet50|full'."""
    return f"{arch}|{regime}"


def model_label(key, newline=False):
    """Display name for tables and figures, e.g. 'ResNet50 full fine-tune'."""
    arch, regime = key.split("|")
    reg = ARCH[arch]["partial_name"] if regime == "layer4" else REGIME_NAMES[regime]
    return f"{SHORT_ARCH[arch]}{chr(10) if newline else ' '}{reg}"


def load_runs(runs: Path):
    """One row per finished run, read from each runs/<run>/results.json."""
    rows = []
    for f in sorted(runs.glob("*/results.json")):
        r = json.load(open(f))
        arch = r.get("arch", "resnet50")          # runs made before --arch existed are ResNet50
        rows.append({"run": f.parent.name, "arch": arch, "regime": r["regime"], "model": model_key(arch, r["regime"]),
                     "seed": r["seed"], "best_epoch": r["best_epoch"], "train_minutes": r["train_minutes"],
                     "E1_accuracy": r["internal"]["accuracy"], "E1_f1_macro": r["internal"]["f1_macro"],
                     "E2_accuracy": r["external"]["accuracy"], "E2_f1_macro": r["external"]["f1_macro"],
                     "E2_precision_macro": r["external"]["precision_macro"],
                     "E2_recall_macro": r["external"]["recall_macro"],
                     "E2_f1_ci_low": r["external"]["f1_macro_ci95"][0],
                     "E2_f1_ci_high": r["external"]["f1_macro_ci95"][1],
                     "gap_f1_macro": r["gap_f1_macro"], "retention_f1_macro": r["retention_f1_macro"],
                     "params_million": r["params_million"], "latency_ms_batch1": r["latency_ms_batch1"]})
    return pd.DataFrame(rows)


def preds(runs: Path, run: str, which: str):
    """Per-image predictions of one run, indexed and sorted by image_id so runs can be paired."""
    return pd.read_csv(runs / run / f"predictions_{which}.csv").set_index("image_id").sort_index()


def representative(d: pd.DataFrame) -> str:
    """Run whose external macro-F1 is the median of the seeds (used for figures and model comparisons)."""
    d = d.sort_values("E2_f1_macro").reset_index(drop=True)
    return d.loc[(len(d) - 1) // 2, "run"]


def fmt(m, s):
    """'mean ± SD' with three decimals (mean only when there is a single seed)."""
    return f"{m:.3f} ± {s:.3f}" if not np.isnan(s) else f"{m:.3f}"


def main():
    """Build every summary table, statistical test and figure from the finished runs."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", default="")
    ap.add_argument("--mapping", default=str(Path(__file__).with_name("class_mapping.json")))
    ap.add_argument("--extra", action="append", default=[], help='"Model name=path/to/predictions_external.csv"')
    ap.add_argument("--target", type=float, default=0.40, help="A1 provisional E2 macro-F1 target")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()

    runs = Path(a.runs_dir); out = Path(a.out or runs / "summary"); out.mkdir(parents=True, exist_ok=True)
    labels = load_labels(a.mapping); n_cls = len(labels)
    df = load_runs(runs)
    if df.empty:
        raise SystemExit(f"no finished runs found under {runs}")
    models = [model_key(ar, r) for ar in ARCH_ORDER for r in REGIMES if model_key(ar, r) in set(df.model)]
    df = df.sort_values(["arch", "regime", "seed"]); df.to_csv(out / "runs_table.csv", index=False)

    # ---------------------------------------------------------------- mean +/- SD per model
    cols = ["E1_accuracy", "E1_f1_macro", "E2_accuracy", "E2_f1_macro", "gap_f1_macro", "retention_f1_macro"]
    g = df.groupby("model")
    mean, sd = g[cols].mean().reindex(models), g[cols].std(ddof=1).reindex(models)
    n = g.size().reindex(models)
    eff = g[["params_million", "latency_ms_batch1"]].mean().reindex(models)
    rep = {m: representative(df[df.model == m]) for m in models}
    summary = pd.concat([mean.add_suffix("_mean"), sd.add_suffix("_sd"), eff], axis=1)
    summary["n_seeds"], summary["representative_run"] = n, pd.Series(rep)
    summary.index = [model_label(m) for m in models]
    summary.to_csv(out / "model_summary.csv")

    md = ["| Model | Seeds | E1 accuracy | E1 macro-F1 | E2 accuracy | E2 macro-F1 | Gap (E1−E2) | Retention "
          "| Params (M) | Latency (ms) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        md.append(f"| {model_label(m)} | {n[m]} | " + " | ".join(fmt(mean.loc[m, c], sd.loc[m, c]) for c in cols)
                  + f" | {eff.loc[m, 'params_million']:.1f} | {eff.loc[m, 'latency_ms_batch1']:.1f} |")
    md.append("\nValues are mean ± SD over training seeds. E1 = PlantVillage internal test; E2 = PlantDoc external "
              "test (never used for training or selection). Latency = single-image GPU inference time.")
    (out / "model_summary.md").write_text("\n".join(md), encoding="utf-8")

    # ---------------------------------------------------------------- McNemar (same seed, same images)
    tests = []
    pairs = [(x, y) for i, x in enumerate(models) for y in models[i + 1:]
             if x.split("|")[0] == y.split("|")[0] or x.split("|")[1] == y.split("|")[1]]  # same arch or same regime
    for which in ["external", "internal"]:
        for ma, mb in pairs:
            ra, rb = df[df.model == ma].set_index("seed").run, df[df.model == mb].set_index("seed").run
            for s in sorted(set(ra.index) & set(rb.index)):
                pa, pb = preds(runs, ra[s], which), preds(runs, rb[s], which)
                pb = pb.loc[pa.index]
                t = mcnemar(pa.y_true, pa.y_pred, pb.y_pred)
                tests.append({"test_set": which, "model_a": ra[s], "model_b": rb[s],
                              "f1_a": macro_f1(pa.y_true, pa.y_pred, n_cls),
                              "f1_b": macro_f1(pa.y_true, pb.y_pred, n_cls), **t})

    # ---------------------------------------------------------------- teammates' models
    extra_rows = []
    for spec in a.extra:
        name, path = spec.split("=", 1)
        pe = pd.read_csv(path).set_index("image_id").sort_index()
        for m in models:
            po = preds(runs, rep[m], "external")
            ids = po.index.intersection(pe.index)
            if len(ids) != len(po):
                print(f"WARNING: {name} shares only {len(ids)}/{len(po)} external images with {rep[m]} "
                      f"- check that everyone used the same manifest")
            t = mcnemar(po.loc[ids].y_true, po.loc[ids].y_pred, pe.loc[ids].y_pred)
            tests.append({"test_set": "external", "model_a": rep[m], "model_b": name,
                          "f1_a": macro_f1(po.loc[ids].y_true, po.loc[ids].y_pred, n_cls),
                          "f1_b": macro_f1(po.loc[ids].y_true, pe.loc[ids].y_pred, n_cls), **t})
        lo, hi = bootstrap_ci(pe.y_true.values, pe.y_pred.values, n_cls, a.n_boot)
        extra_rows.append({"model": name, "n": len(pe), "E2_f1_macro": macro_f1(pe.y_true, pe.y_pred, n_cls),
                           "E2_accuracy": float((pe.y_true == pe.y_pred).mean()), "ci_low": lo, "ci_high": hi})
    if extra_rows:
        pd.DataFrame(extra_rows).to_csv(out / "extra_models.csv", index=False)
    mc = pd.DataFrame(tests)
    if not mc.empty:
        mc["significant_0.05"] = mc.p_value < 0.05
        mc.to_csv(out / "mcnemar.csv", index=False)

    # ---------------------------------------------------------------- random-guess reference on E2
    # mean macro-F1 of 200 uniformly random label assignments: the floor any model should beat
    ext_true = preds(runs, rep[models[0]], "external").y_true.values
    rng = np.random.default_rng(0)
    random_ref = float(np.mean([macro_f1(ext_true, rng.integers(0, n_cls, len(ext_true)), n_cls) for _ in range(200)]))

    # ---------------------------------------------------------------- figure 1: E1 vs E2 per model
    fig, ax = plt.subplots(figsize=(max(7.5, 1.9 * len(models) + 2), 4.4))
    x, w = np.arange(len(models)), 0.36
    for k, (col, colr, lab) in enumerate([("E1_f1_macro", E1_COLOR, "E1 PlantVillage internal test"),
                                          ("E2_f1_macro", E2_COLOR, "E2 PlantDoc external test")]):
        xs = x + (k - 0.5) * w
        ax.bar(xs, mean[col], w * 0.92, color=colr, label=lab, zorder=2)
        ax.errorbar(xs, mean[col], yerr=sd[col].fillna(0), fmt="none", ecolor=INK, capsize=3, lw=1, zorder=3)
        for xi, m in zip(xs, models):
            vals = df[df.model == m][col].values
            ax.scatter(np.full(len(vals), xi), vals, s=14, color="white", edgecolor=INK, lw=0.8, zorder=4)
            s_ = 0 if np.isnan(sd.loc[m, col]) else sd.loc[m, col]
            ax.text(xi, mean.loc[m, col] + s_ + 0.02, f"{mean.loc[m, col]:.2f}", ha="center", fontsize=8, color=INK)
    for yv, txt, ls in [(a.target, f"A1 E2 target {a.target:.2f}", "--"), (random_ref, f"random guess {random_ref:.2f}", ":")]:
        ax.axhline(yv, color=INK_2, ls=ls, lw=1)
        ax.text(len(models) - 0.5, yv + 0.012, txt, ha="right", fontsize=7.5, color=INK_2)
    for i in range(1, len(models)):
        if models[i].split("|")[0] != models[i - 1].split("|")[0]:
            ax.axvline(i - 0.5, color=GRID, lw=1.5)
    ax.set_xticks(x, [model_label(m, newline=True) for m in models], fontsize=8.5)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Macro-F1")
    ax.set_title("Internal vs external macro-F1 (bars = mean, whiskers = SD, dots = seeds)", fontsize=9.5)
    _style(ax); ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    fig.tight_layout(); fig.savefig(out / "fig_internal_vs_external.png", dpi=200); plt.close(fig)

    # ---------------------------------------------------------------- figure 2: per-class F1 heatmap
    mat, colnames = [], []
    for m in models:
        for which, tag in [("internal", "E1"), ("external", "E2")]:
            f1s = [pd.read_csv(runs / run / f"per_class_{which}.csv", index_col=0)["f1"].reindex(labels).values
                   for run in df[df.model == m].run]
            mat.append(np.mean(f1s, axis=0)); colnames.append(f"{model_label(m, newline=True)}\n{tag}")
    mat = np.array(mat).T
    pd.DataFrame(mat, index=labels, columns=[c.replace("\n", " ") for c in colnames]).to_csv(out / "per_class_f1.csv")
    fig, ax = plt.subplots(figsize=(1.2 * len(colnames) + 3, 6.3))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(colnames)), colnames, fontsize=7)
    ax.set_yticks(range(n_cls), [short(l) for l in labels], fontsize=8)
    for i in range(n_cls):
        for j in range(len(colnames)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6.8,
                    color="white" if mat[i, j] > 0.55 else INK)
    for j in range(2, len(colnames), 2):
        ax.axvline(j - 0.5, color="white", lw=3)
    ax.set_title("Per-class F1 (mean over seeds): PlantVillage (E1) vs PlantDoc (E2)", fontsize=9.5)
    fig.colorbar(im, ax=ax, fraction=0.035, label="F1")
    fig.tight_layout(); fig.savefig(out / "fig_per_class_f1.png", dpi=200); plt.close(fig)

    # ---------------------------------------------------------------- figure 3: prediction share on E2
    ncol = min(3, len(models)); nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.6 * nrow), sharey=True, sharex=True, squeeze=False)
    for ax in axes.ravel()[len(models):]:
        ax.axis("off")
    for ax, m in zip(axes.ravel(), models):
        p = preds(runs, rep[m], "external")
        y = np.arange(n_cls)
        ax.barh(y - 0.2, np.bincount(p.y_true, minlength=n_cls) / len(p), 0.38, color=E1_COLOR, label="True class share")
        ax.barh(y + 0.2, np.bincount(p.y_pred, minlength=n_cls) / len(p), 0.38, color=E2_COLOR, label="Predicted class share")
        ax.set_yticks(y, [short(l) for l in labels], fontsize=8); ax.invert_yaxis()
        ax.set_title(f"{model_label(m)} ({rep[m]})", fontsize=9); ax.set_xlabel("Share of PlantDoc images")
        _style(ax); ax.grid(axis="x", color=GRID); ax.grid(axis="y", visible=False)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Are external predictions collapsing onto a few classes?", fontsize=10.5)
    fig.tight_layout(); fig.savefig(out / "fig_external_prediction_share.png", dpi=200); plt.close(fig)

    for m in models:
        tag = m.replace("|", "_")
        for which in ["external", "internal"]:
            src = runs / rep[m] / f"confusion_{which}.png"
            if src.exists():
                shutil.copy(src, out / f"fig_confusion_{which}_{tag}.png")

    # ---------------------------------------------------------------- console summary
    print((out / "model_summary.md").read_text(encoding="utf-8"))
    print(f"\nrandom-guess E2 macro-F1 reference: {random_ref:.3f}")
    if not mc.empty:
        print("\nMcNemar (external):")
        print(mc[mc.test_set == "external"][["model_a", "model_b", "f1_a", "f1_b", "p_value"]].to_string(index=False))
    print(f"\nall outputs -> {out}")


if __name__ == "__main__":
    main()
