"""
paired_bootstrap.py - paired bootstrap confidence intervals for the DIFFERENCE in
macro-F1 between two model configurations evaluated on the same test images.
42177 Image Processing and Pattern Recognition, A2, Group 2. Author: Teddy Ryu.

Why: McNemar's test (aggregate.py) compares paired *correctness*, not macro-F1.
This script answers the macro-F1 question directly. In every bootstrap replicate the
SAME resampled image indices are used for both models (and for all seeds), so the
interval reflects uncertainty from the finite test set while keeping the pairing.

Two estimates per model pair:
  * per seed    : macro-F1(A, seed s) - macro-F1(B, seed s)
  * seed-averaged (headline): mean over shared seeds of macro-F1(A) - mean of macro-F1(B),
    with images resampled jointly for all seeds in each replicate.
Reported: observed difference, 95% percentile interval, and the share of replicates
in which the difference is <= 0 (a one-sided bootstrap proportion, not a formal p-value).

Usage (from the project folder):
  python paired_bootstrap.py                       # external (PlantDoc) test set
  python paired_bootstrap.py --set internal        # PlantVillage internal test set
Output: runs/summary/paired_bootstrap_<set>.csv and a printed table.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# (label A, run prefix A, label B, run prefix B): A - B is reported
PAIRS = [
    ("ResNet50 full", "full", "ResNet50 layer4", "layer4"),
    ("ResNet50 full", "full", "ResNet50 frozen", "frozen"),
    ("ResNet50 layer4", "layer4", "ResNet50 frozen", "frozen"),
    ("ResNet50 full", "full", "EffNet-B0 full", "effb0_full"),
    ("ResNet50 frozen", "frozen", "EffNet-B0 frozen", "effb0_frozen"),
    ("EffNet-B0 full", "effb0_full", "EffNet-B0 frozen", "effb0_frozen"),
]


def macro_f1_batch(y_true, y_pred, idx, n_cls):
    """Macro-F1 for many resamples at once (same definition as scikit-learn with
    labels=0..n_cls-1, average='macro', zero_division=0).
    y_true, y_pred: (n,) int arrays; idx: (B, n) resample indices -> returns (B,)."""
    t, p = y_true[idx], y_pred[idx]                              # (B, n)
    B = idx.shape[0]
    # count, for every replicate and class, true positives, predictions and true images;
    # np.add.at accumulates repeated (replicate, class) pairs correctly
    rows = np.repeat(np.arange(B), idx.shape[1])
    tp = np.zeros((B, n_cls)); pred_cnt = np.zeros((B, n_cls)); true_cnt = np.zeros((B, n_cls))
    hit = (t == p).ravel()
    np.add.at(tp, (rows[hit], t.ravel()[hit]), 1)
    np.add.at(pred_cnt, (rows, p.ravel()), 1)
    np.add.at(true_cnt, (rows, t.ravel()), 1)
    denom = pred_cnt + true_cnt                                   # 2TP+FP+FN
    f1 = np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)
    return f1.mean(axis=1)


def load(runs: Path, run: str, which: str):
    """Predictions of one run sorted by image_id, or None if the run does not exist."""
    f = runs / run / f"predictions_{which}.csv"
    return pd.read_csv(f).set_index("image_id").sort_index() if f.exists() else None


def main():
    """Compute per-seed and seed-averaged paired bootstrap intervals for every pair in PAIRS."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--set", choices=["external", "internal"], default="external")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-classes", type=int, default=12)
    ap.add_argument("--rng", type=int, default=0, help="bootstrap random seed")
    a = ap.parse_args()

    runs = Path(a.runs_dir)
    out = runs / "summary"; out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")]

    # one shared image order and one shared set of resamples for every model and seed
    ref = None
    for _, pa, _, _ in PAIRS:
        for s in seeds:
            ref = load(runs, f"{pa}_s{s}", a.set)
            if ref is not None:
                break
        if ref is not None:
            break
    if ref is None:
        raise SystemExit(f"no predictions_{a.set}.csv found under {runs}")
    ids = ref.index
    y_true = ref.y_true.to_numpy()
    rng = np.random.default_rng(a.rng)
    idx = rng.integers(0, len(ids), size=(a.n_boot, len(ids)))
    full = np.arange(len(ids))[None, :]

    cache = {}

    def scores(run):
        """(observed macro-F1, bootstrap macro-F1 array) for one run, or None if missing."""
        if run not in cache:
            p = load(runs, run, a.set)
            if p is None:
                cache[run] = None
            else:
                if not p.index.equals(ids):
                    raise SystemExit(f"{run}: test images differ from the reference run - check the manifest")
                if not np.array_equal(p.y_true.to_numpy(), y_true):
                    raise SystemExit(f"{run}: true labels differ from the reference run")
                yp = p.y_pred.to_numpy()
                cache[run] = (float(macro_f1_batch(y_true, yp, full, a.n_classes)[0]),
                              macro_f1_batch(y_true, yp, idx, a.n_classes))
        return cache[run]

    rows = []
    for la, pa, lb, pb in PAIRS:
        diffs_obs, diffs_boot, used = [], [], []
        for s in seeds:
            sa, sb = scores(f"{pa}_s{s}"), scores(f"{pb}_s{s}")
            if sa is None or sb is None:
                continue
            d_obs, d_boot = sa[0] - sb[0], sa[1] - sb[1]
            lo, hi = np.quantile(d_boot, [0.025, 0.975])
            rows.append({"set": a.set, "model_a": la, "model_b": lb, "seed": s,
                         "f1_a": sa[0], "f1_b": sb[0], "diff": d_obs,
                         "ci_low": lo, "ci_high": hi,
                         "share_diff_le_0": float((d_boot <= 0).mean())})
            diffs_obs.append(d_obs); diffs_boot.append(d_boot); used.append(s)
        if used:
            # headline estimate: average the per-seed differences replicate by replicate, so the
            # interval keeps the pairing of images across both models and all seeds
            d_obs = float(np.mean(diffs_obs)); d_boot = np.mean(diffs_boot, axis=0)
            lo, hi = np.quantile(d_boot, [0.025, 0.975])
            rows.append({"set": a.set, "model_a": la, "model_b": lb,
                         "seed": "mean of " + "/".join(map(str, used)),
                         "f1_a": np.nan, "f1_b": np.nan, "diff": d_obs,
                         "ci_low": lo, "ci_high": hi,
                         "share_diff_le_0": float((d_boot <= 0).mean())})

    if not rows:
        raise SystemExit("no model pair had predictions for a shared seed")
    df = pd.DataFrame(rows)
    df.to_csv(out / f"paired_bootstrap_{a.set}.csv", index=False)

    head = df[df.seed.astype(str).str.startswith("mean")]
    print(f"Paired bootstrap of macro-F1 differences ({a.set} test set, {len(ids)} images, {a.n_boot} resamples)\n")
    print("| Model A | Model B | Mean diff (A - B) | 95% CI | Share of resamples with diff <= 0 |")
    print("|---|---|---|---|---|")
    for r in head.itertuples():
        print(f"| {r.model_a} | {r.model_b} | {r.diff:+.3f} | [{r.ci_low:+.3f}, {r.ci_high:+.3f}] | {r.share_diff_le_0:.3f} |")
    print("\nPer seed:")
    print(df[~df.seed.astype(str).str.startswith("mean")]
          [["model_a", "model_b", "seed", "f1_a", "f1_b", "diff", "ci_low", "ci_high"]]
          .round(4).to_string(index=False))
    print(f"\nsaved -> {out / f'paired_bootstrap_{a.set}.csv'}")


if __name__ == "__main__":
    main()
