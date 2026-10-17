"""
prepare_data.py - build the shared, portable image manifest for the A2 experiment.
42177 Image Processing and Pattern Recognition, A2, Group 2. Author: Teddy Ryu.

What it does
------------
1. Audits every class folder in PlantVillage and (Cropped) PlantDoc.
2. Keeps only the 12 shared classes defined in class_mapping.json.
3. Drops unreadable images and exact duplicate files inside PlantVillage
   (duplicates could otherwise land in both train and test = leakage).
4. Splits PlantVillage 70 / 15 / 15 (stratified, seed 42) into train / val / test.
5. Marks every PlantDoc image as split = "external" (never used for training,
   early stopping, model selection or hyper-parameter tuning).
6. Writes a fixed Grad-CAM sample (2 external images per class, seed 42) so
   every group model is explained on exactly the same images.

The manifest stores paths RELATIVE to each dataset root plus a stable
`image_id`, so the same file can be shared with teammates on other machines:
each person only passes their own --pv-root / --pd-root at training time.

Usage (Windows PowerShell example)
----------------------------------
python prepare_data.py --pv-root "D:/data/PlantVillage" \
                       --pd-root "D:/data/PlantDoc_cropped" --out outputs
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- helpers
def list_images(folder: Path):
    """All image files directly inside `folder`, sorted for reproducibility."""
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT)


def is_readable(p: Path) -> bool:
    """True if PIL can fully decode the image (PlantDoc is web-scraped; some files are broken)."""
    try:
        with Image.open(p) as im:
            im.convert("RGB").load()
        return True
    except Exception:
        return False


def md5(p: Path) -> str:
    """Content hash used to detect exact duplicate files."""
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm(s: str) -> str:
    """Lower-case and strip punctuation so 'Pepper__bell___Bacterial_spot' (Kaggle)
    and 'Pepper,_bell___Bacterial_spot' (GitHub) compare equal."""
    return "".join(ch for ch in s.lower() if ch.isalnum())


def resolve(root: Path, name: str) -> Path:
    """Find the class folder `name` under `root`, tolerant of naming differences."""
    if (root / name).is_dir():
        return root / name
    hits = [d for d in root.iterdir() if d.is_dir() and _norm(d.name) == _norm(name)]
    if len(hits) != 1:
        available = ", ".join(sorted(d.name for d in root.iterdir() if d.is_dir()))
        raise FileNotFoundError(
            f"Class folder '{name}' not found (or ambiguous) under {root}.\n"
            f"Available folders: {available}\n"
            f"-> edit the matching entry in class_mapping.json.")
    return hits[0]


def audit(root: Path):
    """Image count for every class folder under root (before filtering)."""
    return {d.name: len(list_images(d)) for d in sorted(root.iterdir()) if d.is_dir()}


# --------------------------------------------------------------------------- main
def main():
    """Write manifest.csv, split_counts.csv, gradcam_sample.csv, dataset_audit.json and data_checks.json."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pv-root", required=True, help="folder that directly contains the PlantVillage class folders")
    ap.add_argument("--pd-root", required=True, help="(Cropped) PlantDoc root that contains train/ and test/")
    ap.add_argument("--pd-splits", default="train,test",
                    help="PlantDoc folders used as the external test set (group decision: 'train,test' or 'train')")
    ap.add_argument("--mapping", default=str(Path(__file__).with_name("class_mapping.json")))
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42, help="seed for the data split (fixed for the whole group)")
    ap.add_argument("--gradcam-per-class", type=int, default=2)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pv_root, pd_root = Path(a.pv_root), Path(a.pd_root)
    pd_splits = [s.strip() for s in a.pd_splits.split(",") if s.strip()]
    mapping = json.load(open(a.mapping, encoding="utf-8"))["classes"]
    labels = [m["label"] for m in mapping]

    # ---- 1. audit all folders (documents what was available before filtering)
    aud = {"plantvillage": audit(pv_root)}
    for s in pd_splits:
        aud[f"plantdoc_{s}"] = audit(pd_root / s)
    json.dump(aud, open(out / "dataset_audit.json", "w"), indent=2)

    # ---- 2. collect images of the 12 shared classes
    rows = []
    for y, m in enumerate(mapping):
        pv_dir = resolve(pv_root, m["plantvillage"])
        for p in list_images(pv_dir):
            rel = p.relative_to(pv_root).as_posix()
            rows.append(dict(image_id=f"pv/{rel}", root="pv", relpath=rel, label=m["label"], y=y,
                             source="plantvillage"))
        for s in pd_splits:
            try:
                pd_dir = resolve(pd_root / s, m["plantdoc"])
            except FileNotFoundError:
                print(f"  note: PlantDoc/{s} has no folder for '{m['label']}' - skipped")
                continue
            for p in list_images(pd_dir):
                rel = p.relative_to(pd_root).as_posix()
                rows.append(dict(image_id=f"pd/{rel}", root="pd", relpath=rel, label=m["label"], y=y,
                                 source="plantdoc"))
    df = pd.DataFrame(rows)
    roots = {"pv": pv_root, "pd": pd_root}
    abspath = df.apply(lambda r: roots[r.root] / r.relpath, axis=1)

    # ---- 3a. drop unreadable images
    ok = abspath.map(is_readable)
    unreadable = df.loc[~ok, "image_id"].tolist()
    df, abspath = df[ok].reset_index(drop=True), abspath[ok].reset_index(drop=True)

    # ---- 3b. exact duplicates: within PlantVillage keep the first copy only;
    #          also report PlantVillage<->PlantDoc overlaps (would be leakage).
    print("hashing images for duplicate check ...")
    df["md5"] = abspath.map(md5)
    pv_mask = df.source == "plantvillage"
    pv_only = df[pv_mask]
    dup_pv = pv_only[pv_only.duplicated("md5", keep="first")]
    cross = set(df.loc[pv_mask, "md5"]) & set(df.loc[~pv_mask, "md5"])
    df = df.drop(index=dup_pv.index).reset_index(drop=True)

    # ---- 4. stratified PlantVillage split; 5. PlantDoc = external only
    # Two steps: first hold out val + test (30%), then halve that hold-out into val (15%) and
    # test (15%). Stratifying on y keeps every class's share the same in all three splits.
    pv = df[df.source == "plantvillage"]
    tr, tmp = train_test_split(pv, test_size=a.val_frac + a.test_frac, stratify=pv.y,
                               random_state=a.seed)
    va, te = train_test_split(tmp, test_size=a.test_frac / (a.val_frac + a.test_frac),
                              stratify=tmp.y, random_state=a.seed)
    df["split"] = "external"
    df.loc[tr.index, "split"] = "train"
    df.loc[va.index, "split"] = "val"
    df.loc[te.index, "split"] = "test"
    df.to_csv(out / "manifest.csv", index=False)

    counts = (df.pivot_table(index="label", columns="split", values="image_id", aggfunc="count", fill_value=0)
                .reindex(index=labels, columns=["train", "val", "test", "external"], fill_value=0))
    counts.loc["TOTAL"] = counts.sum()
    counts.to_csv(out / "split_counts.csv")

    # ---- 6. fixed Grad-CAM sample, shared by all models
    ext = df[df.split == "external"]
    sample = pd.concat([g.sample(min(len(g), a.gradcam_per_class), random_state=a.seed)
                        for _, g in ext.groupby("y")])
    sample[["image_id", "label", "y"]].to_csv(out / "gradcam_sample.csv", index=False)

    report = {"unreadable_dropped": unreadable,
              "plantvillage_exact_duplicates_dropped": dup_pv["image_id"].tolist(),
              "plantvillage_plantdoc_identical_files": len(cross),
              "pd_splits_used": pd_splits, "split_seed": a.seed}
    json.dump(report, open(out / "data_checks.json", "w"), indent=2)

    print(counts.to_string())
    print(f"\nunreadable dropped: {len(unreadable)} | PV duplicates dropped: {len(dup_pv)} | "
          f"identical PV<->PlantDoc files: {len(cross)}")
    print(f"manifest -> {out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
