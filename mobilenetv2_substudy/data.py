"""
Data preparation, datasets and transforms for the MobileNetV2 experiment.

prepare_data() builds the same audited, stratified 70/15/15 PlantVillage split and
the same Cropped PlantDoc external set as the group's main pipeline (identical
audit rules and split seed), and writes them as CSV manifests.
"""
import hashlib

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

import mnv2_config as C

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


# ----------------------------------------------------------------------------
# Curation and split
# ----------------------------------------------------------------------------
def _md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _readable(path):
    try:
        with Image.open(path) as im:
            im.convert("RGB").load()
        return True
    except Exception:
        return False


def _collect(folders, roots):
    """(path, label) for every image of the shared classes. Only the top-level
    PlantVillage folders are read, so the nested duplicate copy is ignored."""
    rows = []
    for label, folder in enumerate(folders):
        for root in roots:
            d = root / folder
            if d.is_dir():
                rows += [dict(path=str(p), label=label) for p in sorted(d.iterdir())
                         if p.suffix.lower() in IMG_EXT]
    return pd.DataFrame(rows)


def _audit(df, name):
    """Remove unreadable files, images whose content appears under two labels,
    and exact duplicates (MD5)."""
    n0 = len(df)
    df = df[df["path"].map(_readable)].copy()
    df["md5"] = df["path"].map(_md5)
    n_labels = df.groupby("md5")["label"].nunique()
    df = df[~df["md5"].isin(n_labels[n_labels > 1].index)]
    df = df.drop_duplicates("md5").reset_index(drop=True)
    print(f"{name}: {n0} images -> {len(df)} after audit")
    return df


def prepare_data(force=False):
    """Write the split manifests (skipped when they already exist)."""
    if C.PV_MANIFEST.exists() and C.PD_MANIFEST.exists() and not force:
        return
    pv = _audit(_collect([c[1] for c in C.CLASS_MAP], [C.PV_DIR]), "PlantVillage")
    pd_ext = _audit(_collect([c[2] for c in C.CLASS_MAP], [C.PD_DIR / "train", C.PD_DIR / "test"]),
                    "PlantDoc")
    pd_ext = pd_ext[~pd_ext["md5"].isin(set(pv["md5"]))].reset_index(drop=True)  # no leakage

    tr, rest = train_test_split(pv.index, test_size=0.30, stratify=pv.label, random_state=C.SPLIT_SEED)
    va, te = train_test_split(rest, test_size=0.50, stratify=pv.loc[rest, "label"], random_state=C.SPLIT_SEED)
    pv["split"] = ""
    pv.loc[tr, "split"], pv.loc[va, "split"], pv.loc[te, "split"] = "train", "val", "test"
    pd_ext["split"] = "external"
    pv.to_csv(C.PV_MANIFEST, index=False)
    pd_ext.to_csv(C.PD_MANIFEST, index=False)
    print(pv.split.value_counts().to_dict(), "| external:", len(pd_ext))


# ----------------------------------------------------------------------------
# Transforms: augmentation on the training split only
# ----------------------------------------------------------------------------
def train_transform():
    return T.Compose([
        T.RandomResizedCrop(C.IMG_SIZE, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(20),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        T.ToTensor(),
        T.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])


def eval_transform():
    return T.Compose([
        T.Resize((C.IMG_SIZE, C.IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])


class ManifestDataset(Dataset):
    def __init__(self, df, transform):
        self.paths = df["path"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB")), self.labels[i]


def load_split(name):
    """name in {'train', 'val', 'test', 'external'}."""
    if name == "external":
        return pd.read_csv(C.PD_MANIFEST)
    df = pd.read_csv(C.PV_MANIFEST)
    return df[df.split == name].reset_index(drop=True)


def make_loader(name, train=False):
    ds = ManifestDataset(load_split(name), train_transform() if train else eval_transform())
    workers = C.NUM_WORKERS if train else 0       # evaluation decodes in the main process
    return DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=train, num_workers=workers,
                      pin_memory=True, persistent_workers=workers > 0, drop_last=train)
