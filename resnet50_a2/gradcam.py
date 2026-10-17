"""
gradcam.py - Grad-CAM heat maps (Selvaraju et al. 2017) for a trained ResNet50 or EfficientNet-B0
on the FIXED image sample written by prepare_data.py (outputs/gradcam_sample.csv),
so that every model in the group is explained on exactly the same images.
42177 Image Processing and Pattern Recognition, A2, Group 2. Author: Teddy Ryu.

The heat map shows which regions of the image drove the prediction. If the
model attends to the background or leaf edge rather than the lesions on
PlantDoc, that is evidence of shortcut learning (A1 Section 5.4).

Example
-------
python gradcam.py --run runs/full_s42 --pv-root D:/data/PlantVillage --pd-root D:/data/PlantDoc_cropped
python gradcam.py --run runs/full_s42 ... --set internal     # PlantVillage test images for contrast
Output: <run>/gradcam_<set>.png and <run>/gradcam_<set>.csv
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from common import ARCH, INK, build_model, eval_tf, resolve_paths, short


class GradCAM:
    """Minimal Grad-CAM on a chosen convolutional block (ResNet50 layer4, EfficientNet-B0 features.8)."""

    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(self._save_act)

    def _save_act(self, module, inp, out):
        """Forward hook: keep the block's feature maps and register a hook that keeps their gradient."""
        self.acts = out
        out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x, target=None):
        """Returns (cam in [0,1] at input resolution, predicted class, softmax probs)."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        probs = torch.softmax(logits, 1)[0]
        pred = int(logits.argmax(1))
        cls = pred if target is None else target
        logits[0, cls].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)          # global-average-pooled gradients
        cam = F.relu((weights * self.acts).sum(1, keepdim=True))       # weighted sum of feature maps
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy(), pred, probs.detach().cpu().numpy()


def main():
    """Explain one trained run on the fixed sample and save the heat-map grid and a CSV."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run folder containing best_model.pt")
    ap.add_argument("--manifest", default="outputs/manifest.csv")
    ap.add_argument("--sample", default="outputs/gradcam_sample.csv")
    ap.add_argument("--pv-root", required=True)
    ap.add_argument("--pd-root", required=True)
    ap.add_argument("--set", choices=["external", "internal"], default="external")
    ap.add_argument("--per-class", type=int, default=2, help="internal set only: images per class")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    device = torch.device(a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu")
    run = Path(a.run)
    ck = torch.load(run / "best_model.pt", map_location="cpu", weights_only=False)
    labels = ck["classes"]
    arch = ck["config"].get("arch", "resnet50")          # older checkpoints are ResNet50
    model = build_model(arch, len(labels), pretrained=False)
    model.load_state_dict(ck["state_dict"]); model.to(device).eval()
    cam = GradCAM(model, model.get_submodule(ARCH[arch]["cam"]))   # last convolutional block

    man = resolve_paths(pd.read_csv(a.manifest), a.pv_root, a.pd_root).set_index("image_id")
    if a.set == "external":
        ids = pd.read_csv(a.sample).image_id.tolist()
    else:   # fixed internal sample for contrast (seed 42)
        te = man[man.split == "test"].reset_index()
        ids = pd.concat([g.sample(min(len(g), a.per_class), random_state=42)
                         for _, g in te.groupby("y")]).image_id.tolist()

    tf = eval_tf(224)
    rows, panels = [], []
    for iid in ids:
        r = man.loc[iid]
        img = Image.open(r.path).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        heat, pred, probs = cam(x)
        shown = np.asarray(tf.transforms[1](tf.transforms[0](img))) / 255.0     # resized + cropped view
        panels.append((shown, heat, int(r.y), pred, float(probs[pred])))
        rows.append({"image_id": iid, "y_true": int(r.y), "y_pred": pred, "label_true": labels[int(r.y)],
                     "label_pred": labels[pred], "confidence": float(probs[pred]), "correct": pred == int(r.y)})
    pd.DataFrame(rows).to_csv(run / f"gradcam_{a.set}.csv", index=False)

    # grid: each panel = image with heat-map overlay; title gives true/pred and correctness
    ncol = 6
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.65 * nrow))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (im, heat, yt, yp, conf) in zip(np.atleast_1d(axes).ravel(), panels):
        ax.imshow(im); ax.imshow(heat, cmap="jet", alpha=0.42)
        ok = yt == yp
        ax.set_title(f"T: {short(labels[yt])}\nP: {short(labels[yp])} ({conf:.2f})", fontsize=6.5,
                     color="#008300" if ok else "#c0392b")
    acc = np.mean([p[2] == p[3] for p in panels])
    src = "PlantDoc (external)" if a.set == "external" else "PlantVillage internal test"
    fig.suptitle(f"Grad-CAM, {ARCH[arch]['name']} {run.name} - {src}; green = correct, red = wrong "
                 f"({acc:.0%} correct on this sample)", fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(run / f"gradcam_{a.set}.png", dpi=200); plt.close(fig)
    print(f"saved -> {run / f'gradcam_{a.set}.png'}")


if __name__ == "__main__":
    main()
