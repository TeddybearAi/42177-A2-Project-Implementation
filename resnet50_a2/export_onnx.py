"""
export_onnx.py - export trained checkpoints to ONNX so the evaluation can be
reproduced in MATLAB (matlab/evaluate_in_matlab.m).
42177 Image Processing and Pattern Recognition, A2, Group 2. Author: Teddy Ryu.

The ImageNet normalisation is built into the exported model, so MATLAB only has
to supply RGB images scaled to [0, 1], resized and centre-cropped to 224 x 224.

Each export is checked: the ONNX model (run with onnxruntime) must give the same
logits as the PyTorch model on the same input.

Usage (from the project folder):
  python export_onnx.py                              # representative fully fine-tuned runs
  python export_onnx.py --runs full_s44 effb0_full_s44 frozen_s43
Output: matlab/<run>.onnx
Needs: pip install onnx onnxruntime
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common import MEAN, STD, build_model


class NormalisedModel(nn.Module):
    """Wraps a classifier so it takes [0, 1] RGB input and applies the ImageNet
    normalisation itself (MATLAB then only has to resize, crop and scale the images)."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def export(run_dir: Path, out_dir: Path, opset: int):
    """Export runs/<run>/best_model.pt to <out_dir>/<run>.onnx and check it against PyTorch."""
    ck = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    arch = ck["config"].get("arch", "resnet50")          # runs made before --arch existed are ResNet50
    net = build_model(arch, len(ck["classes"]), pretrained=False)
    net.load_state_dict(ck["state_dict"])
    model = NormalisedModel(net).eval()                  # eval: dropout off, BatchNorm uses running stats

    out = out_dir / f"{run_dir.name}.onnx"
    dummy = torch.rand(1, 3, 224, 224)                   # example input that fixes the graph's shapes
    # legacy TorchScript exporter (dynamo=False): its opset-13 graphs import cleanly into MATLAB
    torch.onnx.export(model, dummy, str(out), input_names=["image"], output_names=["logits"],
                      opset_version=opset, dynamo=False, do_constant_folding=True)

    # ---- check: ONNX output must match PyTorch on the same random [0, 1] batch
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        ref = model(x).numpy()
    got = sess.run(None, {"image": x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    same = int(ref.argmax(1)[0] == got.argmax(1)[0])
    print(f"{run_dir.name}: {arch} -> {out}  (max |logit diff| {diff:.2e}, same prediction {bool(same)})")
    if diff > 1e-3:
        raise SystemExit(f"ONNX output differs from PyTorch for {run_dir.name}")


def main():
    """Export each run named in --runs."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--runs", nargs="+", default=["full_s44", "effb0_full_s44"])
    ap.add_argument("--out", default="matlab")
    ap.add_argument("--opset", type=int, default=13, help="ONNX opset; 13 is widely supported by MATLAB's importer")
    a = ap.parse_args()
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    for r in a.runs:
        export(Path(a.runs_dir) / r, out_dir, a.opset)


if __name__ == "__main__":
    main()
