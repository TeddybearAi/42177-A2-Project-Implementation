"""Cross-model paired comparisons on the shared external set, reusing the group's
paired_bootstrap.py unchanged (2,000 resamples, rng 0, percentile CI, unstratified,
same image indices for both models and all seeds) plus an exact McNemar test per seed.
Usage: python cross_model.py <path to resnet50_a2> <runs folder>
The runs folder holds <prefix>_s<seed>/predictions_external.csv for prefixes
full, effb0_full, yolo, cnn and vit (ViT R1/R4/R5 stored as vit_s42/43/44)."""
import sys, pathlib
repo, runs = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
import numpy as np, pandas as pd
from scipy.stats import binomtest
import paired_bootstrap as pb

PAIRS = [
    ("ResNet50 full", "full", "EffNet-B0 full", "effb0_full"),   # cross-check vs existing Table 2
    ("ResNet50 full", "full", "YOLO11s-cls", "yolo"),
    ("ResNet50 full", "full", "Baseline CNN", "cnn"),
    ("ResNet50 full", "full", "ViT-S/16 attention", "vit"),
    ("EffNet-B0 full", "effb0_full", "YOLO11s-cls", "yolo"),
    ("EffNet-B0 full", "effb0_full", "Baseline CNN", "cnn"),
    ("EffNet-B0 full", "effb0_full", "ViT-S/16 attention", "vit"),
    ("YOLO11s-cls", "yolo", "Baseline CNN", "cnn"),
    ("YOLO11s-cls", "yolo", "ViT-S/16 attention", "vit"),
    ("Baseline CNN", "cnn", "ViT-S/16 attention", "vit"),
]
pb.PAIRS = PAIRS
sys.argv = ["paired_bootstrap.py", "--runs-dir", runs, "--set", "external"]
pb.main()

# exact McNemar per matched seed (paired correctness), aligned by image_id
def load(run):
    p = pd.read_csv(pathlib.Path(runs)/run/"predictions_external.csv").set_index("image_id").sort_index()
    return p
rows = []
for la, pa, lb, pbx in PAIRS:
    for s in (42, 43, 44):
        A, B = load(f"{pa}_s{s}"), load(f"{pbx}_s{s}")
        assert A.index.equals(B.index) and (A.y_true == B.y_true).all()
        ok_a, ok_b = A.y_pred == A.y_true, B.y_pred == B.y_true
        b, c = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
        p = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
        rows.append(dict(model_a=la, model_b=lb, seed=s, n=len(A), acc_a=ok_a.mean(), acc_b=ok_b.mean(),
                         a_right_b_wrong=b, a_wrong_b_right=c, p_exact=p))
m = pd.DataFrame(rows)
m.to_csv(pathlib.Path(runs)/"summary"/"mcnemar_cross_model.csv", index=False)
pd.set_option("display.width", 200)
print(m.round(4).to_string(index=False))
