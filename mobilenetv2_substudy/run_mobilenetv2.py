"""
One command for the whole MobileNetV2 experiment: prepare data -> train -> evaluate -> compare.

    python run_mobilenetv2.py                       # seeds 0, 1, 2 (~8 min per seed on an RTX 5070 laptop GPU)
    python run_mobilenetv2.py --seeds 0             # one seed only
    python run_mobilenetv2.py --seeds 9 --epochs 1 --limit 20   # quick check that everything runs

Finished seeds are skipped and an interrupted seed resumes from its last epoch,
so after a crash or shutdown just run the same command again.
"""
import argparse

import mnv2_config as C
from data import prepare_data
from evaluate import evaluate_seed, summarize
from train import train


def main():
    ap = argparse.ArgumentParser(description="MobileNetV2: PlantVillage -> Cropped PlantDoc")
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--limit", type=int, default=None, help="batches per epoch (quick test only)")
    args = ap.parse_args()

    prepare_data()
    for s in args.seeds:
        train(s, epochs=args.epochs, limit=args.limit)
        evaluate_seed(s)
    summarize(args.seeds)
    print(f"\nResults written to {C.RESULTS_DIR}")


if __name__ == "__main__":      # required on Windows for DataLoader worker processes
    main()
