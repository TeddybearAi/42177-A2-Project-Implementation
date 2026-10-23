"""
One command for the whole baseline CNN experiment: prepare data -> train -> evaluate.

    python run_cnn.py                       # seeds 42, 43, 44 (about 20 min per seed on an RTX 5070)
    python run_cnn.py --seeds 42            # one seed only
    python run_cnn.py --epochs 1 --limit 20 # quick check that everything runs (~2 min)

Finished seeds are skipped and an interrupted seed resumes from its last epoch,
so after a crash or shutdown just run the same command again.
"""
import argparse

import cnn_config as C
from data import prepare_data
from evaluate import evaluate_seed, summarize
from train import train


def main():
    ap = argparse.ArgumentParser(description="Baseline CNN: PlantVillage -> Cropped PlantDoc")
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
