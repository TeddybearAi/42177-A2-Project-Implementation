"""
Train the baseline CNN on the PlantVillage training split for one seed.

* AdamW, cosine learning-rate decay, label smoothing, mixed precision.
* After every epoch the model is scored on PlantVillage *validation*; the epoch with
  the best validation macro-F1 is saved as best.pt (PlantDoc is never used here).
* Early stopping after PATIENCE epochs without improvement.
* last.pt is written after every epoch, so an interrupted run resumes where it stopped.
"""
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

import cnn_config as C
from data import make_loader
from model import BaselineCNN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device):
    """Softmax probabilities and true labels for every image in a loader."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(x.to(device, non_blocking=True))
        probs.append(torch.softmax(out.float(), 1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def train(seed, epochs=C.EPOCHS, limit=None):
    """Train one seed. `limit` = batches per epoch (quick tests only)."""
    out = C.RUNS_DIR / f"cnn_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists():
        print(f"cnn_s{seed}: already trained, skipping")
        return

    set_seed(seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=C.LABEL_SMOOTHING)

    start, best_f1, bad, history = 0, -1.0, 0, []
    if (out / "last.pt").exists():                       # resume an interrupted run
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start, best_f1, bad, history = ck["epoch"] + 1, ck["best_f1"], ck["bad"], ck["history"]
        torch.set_rng_state(ck["rng_cpu"].cpu())         # RNG states must be CPU tensors
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(ck["rng_cuda"].cpu())
        np.random.set_state(ck["rng_np"])
        print(f"cnn_s{seed}: resuming at epoch {start}")

    train_loader, val_loader = make_loader("train", train=True), make_loader("val")

    for epoch in range(start, epochs):
        t0 = time.time()
        model.train()
        run_loss, n = 0.0, 0
        for i, (x, y) in enumerate(train_loader):
            if limit and i >= limit:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * len(y); n += len(y)
        sched.step()

        probs, labels = predict(model, val_loader, device)
        acc = float((probs.argmax(1) == labels).mean())
        f1 = f1_score(labels, probs.argmax(1), average="macro")
        history.append(dict(epoch=epoch, train_loss=run_loss / max(n, 1), val_acc=acc,
                            val_macro_f1=f1, seconds=time.time() - t0))
        print(f"cnn_s{seed} ep{epoch:02d}  loss {run_loss / max(n, 1):.4f}  val acc {acc:.4f}  "
              f"val F1 {f1:.4f}  ({time.time() - t0:.0f}s)", flush=True)

        if f1 > best_f1:                                   # keep the best validation epoch
            best_f1, bad = f1, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        scaler=scaler.state_dict(), epoch=epoch, best_f1=best_f1, bad=bad, history=history,
                        rng_cpu=torch.get_rng_state(),
                        rng_cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else torch.empty(0),
                        rng_np=np.random.get_state()), out / "last.pt")
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        if bad >= C.PATIENCE:
            print(f"cnn_s{seed}: early stop at epoch {epoch}")
            break

    (out / "last.pt").unlink(missing_ok=True)
    with open(out / "done.json", "w") as f:
        json.dump(dict(seed=seed, best_val_macro_f1=best_f1, epochs_run=len(history)), f, indent=2)
