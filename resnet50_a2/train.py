"""
train.py - one training run of an ImageNet-pretrained ResNet50 (main model) or
EfficientNet-B0 (alternative, --arch efficientnet_b0) on PlantVillage,
evaluated on the PlantVillage internal test set (E1) and, without any
retraining, on PlantDoc (E2).
42177 Image Processing and Pattern Recognition, A2, Group 2. Author: Teddy Ryu.

Fine-tuning regimes (--regime), as proposed in A1 Section 3.2.2:
  frozen : backbone frozen, only the new head is trained (linear probe)
  layer4 : head + last residual stage (layer4) are trained
  full   : the whole network is trained

Every regime starts with --warmup-epochs of head-only training so the randomly
initialised head does not damage the pretrained features, then continues with
AdamW, discriminative learning rates (backbone 1e-4, head 1e-3), cosine decay,
label smoothing 0.1 and early stopping on PlantVillage *validation* macro-F1.
PlantDoc is touched exactly once, after the best checkpoint has been fixed.

Example
-------
python train.py --manifest outputs/manifest.csv --pv-root D:/data/PlantVillage \
                --pd-root D:/data/PlantDoc_cropped --regime full --seed 42 --device cuda:0
Outputs go to runs/<regime>_s<seed>/ (ResNet50) or runs/effb0_<regime>_s<seed>/ (EfficientNet-B0).
For EfficientNet-B0, regime 'layer4' means its last two MBConv stages + final conv (features.6-8).
"""
import os
# One BLAS/OpenMP thread per process: with many DataLoader worker processes on a 32-thread CPU,
# per-process thread pools otherwise exhaust memory on Windows ("OpenBLAS error: Memory allocation").
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import (ARCH, ManifestDataset, bootstrap_ci, build_model, count_params, eval_tf, load_labels,
                    macro_f1, plot_curves, resolve_paths, save_eval, set_train_mode, set_trainable,
                    train_tf, trainable_modules)


# --------------------------------------------------------------------------- reproducibility
def seed_everything(seed: int):
    """Seed Python, NumPy and PyTorch (CPU and all GPUs) with the training seed."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def worker_init(worker_id):
    """Give every DataLoader worker its own deterministic random stream."""
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s + worker_id); random.seed(s + worker_id)


# --------------------------------------------------------------------------- one pass over a loader
def run_epoch(model, loader, device, crit, amp_dtype, opt=None, trainable=None, max_batches=0, scaler=None,
              n_cls=12):
    """Train (if opt is given) or evaluate for one epoch. `scaler` is only needed for fp16.
    Returns loss, accuracy, macro-F1, y_true, y_pred, softmax probs, dataset row indices."""
    training = opt is not None
    if training:
        set_train_mode(model, trainable)
    else:
        model.eval()
    loss_sum, ys, ps, probs, idxs = 0.0, [], [], [], []
    with torch.set_grad_enabled(training):
        for b, (x, y, idx) in enumerate(loader):
            if max_batches and b >= max_batches:
                break
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                logits = model(x)
                loss = crit(logits, y)
            if training:
                opt.zero_grad(set_to_none=True)
                if scaler is not None:          # fp16: scale loss to avoid gradient underflow
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                else:                           # bf16 / fp32: plain backward pass
                    loss.backward(); opt.step()
            loss_sum += loss.item() * y.size(0)
            ys.append(y.cpu()); ps.append(logits.argmax(1).cpu())
            probs.append(torch.softmax(logits.float(), 1).detach().cpu()); idxs.append(idx)
    ys, ps = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return (loss_sum / len(ys), float((ys == ps).mean()), macro_f1(ys, ps, n_cls), ys, ps,
            torch.cat(probs).numpy(), torch.cat(idxs).numpy())


@torch.no_grad()
def latency_ms(model, device, size, amp_dtype, reps=50):
    """Average single-image inference time in milliseconds (secondary efficiency metric)."""
    model.eval()
    x = torch.randn(1, 3, size, size, device=device).contiguous(memory_format=torch.channels_last)
    with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
        for _ in range(10):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(reps):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return 1000 * (time.perf_counter() - t0) / reps


# --------------------------------------------------------------------------- main
def run_name(arch, regime, seed):
    """Run folder name; ResNet50 keeps the original names so earlier runs stay valid."""
    return f"{regime}_s{seed}" if arch == "resnet50" else f"effb0_{regime}_s{seed}"


def parse_args(argv=None):
    """Command-line options; the defaults are the settings used for every reported run."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="outputs/manifest.csv")
    ap.add_argument("--pv-root", required=True)
    ap.add_argument("--pd-root", required=True)
    ap.add_argument("--mapping", default=str(Path(__file__).with_name("class_mapping.json")))
    ap.add_argument("--arch", choices=list(ARCH), default="resnet50")
    ap.add_argument("--regime", choices=["frozen", "layer4", "full"], required=True)
    ap.add_argument("--seed", type=int, default=42, help="training seed (data split stays fixed)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="", help="default: runs/[effb0_]<regime>_s<seed>")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20, help="epochs after warm-up (upper bound; early stopping)")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16")
    ap.add_argument("--no-aug", action="store_true", help="disable training augmentation (ablation)")
    ap.add_argument("--no-pretrained", action="store_true", help="random init (smoke tests only)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--max-batches", type=int, default=0, help="limit batches per epoch (smoke tests only)")
    return ap.parse_args(argv)


def main(argv=None):
    """Train one (architecture, regime, seed) configuration and save its evaluation outputs."""
    a = parse_args(argv)
    seed_everything(a.seed)
    device = torch.device(a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    amp_dtype = None if (a.amp == "off" or device.type == "cpu") else \
        (torch.bfloat16 if a.amp == "bf16" else torch.float16)
    scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None

    out = Path(a.out or f"runs/{run_name(a.arch, a.regime, a.seed)}"); out.mkdir(parents=True, exist_ok=True)
    labels = load_labels(a.mapping)
    df = resolve_paths(pd.read_csv(a.manifest), a.pv_root, a.pd_root)
    split = {s: df[df.split == s].reset_index(drop=True) for s in ["train", "val", "test", "external"]}

    # ---- data loaders (Windows-safe: everything runs under the __main__ guard)
    g = torch.Generator(); g.manual_seed(a.seed)
    def loader(s, tf, shuffle):
        return DataLoader(ManifestDataset(split[s], tf), batch_size=a.batch_size, shuffle=shuffle,
                          num_workers=a.workers, pin_memory=device.type == "cuda",
                          persistent_workers=a.workers > 0 and s in ("train", "val"),  # test sets are read once
                          worker_init_fn=worker_init, generator=g,
                          drop_last=False)
    dl_tr = loader("train", train_tf(a.img_size, augment=not a.no_aug), True)
    dl_va, dl_te, dl_ex = (loader(s, eval_tf(a.img_size), False) for s in ["val", "test", "external"])

    # ---- model, loss
    model = build_model(a.arch, len(labels), pretrained=not a.no_pretrained, dropout=a.dropout)
    head_prefix, arch_name = ARCH[a.arch]["head"] + ".", ARCH[a.arch]["name"]
    model = model.to(device).to(memory_format=torch.channels_last)
    crit = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)

    cfg = vars(a) | {"device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                     "torch": torch.__version__, "classes": labels,
                     "n_per_split": {k: len(v) for k, v in split.items()},
                     "init": "random" if a.no_pretrained else f"torchvision {arch_name} ImageNet-1k weights",
                     "selection": "best PlantVillage val macro-F1; PlantDoc never used for training or selection"}
    json.dump(cfg, open(out / "config.json", "w"), indent=2)

    def make_opt(names, n_epochs):
        """AdamW with discriminative LRs (head vs backbone) and cosine decay."""
        set_trainable(model, names)
        head = [p for n, p in model.named_parameters() if n.startswith(head_prefix)]
        body = [p for n, p in model.named_parameters()
                if p.requires_grad and not n.startswith(head_prefix)]
        groups = ([{"params": head, "lr": a.lr_head}]           # frozen: head group only
                  + ([{"params": body, "lr": a.lr_backbone}] if body else []))
        opt = torch.optim.AdamW(groups, weight_decay=a.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_epochs, 1))
        return opt, sched

    # ---- two-stage training: head-only warm-up, then the regime's fine-tuning stage.
    # The checkpoint with the best PlantVillage validation macro-F1 (over both stages) is kept;
    # early stopping applies to the fine-tuning stage only.
    hist, best = [], {"f1": -1.0, "epoch": 0, "state": None}
    ep, t_start = 0, time.time()
    for stage, n_ep in [("warmup", a.warmup_epochs), ("finetune", a.epochs)]:
        names = trainable_modules(a.arch, a.regime, stage)
        opt, sched = make_opt(names, n_ep)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"== {stage}: training {names} ({n_train/1e6:.2f}M params) "
              f"for up to {n_ep} epochs", flush=True)
        stale = 0
        for _ in range(n_ep):
            ep += 1; t0 = time.time()
            tl, ta, tf1, *_ = run_epoch(model, dl_tr, device, crit, amp_dtype, opt, names,
                                        a.max_batches, scaler, len(labels))      # train
            vl, va, vf1, *_ = run_epoch(model, dl_va, device, crit, amp_dtype,
                                        max_batches=a.max_batches, n_cls=len(labels))  # val
            sched.step()
            hist.append(dict(epoch=ep, stage=stage, train_loss=tl, train_acc=ta,
                             train_f1_macro=tf1, val_loss=vl, val_acc=va, val_f1_macro=vf1,
                             secs=time.time() - t0))
            pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
            flag = ""
            # new best on validation: keep a CPU copy of the weights
            if vf1 > best["f1"]:
                best = {"f1": vf1, "epoch": ep,
                        "state": {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}}
                stale, flag = 0, " *best"
            else:
                stale += 1
            print(f"[{a.arch} {a.regime} s{a.seed}] ep{ep:02d} {stage:8s} "
                  f"train loss {tl:.4f} acc {ta:.4f} | val loss {vl:.4f} acc {va:.4f} "
                  f"F1 {vf1:.4f} | {time.time() - t0:.0f}s{flag}", flush=True)
            if stage == "finetune" and stale >= a.patience:
                print(f"early stop: no val macro-F1 gain for {a.patience} epochs"); break
    train_secs = time.time() - t_start

    # ---- fix the best checkpoint, THEN evaluate internal (E1) and external (E2) once
    model.load_state_dict(best["state"])
    torch.save({"state_dict": best["state"], "classes": labels, "config": cfg, "epoch": best["epoch"]},
               out / "best_model.pt")
    plot_curves(hist, out / "training_curves.png", f"{arch_name} ({a.regime}, seed {a.seed})")

    res = {"arch": a.arch, "regime": a.regime, "seed": a.seed, "best_epoch": best["epoch"], "best_val_f1_macro": best["f1"],
           "train_minutes": train_secs / 60}
    n_cls = len(labels)
    for key, dl, s, title in [("internal", dl_te, "test", "PlantVillage internal test (E1)"),
                              ("external", dl_ex, "external", "PlantDoc external test (E2)")]:
        t0 = time.time()
        _, _, _, yt, yp, pr, ix = run_epoch(model, dl, device, crit, amp_dtype, max_batches=a.max_batches, n_cls=len(labels))
        ids = split[s]["image_id"].to_numpy()[ix]
        summ = save_eval(yt, yp, pr, ids, labels, key, out, f"{arch_name} {a.regime} (seed {a.seed}) - {title}")
        summ["f1_macro_ci95"] = bootstrap_ci(yt, yp, n_cls, a.n_boot, seed=a.seed)
        summ["images_per_sec"] = len(yt) / (time.time() - t0)
        res[key] = summ

    i, e = res["internal"]["f1_macro"], res["external"]["f1_macro"]
    res["gap_f1_macro"] = i - e                         # absolute internal-minus-external gap
    res["retention_f1_macro"] = e / i if i else None    # share of internal macro-F1 retained externally
    res["params_million"] = count_params(model) / 1e6
    res["latency_ms_batch1"] = latency_ms(model, device, a.img_size, amp_dtype)
    json.dump(res, open(out / "results.json", "w"), indent=2)

    print(f"\nE1 internal macro-F1 {i:.4f} | E2 external macro-F1 {e:.4f} "
          f"(95% CI {res['external']['f1_macro_ci95'][0]:.3f}-{res['external']['f1_macro_ci95'][1]:.3f}) | "
          f"gap {i - e:.4f} | retention {res['retention_f1_macro']:.3f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
