

import argparse
import os
import random
import numpy as np
import torch
import config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="SeeNet Speech Emotion Recognition")
    p.add_argument("--data-dir",   type=str,   default=config.DATA_DIR)
    p.add_argument("--epochs",     type=int,   default=config.EPOCHS)
    p.add_argument("--batch-size", type=int,   default=config.BATCH_SIZE)
    p.add_argument("--lr",         type=float, default=config.LR)
    p.add_argument("--lam",        type=float, default=config.LAMBDA_RAVDESS)
    p.add_argument("--no-ebm",     action="store_true")
    p.add_argument("--device",     type=str,   default=None,
                   choices=["cpu", "cuda", "mps"])
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--download-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Apply CLI overrides to config
    config.DATA_DIR         = args.data_dir
    config.EPOCHS           = args.epochs
    config.BATCH_SIZE       = args.batch_size
    config.LR               = args.lr
    config.LAMBDA_RAVDESS   = args.lam

    # ── Banner ────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  SeeNet — Soft Emotion Expert Network")
    print("  Paper: IEEE TAFFC 2025  |  Li et al.")
    print("═" * 60)
    print(f"  Backbone   : {config.BACKBONE}")
    print(f"  Input dur  : {config.INPUT_SEC}s ({config.MAX_LEN} samples)")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print(f"  Lambda (λ) : {args.lam}")
    print(f"  EBM        : {'OFF' if args.no_ebm else 'ON'}")
    print(f"  Seed       : {args.seed}")

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = config.get_device()
    print(f"  Device     : {device}")
    print("═" * 60 + "\n")

    # ── Step 1: Data ──────────────────────────────────────────────────────
    print("─" * 60)
    print("  STEP 1 — Data")
    print("─" * 60)
    from download_data import download_ravdess, scan_ravdess
    download_ravdess(args.data_dir)

    if args.download_only:
        samples = scan_ravdess(args.data_dir)
        print(f"\n✓ {len(samples)} samples ready. Exiting (--download-only).")
        return

    # ── Step 2: Train ─────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  STEP 2 — LOSO Cross-Validation Training")
    print("─" * 60)
    from train import run_loso
    run_loso(
        data_dir = args.data_dir,
        epochs   = args.epochs,
        use_ebm  = not args.no_ebm,
        device   = device,
    )


if __name__ == "__main__":
    main()
