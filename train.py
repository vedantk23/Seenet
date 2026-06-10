

import os
import gc
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, accuracy_score
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
import numpy as np

import config
from model import SeeNet
from ebm import EBMAugmenter
from dataset import RAVDESSDataset, LabelEncoder, build_loaders, load_audio, get_loso_splits


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Metrics  (Section IV-D)
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true: list, y_pred: list) -> Dict[str, float]:
    """
    UA = Unweighted Accuracy = macro recall (average recall per class)
    WA = Weighted Accuracy   = overall accuracy
    """
    ua = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
    wa = accuracy_score(y_true, y_pred) * 100
    return {"UA": ua, "WA": wa}


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Task Loss  (Eqs. 4, 9, 10)
# ─────────────────────────────────────────────────────────────────────────────
def compute_loss(
    cls_logits:  torch.Tensor,
    labels:      torch.Tensor,
    see_logits:  List[torch.Tensor],
    see_labels:  List[torch.Tensor],
    lam:         float,
    num_classes: int,
) -> Tuple[torch.Tensor, Dict]:
    """
    Eq. 9:  L_total = (1-λ) * L_cls + λ * (1/C) * Σ_k L_k
    Eq. 10: L_cls   = CrossEntropy(cls_logits, labels)
    Eq. 4:  L_k     = BCEWithLogitsLoss(see_logit_k, y_k)
    """
    L_cls = nn.CrossEntropyLoss()(cls_logits, labels)

    L_see    = torch.tensor(0.0, device=cls_logits.device)
    n_active = len(see_logits)
    for logit, lbl in zip(see_logits, see_labels):
        if logit.shape[0] == 0:
            continue
        L_see += nn.BCEWithLogitsLoss()(logit.squeeze(-1), lbl.float())

    if n_active > 0:
        L_see = L_see / num_classes    # (1/C) * Σ L_k  — Eq. 9

    if lam == 0:
        L_total = L_cls
    elif lam == 1:
        L_total = L_see
    else:
        L_total = (1 - lam) * L_cls + lam * L_see

    return L_total, {
        "L_total": L_total.item(),
        "L_cls":   L_cls.item(),
        "L_see":   L_see.item() if n_active > 0 else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Build EBM Corpus
# ─────────────────────────────────────────────────────────────────────────────
def build_corpus(train_samples: List[Dict]) -> List[torch.Tensor]:
    """Pre-load all training waveforms (raw, variable-length) as EBM corpus U."""
    print("  [EBM] Building corpus ...")
    corpus = []
    for s in tqdm(train_samples, desc="  Loading corpus", leave=False):
        try:
            corpus.append(load_audio(s["wav_path"]).cpu())
        except Exception:
            pass
    return corpus


# ─────────────────────────────────────────────────────────────────────────────
# One Training Epoch  — with AMP (FP16)
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(
    model:     SeeNet,
    loader:    DataLoader,
    optimizer: optim.Optimizer,
    scaler:    GradScaler,
    ebm:       Optional[EBMAugmenter],
    device:    torch.device,
) -> Dict:
    model.train()
    total_loss = cls_loss = see_loss = 0.0
    all_preds, all_labels = [], []
    n_batches = 0

    use_amp = (device.type == "cuda")

    for waveforms, labels, _ in tqdm(loader, desc="  Train", leave=False):
        waveforms = waveforms.to(device)
        labels    = labels.to(device)

        # Online EBM augmentation (Algorithm 1)
        if ebm is not None:
            waveforms = ebm(waveforms)

        optimizer.zero_grad()

        # ── AMP forward pass ─────────────────────────────────────────────
        # autocast: runs WavLM in FP16 → ~50% less GPU memory
        with autocast(enabled=use_amp):
            result = model(waveforms, labels)
            loss, comps = compute_loss(
                cls_logits  = result["cls_logits"],
                labels      = labels,
                see_logits  = result["see_logits"],
                see_labels  = result["see_labels"],
                lam         = model.lam,
                num_classes = model.num_classes,
            )

        # ── AMP backward ─────────────────────────────────────────────────
        # scaler prevents FP16 underflow during backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += comps["L_total"]
        cls_loss   += comps["L_cls"]
        see_loss   += comps["L_see"]
        n_batches  += 1

        preds = result["cls_logits"].argmax(-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    return {
        "loss":     total_loss / max(n_batches, 1),
        "L_cls":    cls_loss   / max(n_batches, 1),
        "L_see":    see_loss   / max(n_batches, 1),
        "train_UA": metrics["UA"],
        "train_WA": metrics["WA"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model:  SeeNet,
    loader: DataLoader,
    device: torch.device,
    mode:   str = "joint",
) -> Dict:
    model.eval()
    all_preds, all_labels = [], []
    for waveforms, labels, _ in tqdm(loader, desc="  Eval", leave=False):
        waveforms = waveforms.to(device)
        preds = model.predict(waveforms, mode=mode)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())
    return compute_metrics(all_labels, all_preds)


# ─────────────────────────────────────────────────────────────────────────────
# Single LOSO Fold
# ─────────────────────────────────────────────────────────────────────────────
def train_fold(
    fold_idx:      int,
    train_samples: List[Dict],
    test_samples:  List[Dict],
    encoder:       LabelEncoder,
    device:        torch.device,
    epochs:        int  = config.EPOCHS,
    use_ebm:       bool = True,
) -> Dict:
    os.makedirs(config.CKPT_DIR, exist_ok=True)
    C = encoder.num_classes

    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx+1}  |  Train={len(train_samples)}  Test={len(test_samples)}")
    print(f"{'='*60}")

    # Model
    model = SeeNet(
        num_classes   = C,
        backbone_name = config.BACKBONE,
        hidden        = config.BACKBONE_HIDDEN,
        lam           = config.LAMBDA_RAVDESS,
    ).to(device)

    # AMP GradScaler — prevents FP16 underflow during backward
    scaler = GradScaler(enabled=(device.type == "cuda"))
    print(f"  [AMP] Mixed precision FP16: {'ON  (saves ~50% VRAM)' if device.type == 'cuda' else 'OFF (CPU)'}")

    # Optimizer + Scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.LR,
                           weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # EBM corpus
    ebm = None
    if use_ebm:
        ebm = EBMAugmenter()
        ebm.set_corpus(build_corpus(train_samples))

    # DataLoaders
    train_loader, test_loader = build_loaders(
        train_samples, test_samples, encoder,
        batch_size  = config.BATCH_SIZE,
        num_workers = 2,
    )

    best_ua, best_wa = 0.0, 0.0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer, scaler, ebm, device)
        va = evaluate(model, test_loader, device, mode="joint")
        scheduler.step()

        row = {
            "epoch":   epoch,
            **tr,
            "val_UA":  va["UA"],
            "val_WA":  va["WA"],
            "time":    round(time.time() - t0, 1),
        }
        history.append(row)

        if va["UA"] > best_ua:
            best_ua, best_wa = va["UA"], va["WA"]
            torch.save(model.state_dict(),
                       os.path.join(config.CKPT_DIR, f"fold{fold_idx+1}_best.pt"))

        print(f"  Ep {epoch:2d}/{epochs} | "
              f"Loss={tr['loss']:.4f}  L_cls={tr['L_cls']:.4f}  L_see={tr['L_see']:.4f} | "
              f"ValUA={va['UA']:.1f}%  ValWA={va['WA']:.1f}%  [{row['time']}s]")

    print(f"\n  ✓ Fold {fold_idx+1} → Best UA={best_ua:.2f}%  WA={best_wa:.2f}%")

    # ── Free GPU memory before next fold ──────────────────────────────────
    del model, optimizer, scheduler, scaler, train_loader, test_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {"fold": fold_idx + 1, "best_UA": best_ua, "best_WA": best_wa, "history": history}


# ─────────────────────────────────────────────────────────────────────────────
# Full LOSO Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────
def run_loso(
    data_dir: str                    = config.DATA_DIR,
    epochs:   int                    = config.EPOCHS,
    use_ebm:  bool                   = True,
    device:   Optional[torch.device] = None,
) -> List[Dict]:

    if device is None:
        device = config.get_device()

    print(f"\n{'='*60}")
    print(f"  SeeNet — LOSO Cross-Validation")
    print(f"  Device   : {device}")
    print(f"  Backbone : {config.BACKBONE}")
    print(f"  Epochs   : {epochs}")
    print(f"  EBM      : {use_ebm}")
    print(f"  λ        : {config.LAMBDA_RAVDESS}")
    print(f"{'='*60}\n")

    print("Building LOSO splits:")
    splits, encoder = get_loso_splits(data_dir)

    fold_results = []
    for idx, (train_s, test_s) in enumerate(splits):
        r = train_fold(idx, train_s, test_s, encoder, device, epochs, use_ebm)
        fold_results.append(r)

    # ── Summary ──────────────────────────────────────────────────────────
    uas = [r["best_UA"] for r in fold_results]
    was = [r["best_WA"] for r in fold_results]

    print(f"\n{'='*60}")
    print("  LOSO Results")
    print(f"{'='*60}")
    print(f"  {'Fold':>5}  {'UA%':>7}  {'WA%':>7}")
    print(f"  {'─'*5}  {'─'*7}  {'─'*7}")
    for r in fold_results:
        print(f"  {r['fold']:>5}  {r['best_UA']:>7.2f}  {r['best_WA']:>7.2f}")
    print(f"  {'─'*5}  {'─'*7}  {'─'*7}")
    print(f"  {'Mean':>5}  {np.mean(uas):>7.2f}  {np.mean(was):>7.2f}")
    print(f"  {'Std':>5}  {np.std(uas):>7.2f}  {np.std(was):>7.2f}")
    print(f"\n  Final → UA = {np.mean(uas):.2f} ± {np.std(uas):.2f}%")
    print(f"          WA = {np.mean(was):.2f} ± {np.std(was):.2f}%")
    print(f"{'='*60}\n")

    # Save results
    os.makedirs(config.LOG_DIR, exist_ok=True)
    summary = {
        "config": {
            "backbone": config.BACKBONE, "epochs": epochs,
            "batch_size": config.BATCH_SIZE, "lr": config.LR,
            "lambda": config.LAMBDA_RAVDESS, "ebm": use_ebm,
        },
        "folds":   [{"fold": r["fold"], "UA": r["best_UA"], "WA": r["best_WA"]}
                    for r in fold_results],
        "mean_UA": float(np.mean(uas)), "std_UA": float(np.std(uas)),
        "mean_WA": float(np.mean(was)), "std_WA": float(np.std(was)),
    }
    out = os.path.join(config.LOG_DIR, "loso_results.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Results saved → {out}")

    return fold_results
