"""
dataset.py — RAVDESS Dataset + LOSO Cross-Validation
======================================================
Implements:
  - Audio loading (16 kHz mono) using soundfile (reliable on Kaggle/Linux)
  - Pad / center-trim to 6 seconds  (paper Section IV-B)
  - 8-class label encoder for RAVDESS
  - Leave-One-Session-Out (LOSO) cross-validation splits
    → 6 sessions × 4 actors each  (paper Section IV-A)
  - PyTorch DataLoaders with collate function
"""

import os
import torch
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional

import config
from download_data import scan_ravdess


# ─────────────────────────────────────────────────────────────────────────────
# Label Encoder
# ─────────────────────────────────────────────────────────────────────────────
class LabelEncoder:
    """Maps RAVDESS emotion codes ('01'-'08') to integer class indices 0-7."""

    def __init__(self):
        codes = sorted(config.RAVDESS_EMOTIONS.keys())          # ['01',…,'08']
        self.code2idx  = {c: i for i, c in enumerate(codes)}
        self.idx2name  = {i: config.RAVDESS_EMOTIONS[c] for i, c in enumerate(codes)}
        self.num_classes = len(codes)

    def encode(self, code: str) -> Optional[int]:
        return self.code2idx.get(code, None)

    def decode(self, idx: int) -> str:
        return self.idx2name.get(idx, "unknown")


# ─────────────────────────────────────────────────────────────────────────────
# Audio Utilities
# ─────────────────────────────────────────────────────────────────────────────
def load_audio(wav_path: str, target_sr: int = config.SAMPLE_RATE) -> torch.Tensor:
    """
    Load WAV → mono float32 tensor, resample to target_sr if needed.
    Uses soundfile (works on Kaggle Linux without extra codecs).
    Returns 1-D tensor of shape (num_samples,).
    """
    data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    # data: (num_samples, channels) → mono
    waveform = torch.from_numpy(data.mean(axis=1))   # (num_samples,)

    # Resample if sample rate differs
    if sr != target_sr:
        import torchaudio
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform  = resampler(waveform.unsqueeze(0)).squeeze(0)

    return waveform


def pad_or_trim(waveform: torch.Tensor, max_len: int = config.MAX_LEN) -> torch.Tensor:
    """
    Center-trim if longer than max_len, zero-pad on right if shorter.
    Paper Section IV-B: input duration = 6 s (96,000 samples at 16 kHz).
    """
    n = waveform.shape[0]
    if n >= max_len:
        start = (n - max_len) // 2
        return waveform[start: start + max_len]
    return torch.cat([waveform, torch.zeros(max_len - n)])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class RAVDESSDataset(Dataset):
    """
    Each __getitem__ returns (waveform, label, wav_path).
      waveform : FloatTensor (MAX_LEN,)  — fixed 6-second clip
      label    : int
      wav_path : str
    """

    def __init__(self, samples: List[Dict], encoder: LabelEncoder):
        self.encoder = encoder
        # Keep only samples with a valid label in our mapping
        self.samples = [s for s in samples if encoder.encode(s["emotion_code"]) is not None]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, str]:
        s        = self.samples[idx]
        waveform = pad_or_trim(load_audio(s["wav_path"]))
        label    = self.encoder.encode(s["emotion_code"])
        return waveform, label, s["wav_path"]


# ─────────────────────────────────────────────────────────────────────────────
# LOSO Cross-Validation Splits
# ─────────────────────────────────────────────────────────────────────────────
def get_loso_splits(data_dir: str = config.DATA_DIR):
    """
    Returns:
      splits  : list of (train_samples, test_samples) for each of 6 folds
      encoder : LabelEncoder

    Paper Section IV-B:
      "We evaluate our method by conducting leave-one-session-out (LOSO)
       cross-validation … with five or six folds."
    RAVDESS → 6 sessions.
    """
    all_samples = scan_ravdess(data_dir)
    encoder     = LabelEncoder()
    valid       = [s for s in all_samples if encoder.encode(s["emotion_code"]) is not None]

    splits = []
    for held_out in range(1, config.NUM_SESSIONS + 1):
        train = [s for s in valid if s["session_id"] != held_out]
        test  = [s for s in valid if s["session_id"] == held_out]
        splits.append((train, test))
        print(f"  Session {held_out}: train={len(train)}, test={len(test)}")

    return splits, encoder


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader helpers
# ─────────────────────────────────────────────────────────────────────────────
def _collate(batch):
    waveforms, labels, paths = zip(*batch)
    return torch.stack(waveforms), torch.tensor(labels, dtype=torch.long), list(paths)


def build_loaders(
    train_samples, test_samples, encoder,
    batch_size=config.BATCH_SIZE, num_workers=2
) -> Tuple[DataLoader, DataLoader]:
    train_ds = RAVDESSDataset(train_samples, encoder)
    test_ds  = RAVDESSDataset(test_samples,  encoder)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=_collate, num_workers=num_workers,
                              drop_last=True, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              collate_fn=_collate, num_workers=num_workers,
                              pin_memory=True)
    return train_loader, test_loader
