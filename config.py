"""
config.py — SeeNet Hyperparameters
====================================
All settings from the paper (Section IV-B).
Paper: "SeeNet: A Soft Emotion Expert and Data Augmentation Method to Enhance
       Speech Emotion Recognition" (IEEE TAFFC 2025)
"""

import os
import torch


# ─────────────────────────────────────────────
# Device (Kaggle: CUDA T4/P100)
# ─────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# ─────────────────────────────────────────────
# Backbone
# ─────────────────────────────────────────────
BACKBONE          = "microsoft/wavlm-base"   # ~380MB; swap to "microsoft/wavlm-large" for paper-exact
BACKBONE_HIDDEN   = 768                       # WavLM-Base hidden size (Large = 1024)

# ─────────────────────────────────────────────
# Audio  (Section IV-B)
# ─────────────────────────────────────────────
SAMPLE_RATE       = 16_000
INPUT_SEC         = 6                         # 6 seconds — paper's optimal (Section V-A)
MAX_LEN           = SAMPLE_RATE * INPUT_SEC   # 96,000 samples

# ─────────────────────────────────────────────
# Training  (Section IV-B)
# ─────────────────────────────────────────────
LR                = 1e-4                      # Same as paper
WEIGHT_DECAY      = 1e-4
BATCH_SIZE        = 8                         # Safe for T4 with AMP+grad_ckpt; increase to 16/32 if no OOM
EPOCHS            = 5                         # Paper: 30 for RAVDESS; 5 for quick run

# ─────────────────────────────────────────────
# SEE Module  (Section III-B)
# ─────────────────────────────────────────────
SEE_HIDDEN        = 256                       # Paper: 256
SEE_OUT           = 1                         # Binary classifier

# ─────────────────────────────────────────────
# Multi-Task Lambda  (Eq. 9)
# ─────────────────────────────────────────────
LAMBDA_RAVDESS    = 4e-2                      # Paper: 4e-2 for RAVDESS
LAMBDA_IEMOCAP    = 1e-2                      # Paper: 1e-2 for IEMOCAP

# ─────────────────────────────────────────────
# Energy-Based Mixup  (Algorithm 1)
# ─────────────────────────────────────────────
EBM_PROB          = 0.5                       # Probability of mixing per sample
EBM_R_LOW         = -5                        # r ~ U(-5, 5)
EBM_R_HIGH        = 5

# ─────────────────────────────────────────────
# RAVDESS  (Section IV-A)
# ─────────────────────────────────────────────
# 8 emotions, filename position 03 = emotion code
RAVDESS_EMOTIONS = {
    "01": "neutral",   "02": "calm",     "03": "happy",    "04": "sad",
    "05": "angry",     "06": "fearful",  "07": "disgust",  "08": "surprised",
}
NUM_CLASSES     = 8    # Full 8-class RAVDESS
NUM_SESSIONS    = 6    # LOSO: 6 sessions of 4 actors each

# ─────────────────────────────────────────────
# Paths  (edit for your Kaggle environment)
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
# On Kaggle: RAVDESS is typically at /kaggle/input/ravdess-emotional-speech-audio/
# or you can download it. Set DATA_DIR accordingly.
DATA_DIR        = os.path.join(BASE_DIR, "data", "RAVDESS")
CKPT_DIR        = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR         = os.path.join(BASE_DIR, "logs")
