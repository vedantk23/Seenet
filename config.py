import os
import torch



def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")



BACKBONE          = "microsoft/wavlm-base"   # ~380MB; swap to "microsoft/wavlm-large" for paper-exact
BACKBONE_HIDDEN   = 768                       # WavLM-Base hidden size (Large = 1024)


SAMPLE_RATE       = 16_000
INPUT_SEC         = 6                         # 6 seconds — paper's optimal (Section V-A)
MAX_LEN           = SAMPLE_RATE * INPUT_SEC   # 96,000 samples


LR                = 1e-4                      # Same as paper
WEIGHT_DECAY      = 1e-4
BATCH_SIZE        = 8                         # Safe for T4 with AMP+grad_ckpt; increase to 16/32 if no OOM
EPOCHS            = 5                         # Paper: 30 for RAVDESS; 5 for quick run


SEE_HIDDEN        = 256                       # Paper: 256
SEE_OUT           = 1                         # Binary classifier


LAMBDA_RAVDESS    = 4e-2                      # Paper: 4e-2 for RAVDESS
LAMBDA_IEMOCAP    = 1e-2                      # Paper: 1e-2 for IEMOCAP


EBM_PROB          = 0.5                       # Probability of mixing per sample
EBM_R_LOW         = -5                        # r ~ U(-5, 5)
EBM_R_HIGH        = 5


# 8 emotions, filename position 03 = emotion code
RAVDESS_EMOTIONS = {
    "01": "neutral",   "02": "calm",     "03": "happy",    "04": "sad",
    "05": "angry",     "06": "fearful",  "07": "disgust",  "08": "surprised",
}
NUM_CLASSES     = 8    # Full 8-class RAVDESS
NUM_SESSIONS    = 6    # LOSO: 6 sessions of 4 actors each

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))

DATA_DIR        = os.path.join(BASE_DIR, "data", "RAVDESS")
CKPT_DIR        = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR         = os.path.join(BASE_DIR, "logs")
