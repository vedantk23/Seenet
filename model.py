

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Dict, List, Optional

import config
from hard_negative import build_see_batch


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Soft Attention Pooling  (Eqs. 1–2)
# ─────────────────────────────────────────────────────────────────────────────
class SoftAttentionPooling(nn.Module):
   

    def __init__(self, hidden: int = config.BACKBONE_HIDDEN):
        super().__init__()
        self.W = nn.Parameter(torch.empty(1, hidden))
        nn.init.xavier_uniform_(self.W.unsqueeze(0))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states : (B, N, D)
        Returns:
            e             : (B, D)
        """
        # Eq. 1: element-wise Hadamard → tanh → sum over D → softmax over N
        scores  = torch.tanh(self.W * hidden_states).sum(dim=-1)  # (B, N)
        weights = F.softmax(scores, dim=-1)                        # (B, N)
        # Eq. 2: weighted sum over N frames
        e = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)  # (B, D)
        return e


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SEE Module  (Eq. 3)
# ─────────────────────────────────────────────────────────────────────────────
class SEEModule(nn.Module):
    

    def __init__(
        self,
        input_dim: int = config.BACKBONE_HIDDEN,
        hidden_dim: int = config.SEE_HIDDEN,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, E_k: torch.Tensor) -> torch.Tensor:
        return self.mlp(E_k)   # (B', 1)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Full SeeNet
# ─────────────────────────────────────────────────────────────────────────────
class SeeNet(nn.Module):

    def __init__(
        self,
        num_classes:   int   = config.NUM_CLASSES,
        backbone_name: str   = config.BACKBONE,
        hidden:        int   = config.BACKBONE_HIDDEN,
        lam:           float = config.LAMBDA_RAVDESS,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden      = hidden
        self.lam         = lam

        # ── Backbone ─────────────────────────────────────────────────────
        print(f"  [SeeNet] Loading backbone: {backbone_name}")
        self.backbone = AutoModel.from_pretrained(backbone_name)

        # Gradient checkpointing — recomputes activations during backward
        # instead of storing them. Trades ~20% more compute for ~40% less VRAM.
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
            print("  [SeeNet] Gradient checkpointing: ON  (saves ~40% VRAM)")

        # ── Soft Attention Pooling ────────────────────────────────────────
        self.attn_pool = SoftAttentionPooling(hidden)

        # ── Primary task: C-class classifier ─────────────────────────────
        self.cls_head = nn.Linear(hidden, num_classes)

        # ── Auxiliary task: C binary SEE experts ─────────────────────────
        self.see_modules = nn.ModuleList([
            SEEModule(hidden) for _ in range(num_classes)
        ])

        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"  [SeeNet] C={num_classes}, D={hidden}, λ={lam}, params={n_params:.1f}M")

    # ── Utterance embeddings ──────────────────────────────────────────────
    def get_embeddings(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:   waveforms : (B, L)  — raw 16 kHz audio, fixed 6 s
        Returns: e        : (B, D)  — utterance-level embeddings
        """
        out = self.backbone(waveforms)
        hs  = out.last_hidden_state          # (B, T_frames, D)
        e   = self.attn_pool(hs)             # (B, D)
        return e

    # ── Forward (training) ────────────────────────────────────────────────
    def forward(
        self,
        waveforms: torch.Tensor,
        labels:    Optional[torch.Tensor] = None,
    ) -> Dict:
        embeddings = self.get_embeddings(waveforms)    # (B, D)
        cls_logits = self.cls_head(embeddings)          # (B, C)

        see_logits, see_labels = [], []

        if labels is not None and self.lam > 0:
            for k in range(self.num_classes):
                Ek, yk = build_see_batch(embeddings, labels, k)
                if Ek.shape[0] == 0:
                    continue
                see_logits.append(self.see_modules[k](Ek))
                see_labels.append(yk)

        return {
            "embeddings":  embeddings,
            "cls_logits":  cls_logits,
            "see_logits":  see_logits,
            "see_labels":  see_labels,
        }

    # ── Inference ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, waveforms: torch.Tensor, mode: str = "joint") -> torch.Tensor:
        """
        mode="cls"   → Eq. 13  (CLS head only)
        mode="see"   → Eq. 12  (SEE modules only)
        mode="joint" → Eq. 15  (weighted combination)
        """
        embeddings = self.get_embeddings(waveforms)
        cls_logits = self.cls_head(embeddings)

        if mode == "cls" or self.lam == 0:
            return cls_logits.argmax(dim=-1)

        see_logits_list = [self.see_modules[k](embeddings) for k in range(self.num_classes)]
        Y_see = torch.cat(see_logits_list, dim=-1)   # (B, C) — Eq. 11

        if mode == "see" or self.lam == 1:
            return Y_see.argmax(dim=-1)

        Y_joint = (1 - self.lam) * cls_logits + self.lam * Y_see   # Eq. 14
        return Y_joint.argmax(dim=-1)
