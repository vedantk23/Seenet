

import torch
from typing import Tuple


def mine_hard_negatives(
    E_pos: torch.Tensor,
    E_neg: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
 
    if E_pos.shape[0] == 0 or E_neg.shape[0] == 0:
        P = min(E_pos.shape[0], E_neg.shape[0])
        return E_neg[:P], torch.arange(P, device=E_neg.device)

  
    return E_hard, indices


def build_see_batch(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
   
    pos_mask = labels == k
    neg_mask = ~pos_mask

    E_pos = embeddings[pos_mask]   # (P, D)
    E_neg = embeddings[neg_mask]   # (Q, D)

    P = E_pos.shape[0]
    Q = E_neg.shape[0]

    if P == 0:
        # No positive samples in this batch for class k
        return embeddings[:0], labels[:0].float()

    if Q == 0:
        # All same class — only positives available
        y_k = torch.ones(P, device=embeddings.device)
        return E_pos, y_k

    # Mine P hardest negatives from Q negatives
    E_hard, _ = mine_hard_negatives(E_pos, E_neg)   # (P, D)

    # Eq. 8: concatenate → (2P, D)
    Ek  = torch.cat([E_pos, E_hard], dim=0)
    y_k = torch.cat([
        torch.ones(P,  device=embeddings.device),
        torch.zeros(P, device=embeddings.device),
    ])
    return Ek, y_k
