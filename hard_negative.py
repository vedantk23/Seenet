"""
hard_negative.py — Hard Negative Mining for SEE Modules
=========================================================
Implements Equations 5–8 from the paper.

Paper Section III-B:
  "We utilize hard negative mining techniques to extract a subset of P samples
   from a total of Q negative samples, which are prone to being misclassified
   as positive samples."

Equations:
  Sk       = E_pos_k · (E_neg_k)^T        [Eq. 5]  (P × Q) similarity matrix
  I_neg_k  = argmax(softmax(Sk))          [Eq. 6]  (P,) hardest negative indices
  E_neg_k  = E_neg_k[I_neg_k]            [Eq. 7]  (P, D) hard negatives
  Ek       = concat([E_pos_k; E_neg_k])  [Eq. 8]  (2P, D) balanced batch for SEEk
"""

import torch
from typing import Tuple


def mine_hard_negatives(
    E_pos: torch.Tensor,
    E_neg: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given positive embeddings (P, D) and negative embeddings (Q, D),
    return the P hardest negatives — those most similar to positives.

    Returns:
        E_hard : (P, D)  — hardest negative embeddings
        indices: (P,)    — their indices into E_neg
    """
    if E_pos.shape[0] == 0 or E_neg.shape[0] == 0:
        P = min(E_pos.shape[0], E_neg.shape[0])
        return E_neg[:P], torch.arange(P, device=E_neg.device)

    # Eq. 5: similarity matrix (P, Q)
    Sk = E_pos @ E_neg.t()

    # Eq. 6: for each positive, index of most similar negative
    indices = Sk.argmax(dim=1)           # (P,)

    # Eq. 7: extract hard negatives
    E_hard  = E_neg[indices]             # (P, D)
    return E_hard, indices


def build_see_batch(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build balanced training batch for SEE_k using hard negative mining.

    Args:
        embeddings : (B, D) — utterance embeddings from soft attention
        labels     : (B,)   — ground-truth class indices
        k          : int    — emotion class this SEE expert handles

    Returns:
        Ek  : (2P, D) — [positives ; hard-negatives]
        y_k : (2P,)   — binary labels (1=positive, 0=negative)

    Paper: "the training batch size B' of SEE_k is 2P = 2B/C"
    """
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
