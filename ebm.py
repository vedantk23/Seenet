"""
ebm.py — Energy-Based Mixup (Algorithm 1)
==========================================
Exact implementation of Algorithm 1 from the paper.

Paper Section III-C:
  "EBM has three distinct features:
   1) The duration of the mix is less than half the duration of the original
      speech — will not affect the primary emotions in the speech.
   2) The mixing scale is affected by the energy ratio between the two speech
      signals.
   3) There is no scaling applied to the original speech signal during the
      mixing process — original speech remains unaffected."

Algorithm 1 (verbatim):
  Input:  Corpus U={ui}^N, mixing probability p, batch D={di}^B, D ⊂ U
  Output: D={di}^B  (augmented in-place on a copy)
  for i = 1 → B:
      u  ~ U(0,1)
      if u < p:
          usec ~ uniform(U)                # random utterance from corpus
          r    ~ U(-5, 5)                 # energy mixing ratio
          L1   = len(di)                  # fixed 96000
          L2   = len(usec)                # variable
          l    ~ U{1, …, L1/2}            # mix length < half of di
          if l < L2: s2 ~ U{0,…,L2-l}   # start in usec
          else:      s2=0; l=L2
          s1   ~ U{0,…,L1-l}             # start in di
          E1   = Σ(di²) / L1
          E2   = Σ(usec²) / L2
          α    = sqrt(E1 / (10^(r/10) * E2))
          di[s1:s1+l] += α * usec[s2:s2+l]   # original di unscaled
  return D
"""

import torch
import numpy as np
from typing import List
import config


def _energy(x: torch.Tensor) -> float:
    """Mean signal energy: E = Σ(x²) / L  (Algorithm 1, lines 15-16)."""
    return float((x * x).sum() / x.shape[0]) + 1e-8


def energy_based_mixup(
    batch: torch.Tensor,
    corpus: List[torch.Tensor],
    p: float = config.EBM_PROB,
) -> torch.Tensor:
    """
    Apply Energy-Based Mixup to a batch of fixed-length waveforms.

    Args:
        batch  : (B, L) — fixed-length padded waveforms (L = MAX_LEN)
        corpus : list of variable-length raw waveforms from the training set (U)
        p      : mixing probability (default 0.5)

    Returns:
        Augmented batch (B, L) — same shape, copy of input.
    """
    if not corpus:
        return batch

    B, L   = batch.shape
    out    = batch.clone()
    N      = len(corpus)

    for i in range(B):
        # Line 2: u ~ U(0,1)
        if np.random.uniform() >= p:
            continue                          # No mix for this sample

        # Line 4: sample usec from corpus
        usec = corpus[np.random.randint(N)].float()
        di   = out[i]                         # (L,) — fixed

        L1 = L
        L2 = usec.shape[0]

        # Line 5: r ~ U(-5, 5)
        r = np.random.uniform(config.EBM_R_LOW, config.EBM_R_HIGH)

        # Line 8: l ~ U{1, …, L1/2}   (mix < half duration)
        l = np.random.randint(1, max(2, L1 // 2 + 1))

        # Lines 9-13
        if l < L2:
            s2 = np.random.randint(0, L2 - l + 1)
        else:
            s2 = 0
            l  = L2                           # can't mix more than usec length

        l  = min(l, L1)                       # safety clamp

        # Line 14: s1 ~ U{0, …, L1-l}
        s1 = np.random.randint(0, L1 - l + 1)

        # Lines 15-17: energies & mixing scale
        E1    = _energy(di)
        E2    = _energy(usec)
        alpha = np.sqrt(E1 / (10 ** (r / 10) * E2 + 1e-8))

        # Line 18: mix only the selected region; original di is unscaled
        seg = usec[s2: s2 + l].to(di.device)
        out[i, s1: s1 + l] = out[i, s1: s1 + l] + alpha * seg

    return out


class EBMAugmenter:
    """Stateful wrapper — holds the training corpus for one LOSO fold."""

    def __init__(self, p: float = config.EBM_PROB):
        self.p      = p
        self.corpus: List[torch.Tensor] = []

    def set_corpus(self, corpus: List[torch.Tensor]):
        self.corpus = corpus
        print(f"  [EBM] Corpus ready: {len(self.corpus)} utterances")

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        return energy_based_mixup(batch, self.corpus, self.p)
