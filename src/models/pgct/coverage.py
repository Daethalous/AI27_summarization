"""
Coverage Mechanism for Pointer-Generator Transformer (PGCT)
------------------------------------------------------------
Implements both stepwise (for inference) and parallel (for training)
coverage-aware attention mechanisms.

Supports:
- Masking PAD tokens via src_mask or src_lens
- Stepwise coverage update (seq2seq-style)
- Parallelized attention for transformer-style full-sequence training
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoverageMechanism(nn.Module):
    def __init__(self, hidden_size: int, coverage_weight: float = 1.0):
        """
        Args:
            hidden_size: Dimensionality of decoder hidden state.
            coverage_weight: Weight λ for coverage loss term.
        """
        super().__init__()
        self.coverage_weight = coverage_weight
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    # Stepwise Coverage Attention (used in inference / sequential decoding)
    def compute_coverage_attention(
        self,
        decoder_output: torch.Tensor,  # [B, H]
        encoder_outputs: torch.Tensor,  # [B, S, H]
        coverage_vector: Optional[torch.Tensor] = None,  # [B, S]
        src_lens: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute attention and coverage loss for one decoding step.

        Returns:
            context: [B, H] context vector
            attn_weights: [B, S] attention distribution
            cov_loss_t: scalar tensor coverage loss for this step
        """
        B, S, H = encoder_outputs.size()

        # --- (1) prepare mask ---
        if src_mask is None and src_lens is not None:
            src_mask = torch.arange(S, device=encoder_outputs.device).unsqueeze(0) >= src_lens.unsqueeze(1)
        elif src_mask is None:
            src_mask = torch.zeros(B, S, dtype=torch.bool, device=encoder_outputs.device)

        # --- (2) compute attention scores ---
        # decoder_output: [B, H] → [B, 1, H]
        dec_proj = decoder_output.unsqueeze(1).expand(-1, S, -1)
        energy_input = torch.cat((encoder_outputs, dec_proj), dim=-1)  # [B, S, 2H]
        energy = torch.tanh(self.linear(energy_input))  # [B, S, H]
        attn_scores = self.v(energy).squeeze(-1)  # [B, S]

        # --- (3) integrate coverage bias ---
        if coverage_vector is not None:
            attn_scores = attn_scores - 1e3 * src_mask.float()  # mask first
            attn_scores = attn_scores - 0.5 * coverage_vector  # reduce prob. of reattending
        else:
            attn_scores = attn_scores.masked_fill(src_mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, S]

        # --- (4) context vector ---
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [B, H]

        # --- (5) coverage update and loss ---
        if coverage_vector is None:
            coverage_vector = attn_weights
            cov_loss_t = torch.zeros(1, device=attn_weights.device)
        else:
            cov_loss_t = torch.sum(torch.min(attn_weights, coverage_vector), dim=1)  # [B]
            coverage_vector = coverage_vector + attn_weights  # [B, S]
        cov_loss_t = (self.coverage_weight * cov_loss_t.mean()).unsqueeze(0)

        return context, attn_weights, cov_loss_t, coverage_vector

    # Parallel Coverage Attention (used in training for transformer)
    def compute_all_attention(
        self,
        decoder_outputs: torch.Tensor,  # [B, T, H]
        encoder_outputs: torch.Tensor,  # [B, S, H]
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute attention for all decoder steps in parallel (training only).

        Returns:
            attn_weights: [B, T, S]
        """
        B, T, H = decoder_outputs.size()
        S = encoder_outputs.size(1)

        # (1) attention scores
        attn_scores = torch.bmm(decoder_outputs, encoder_outputs.transpose(1, 2))  # [B, T, S]

        # (2) apply mask
        if src_mask is not None:
            attn_scores = attn_scores.masked_fill(src_mask.unsqueeze(1), float('-inf'))

        # (3) normalize
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, T, S]
        return attn_weights

    # Coverage Loss for Full Sequence (vectorized)
    def compute_parallel_coverage_loss(
        self,
        attn_weights: torch.Tensor,  # [B, T, S]
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute coverage loss for all timesteps in parallel.
        """
        B, T, S = attn_weights.size()
        coverage = torch.cumsum(attn_weights, dim=1).clamp(max=1.0)  # cumulative coverage
        cov_loss = torch.min(attn_weights, coverage).sum(dim=-1)  # [B, T]
        if src_mask is not None:
            cov_loss = cov_loss.masked_fill(src_mask.any(dim=1, keepdim=True), 0.0)
        return self.coverage_weight * cov_loss.mean()
