"""Transformer Decoder for PGCT, integrating Pointer-Generator and Coverage (parallelized training)."""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn

from .pointer_generator import PointerGenerator
from .coverage import CoverageMechanism
from .pgct_encoder import PositionalEncoding


class PGCTDecoder(nn.Module):
    """Transformer decoder with Pointer-Generator and Coverage (PGCT)."""
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 512,
        hidden_size: int = 512,
        num_layers: int = 3,
        nhead: int = 8,
        dropout: float = 0.1,
        pad_idx: int = 0,
        cov_loss_weight: float = 1.0,
        max_tgt_len: int = 100
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.pad_idx = pad_idx
        self.cov_loss_weight = cov_loss_weight
        self.max_tgt_len = max_tgt_len

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.embed_proj = nn.Linear(embed_size, hidden_size) if embed_size != hidden_size else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        self.coverage = CoverageMechanism(hidden_size, coverage_loss_weight=cov_loss_weight)
        self.pointer = PointerGenerator(hidden_size, embed_size, vocab_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True  # 保持 batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.pos_encoding = PositionalEncoding(d_model=hidden_size, max_len=max_tgt_len, dropout=dropout)

    def generate_tgt_mask(self, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            tgt_mask: [T, T] float mask for causal attention (float('-inf') where masked)
            tgt_key_padding_mask: [B, T] bool mask
        """
        B, T = tgt.size()
        causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=tgt.device), diagonal=1)
        tgt_key_padding_mask = (tgt == self.pad_idx)
        return causal_mask, tgt_key_padding_mask

    def forward_step(
        self,
        tgt_step: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor,
        coverage_vector: torch.Tensor,
        src_ids: torch.Tensor,
        src_oov_map: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        PGCT single-step forward. 完全兼容 batch_first=True/False。
        """
        tgt_embed = self.embedding(tgt_step)           # [B, T, D]
        tgt_embed = self.embed_proj(tgt_embed)
        tgt_embed = self.pos_encoding(tgt_embed)

        B, T, D = tgt_embed.size()

        # 修复单步 T=1 时的 mask
        tgt_mask = None
        if T > 1:
            causal_mask, _ = self.generate_tgt_mask(tgt_step)
            tgt_mask = causal_mask

        tgt_key_padding_mask = (tgt_step == self.pad_idx)

        dec_output = self.transformer_decoder(
            tgt=tgt_embed,                             # [B, T, D], batch_first=True
            memory=encoder_outputs,                    # [B, S, D], batch_first=True
            tgt_mask=tgt_mask,                         # [T, T] 或 None
            tgt_key_padding_mask=tgt_key_padding_mask, # [B, T]
            memory_key_padding_mask=src_mask.bool() if src_mask is not None else None
        )

        dec_output_timestep = dec_output[:, -1, :]
        context, attn_weights, coverage_vector, cov_loss_t = self.coverage.compute_stepwise_attention(
            decoder_output=dec_output_timestep,
            encoder_outputs=encoder_outputs,
            coverage_vector=coverage_vector,
            src_mask=src_mask
        )

        raw_embedded_t = self.embedding(tgt_step[:, -1])
        final_dist, _, _ = self.pointer.compute_final_dist(
            decoder_output=dec_output_timestep,
            context=context,
            embedded=raw_embedded_t,
            vocab_size=self.vocab_size,
            src_ids=src_ids,
            src_oov_map=src_oov_map,
            attn_weights=attn_weights,
        )

        return final_dist, attn_weights, coverage_vector, cov_loss_t

    def forward(
        self,
        tgt: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        src_oov_map: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        PGCT forward for training. 完全兼容 batch_first=True/False。
        """
        tgt_input = tgt[:, :-1]
        tgt_mask, tgt_key_padding_mask = self.generate_tgt_mask(tgt_input)

        tgt_embed = self.embedding(tgt_input)           # [B, T, D]
        raw_embedded = tgt_embed
        tgt_embed = self.embed_proj(tgt_embed)
        tgt_embed = self.pos_encoding(tgt_embed)

        dec_output = self.transformer_decoder(
            tgt=tgt_embed,
            memory=encoder_outputs,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_mask
        )

        context_all_steps, all_attn_weights, total_cov_loss = self.coverage.compute_parallel_training(
            decoder_outputs=dec_output,
            encoder_outputs=encoder_outputs,
            src_mask=src_mask
        )

        all_dists = []
        for t in range(tgt_input.size(1)):
            final_dist, _, _ = self.pointer.compute_final_dist(
                decoder_output=dec_output[:, t, :],
                context=context_all_steps[:, t, :],
                embedded=raw_embedded[:, t, :],
                vocab_size=self.vocab_size,
                src_ids=src,
                src_oov_map=src_oov_map,
                attn_weights=all_attn_weights[:, t, :],
            )
            all_dists.append(final_dist)

        all_dists = torch.stack(all_dists, dim=1)
        return all_dists, total_cov_loss.squeeze(0)
