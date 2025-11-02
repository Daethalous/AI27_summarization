"""Transformer Decoder for PGCT, integrating Pointer-Generator and Coverage."""
from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointer_generator import PointerGenerator
from .coverage import CoverageMechanism
from .pgct_encoder import PositionalEncoding


class PGCTDecoder(nn.Module):
    """Transformer解码器（复用PG+Coverage机制）"""
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 256,
        hidden_size: int = 256,
        num_layers: int = 3,
        nhead: int = 8,
        dropout: float = 0.1,
        pad_idx: int = 0,
        cov_loss_weight: float = 1.0,
        max_tgt_len: int = 100
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = hidden_size
        self.pad_idx = pad_idx
        self.cov_loss_weight = cov_loss_weight
        self.max_tgt_len = max_tgt_len
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.embed_proj = nn.Linear(embed_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.coverage = CoverageMechanism(hidden_size)
        self.pointer = PointerGenerator(
            hidden_size=hidden_size,
            embed_size=embed_size,
            vocab_size=vocab_size
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.pos_encoding = PositionalEncoding(d_model=hidden_size, max_len=max_tgt_len, dropout=dropout)

    def generate_tgt_mask(self, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成目标序列掩码（因果掩码 + <PAD>掩码）"""
        tgt_len = tgt.size(1)
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1).bool()
        pad_mask = (tgt == self.pad_idx)
        return causal_mask, pad_mask
    
    # 修正: forward_step 的返回类型应为 4 个张量
    def forward_step(
        self,
        tgt_step: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor,
        coverage_vector: torch.Tensor,
        src_ids: torch.Tensor,
        src_oov_map: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """单步解码"""
        tgt_embed = self.embedding(tgt_step)
        tgt_embed = self.embed_proj(tgt_embed)
        tgt_embed = self.pos_encoding(tgt_embed)
        
        dec_output = self.transformer_decoder(
            tgt=tgt_embed,
            memory=encoder_outputs,
            memory_key_padding_mask=src_mask
        )
        dec_output_t = dec_output[:, -1, :]
        
        context, attn_weights, coverage_vector, cov_loss_t = self.coverage.compute_coverage_attention(
            decoder_output=dec_output_t,
            encoder_outputs=encoder_outputs,
            coverage_vector=coverage_vector,
            src_mask=src_mask
        )
        
        raw_embedded_t = self.embedding(tgt_step[:, -1])
        # 修正: 解包 pointer.compute_final_dist 的返回值
        final_dist, _, _ = self.pointer.compute_final_dist(
            decoder_output=dec_output_t,
            context=context,
            embedded=raw_embedded_t,
            vocab_size=self.vocab_size,
            src_ids=src_ids,
            src_oov_map=src_oov_map,
            attn_weights=attn_weights
        )
        return final_dist, attn_weights, coverage_vector, cov_loss_t

    def forward(
        self,
        tgt: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src: torch.Tensor,
        src_mask: torch.Tensor, # 修正: 补充缺失的 src_mask 参数
        src_oov_map: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """训练阶段前向传播"""
        batch_size, tgt_len = tgt.size()
        all_dists = []
        total_coverage_loss = 0.0
        coverage_vector = torch.zeros(batch_size, encoder_outputs.size(1), device=tgt.device)
        
        if teacher_forcing:
            tgt_input = tgt[:, :-1]
            tgt_len_in = tgt_input.size(1)
            
            tgt_mask, tgt_key_padding_mask = self.generate_tgt_mask(tgt_input)
            tgt_embed = self.embedding(tgt_input)
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
            
            for t in range(tgt_len_in):
                step_output = dec_output[:, t, :]
                
                context, attn_weights, coverage_vector, cov_loss_t = self.coverage.compute_coverage_attention(
                    decoder_output=step_output,
                    encoder_outputs=encoder_outputs,
                    coverage_vector=coverage_vector,
                    src_mask=src_mask
                )
                total_coverage_loss += cov_loss_t.mean()
                
                # 修正: 解包 pointer.compute_final_dist 的返回值
                final_dist, _, _ = self.pointer.compute_final_dist(
                    decoder_output=step_output,
                    context=context,
                    embedded=raw_embedded[:, t, :],
                    vocab_size=self.vocab_size,
                    src_ids=src,
                    src_oov_map=src_oov_map,
                    attn_weights=attn_weights
                )
                all_dists.append(final_dist)
        
        all_dists = torch.stack(all_dists, dim=1) if all_dists else torch.tensor([], device=tgt.device)
        total_coverage_loss = self.cov_loss_weight * (total_coverage_loss / (tgt_len - 1)) if tgt_len > 1 else torch.tensor(0.0, device=tgt.device)
        
        return all_dists, total_coverage_loss
