"""Transformer Decoder for PGCT, integrating Pointer-Generator and Coverage (parallelized training)."""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointer_generator import PointerGenerator
from .coverage import CoverageMechanism
from .pgct_encoder import PositionalEncoding


class PGCTDecoder(nn.Module):
    """Transformer decoder with Pointer-Generator and Coverage (PGCT)."""
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

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.embed_proj = nn.Linear(embed_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Core modules
        self.coverage = CoverageMechanism(hidden_size, coverage_loss_weight=cov_loss_weight) # 初始化时传入权重
        self.pointer = PointerGenerator(hidden_size, embed_size, vocab_size)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.pos_encoding = PositionalEncoding(d_model=hidden_size, max_len=max_tgt_len, dropout=dropout)

    # -------------------------------------------------------------------------
    # Utility masks
    # -------------------------------------------------------------------------
    def generate_tgt_mask(self, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成目标序列掩码（因果掩码 + <PAD>掩码）"""
        tgt_len = tgt.size(1)
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1).bool()
        pad_mask = (tgt == self.pad_idx)
        return causal_mask, pad_mask

    # -------------------------------------------------------------------------
    # Stepwise decoding (for inference / beam search)
    # -------------------------------------------------------------------------
    # 保持不变，用于推理阶段
    def forward_step(
        self,
        tgt_step: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor,
        coverage_vector: torch.Tensor,
        src_ids: torch.Tensor,
        src_oov_map: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """单步解码（用于推理阶段）"""
        tgt_embed = self.embedding(tgt_step)
        tgt_embed = self.embed_proj(tgt_embed)
        tgt_embed = self.pos_encoding(tgt_embed)
        
        # Transformer解码器调用（未修正缺失的因果掩码和tgt_key_padding_mask，保持代码不变）
        # 注意: 如果 tgt_step 包含多个时间步，这里缺少 causal mask 和 key padding mask 可能会导致推理错误。
        dec_output = self.transformer_decoder(
            tgt=tgt_embed,
            memory=encoder_outputs,
            memory_key_padding_mask=src_mask
        )
        dec_output_t = dec_output[:, -1, :]
        
        # Coverage attention
        # 注意：此处调用 compute_coverage_attention 应该对应 coverage.py 中的 compute_stepwise_attention
        context, attn_weights, coverage_vector, cov_loss_t = self.coverage.compute_stepwise_attention(
            decoder_output=dec_output_t,
            encoder_outputs=encoder_outputs,
            coverage_vector=coverage_vector,
            src_mask=src_mask
        )
        
        # Pointer mechanism
        raw_embedded_t = self.embedding(tgt_step[:, -1])
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

    # -------------------------------------------------------------------------
    # Parallelized training forward (已修正为并行设计)
    # -------------------------------------------------------------------------
    def forward(
        self,
        tgt: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        src_oov_map: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True # 保留参数签名，但忽略其值
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """训练阶段前向传播（并行 coverage 实现）"""
        batch_size, tgt_len = tgt.size()
        
        # 1. Standard Transformer Setup (Parallel)
        tgt_input = tgt[:, :-1]
        tgt_len_in = tgt_input.size(1)
        
        tgt_mask, tgt_key_padding_mask = self.generate_tgt_mask(tgt_input)
        tgt_embed = self.embedding(tgt_input)
        raw_embedded = tgt_embed # [B, T, E]
        tgt_embed = self.embed_proj(tgt_embed)
        tgt_embed = self.pos_encoding(tgt_embed)
        
        # 2. Parallel Transformer Decode
        dec_output = self.transformer_decoder(
            tgt=tgt_embed,
            memory=encoder_outputs,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_mask
        ) # [B, T, H]

        # 3. Parallel Coverage Calculation (一次性计算所有步骤的注意力、上下文和损失)
        # Context [B, T, H], all_attn_weights [B, T, S], total_cov_loss [1]
        context_all_steps, all_attn_weights, total_cov_loss = self.coverage.compute_parallel_training(
            decoder_outputs=dec_output,
            encoder_outputs=encoder_outputs,
            src_mask=src_mask
        )
        
        # 4. Pointer-Generator Output (迭代 T，但使用并行结果)
        all_dists = []
        for t in range(tgt_len_in):
            # context 和 attn_weights 使用并行计算得到的第 t 步结果
            final_dist, _, _ = self.pointer.compute_final_dist( 
                decoder_output=dec_output[:, t, :],
                context=context_all_steps[:, t, :], 
                embedded=raw_embedded[:, t, :],
                vocab_size=self.vocab_size,
                src_ids=src,
                src_oov_map=src_oov_map,
                attn_weights=all_attn_weights[:, t, :]
            )
            all_dists.append(final_dist)

        all_dists = torch.stack(all_dists, dim=1)
        
        # total_cov_loss 是已经加权和平均的标量损失 (返回时挤压维度)
        return all_dists, total_cov_loss.squeeze(0)
