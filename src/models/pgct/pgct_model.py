
"""Transformer-Based Pointer-Generator with Coverage Mechanism (PGCT) Model."""
from __future__ import annotations

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 导入通用机制模块
from .pointer_generator import PointerGenerator
from .coverage import CoverageMechanism
# 导入外部解码工具函数（复用贪婪解码和束搜索）
from src.utils.decoding import greedy_decode, beam_search_decode


class PositionalEncoding(nn.Module):
    """Transformer位置编码（正弦余弦编码）"""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PGCTEncoder(nn.Module):
    """Transformer编码器（适配PG+Coverage机制）"""
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 256,
        hidden_size: int = 256,
        num_layers: int = 3,
        nhead: int = 8,
        dropout: float = 0.1,
        pad_idx: int = 0,
        max_src_len: int = 400
    ):
        super().__init__()
        self.d_model = hidden_size
        self.pad_idx = pad_idx
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.embed_proj = nn.Linear(embed_size, hidden_size)
        self.pos_encoding = PositionalEncoding(d_model=hidden_size, max_len=max_src_len, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def generate_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """生成源序列掩码（屏蔽<PAD>）"""
        return (src == self.pad_idx)

    def forward(
        self,
        src: torch.Tensor,
        src_lens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, None]:
        embed = self.embedding(src)
        embed = self.embed_proj(embed)
        embed = self.pos_encoding(embed)
        
        src_mask = self.generate_src_mask(src)
        encoder_outputs = self.transformer_encoder(embed, src_key_padding_mask=src_mask)
        
        return encoder_outputs, None


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
        
        # 复用通用机制模块
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

    def generate_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """生成目标序列掩码（因果掩码+<PAD>掩码）"""
        batch_size, tgt_len = tgt.size()
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        pad_mask = (tgt == self.pad_idx).unsqueeze(1)
        return causal_mask | pad_mask

    def forward_step(
        self,
        tgt_step: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor,
        coverage_vector: torch.Tensor,
        src_ids: torch.Tensor,
        src_oov_map: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """单步解码（供外部解码函数调用）"""
        batch_size = tgt_step.size(0)
        
        # 目标序列嵌入与位置编码
        tgt_embed = self.embedding(tgt_step)
        tgt_embed = self.embed_proj(tgt_embed)
        tgt_embed = self.pos_encoding(tgt_embed)
        
        # 生成目标掩码
        tgt_mask = self.generate_tgt_mask(tgt_step)
        
        # Transformer解码器前向
        dec_output = self.transformer_decoder(
            tgt=tgt_embed,
            memory=encoder_outputs,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_mask
        )
        dec_output = dec_output.squeeze(1)
        
        # 复用Coverage机制计算注意力
        context, attn_weights, coverage_loss_t = self.coverage.compute_coverage_attention(
            decoder_output=dec_output,
            encoder_outputs=encoder_outputs,
            coverage_vector=coverage_vector,
            src_lens=None
        )
        
        # 复用Pointer机制计算混合分布
        embedded = self.embedding(tgt_step).squeeze(1)
        final_dist = self.pointer.compute_final_dist(
            decoder_output=dec_output,
            context=context,
            embedded=embedded,
            vocab_size=self.vocab_size,
            src_ids=src_ids,
            src_oov_map=src_oov_map,
            attn_weights=attn_weights
        )
        
        return final_dist, attn_weights, coverage_loss_t

    def forward(
        self,
        tgt: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src: torch.Tensor,
        src_oov_map: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """训练阶段前向传播"""
        batch_size, tgt_len = tgt.size()
        src_len = encoder_outputs.size(1)
        
        all_dists = []
        total_coverage_loss = 0.0
        coverage_vector = torch.zeros(batch_size, src_len, device=tgt.device)
        src_mask = (src == self.pad_idx)  # 源序列掩码
        
        if teacher_forcing:
            tgt_input = tgt[:, :-1]
            tgt_mask = self.generate_tgt_mask(tgt_input)
            
            tgt_embed = self.embedding(tgt_input)
            tgt_embed = self.embed_proj(tgt_embed)
            tgt_embed = self.pos_encoding(tgt_embed)
            
            dec_output = self.transformer_decoder(
                tgt=tgt_embed,
                memory=encoder_outputs,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_mask
            )
            
            for t in range(tgt_len - 1):
                step_output = dec_output[:, t, :]
                context, attn_weights, cov_loss_t = self.coverage.compute_coverage_attention(
                    decoder_output=step_output,
                    encoder_outputs=encoder_outputs,
                    coverage_vector=coverage_vector,
                    src_lens=None
                )
                total_coverage_loss += cov_loss_t.mean()
                coverage_vector += attn_weights
                
                embedded = tgt_embed[:, t, :]
                final_dist = self.pointer.compute_final_dist(
                    decoder_output=step_output,
                    context=context,
                    embedded=embedded,
                    vocab_size=self.vocab_size,
                    src_ids=src,
                    src_oov_map=src_oov_map,
                    attn_weights=attn_weights
                )
                all_dists.append(final_dist)
        
        all_dists = torch.stack(all_dists, dim=1) if all_dists else torch.tensor([])
        total_coverage_loss = self.cov_loss_weight * (total_coverage_loss / (tgt_len - 1)) if tgt_len > 1 else 0.0
        
        return all_dists, total_coverage_loss


class PGCTModel(nn.Module):
    """完整的Transformer+Pointer-Generator+Coverage模型"""
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 256,
        hidden_size: int = 256,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        nhead: int = 8,
        dropout: float = 0.1,
        pad_idx: int = 0,
        cov_loss_weight: float = 1.0,
        max_src_len: int = 400,
        max_tgt_len: int = 100
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.max_tgt_len = max_tgt_len
        
        self.encoder = PGCTEncoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_encoder_layers,
            nhead=nhead,
            dropout=dropout,
            pad_idx=pad_idx,
            max_src_len=max_src_len
        )
        
        self.decoder = PGCTDecoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_decoder_layers,
            nhead=nhead,
            dropout=dropout,
            pad_idx=pad_idx,
            cov_loss_weight=cov_loss_weight,
            max_tgt_len=max_tgt_len
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_lens: Optional[torch.Tensor] = None,
        src_oov_map: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0
    ) -> Tuple[torch.Tensor, None, None, torch.Tensor]:
        """训练阶段前向传播"""
        encoder_outputs, _ = self.encoder(src, src_lens)
        use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
        
        outputs, coverage_loss = self.decoder(
            tgt=tgt,
            encoder_outputs=encoder_outputs,
            src=src,
            src_oov_map=src_oov_map,
            teacher_forcing=use_teacher_forcing
        )
        
        return outputs, None, None, coverage_loss

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        src_lens: Optional[torch.Tensor] = None,
        src_oov_map: Optional[torch.Tensor] = None,
        max_length: int = 100,
        sos_idx: int = 2,
        eos_idx: int = 3,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """调用外部贪婪解码函数（复用src/utils/decoding.py）"""
        # 调用工具函数执行贪婪解码
        decoded_ids, attn_weights_list = greedy_decode(
            model=self,                  # 传入当前模型实例
            src=src,
            src_lens=src_lens,
            src_oov_map=src_oov_map,
            max_len=max_length,
            sos_idx=sos_idx,
            eos_idx=eos_idx,
            device=device
        )
        
        # 转换为Tensor格式（保持接口兼容）
        predictions = torch.tensor(decoded_ids, dtype=torch.long, device=device).unsqueeze(0)
        attn_weights = torch.tensor(attn_weights_list, dtype=torch.float, device=device).unsqueeze(0)
        
        return predictions, attn_weights

    @torch.no_grad()
    def beam_search(
        self,
        src: torch.Tensor,
        src_lens: Optional[torch.Tensor] = None,
        src_oov_map: Optional[torch.Tensor] = None,
        beam_size: int = 5,
        max_length: int = 100,
        sos_idx: int = 2,
        eos_idx: int = 3,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """调用外部束搜索函数（复用src/utils/decoding.py）"""
        # 调用工具函数执行束搜索
        beam_results = beam_search_decode(
            model=self,                  # 传入当前模型实例
            src=src,
            src_lens=src_lens,
            src_oov_map=src_oov_map,
            max_len=max_length,
            beam_size=beam_size,
            sos_idx=sos_idx,
            eos_idx=eos_idx,
            device=device
        )
        
        # 处理结果（取最优序列，保持接口兼容）
        best_seq = beam_results[0][0]  # 第一个元素为最优序列
        src_len = src.size(1)
        predictions = torch.tensor(best_seq, dtype=torch.long, device=device).unsqueeze(0)
        attn_weights = torch.zeros(1, len(best_seq), src_len, device=device)  # 简化注意力返回
        
        return predictions, attn_weights
