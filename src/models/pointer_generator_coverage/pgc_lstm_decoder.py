
"""Pointer-Generator Decoder with OOV handling and Coverage mechanism（复用通用模块）"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

# 导入通用模块
from .pointer_generator import PointerGenerator
from .coverage import CoverageMechanism
# 保留 baseline 中的 LuongAttention（如需兼容）
from ..baseline.decoder import LuongAttention


class PGCoverageDecoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        pad_idx: int = 0,
        cov_loss_weight: float = 1.0  # 覆盖损失权重
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.pad_idx = pad_idx
        self.cov_loss_weight = cov_loss_weight  # 覆盖损失权重

        # 基础组件
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)

        # 复用通用模块（替代原手写的注意力和指针逻辑）
        self.coverage = CoverageMechanism(hidden_size)  # 覆盖机制
        self.pointer = PointerGenerator(                # 指针生成机制
            hidden_size=hidden_size,
            embed_size=embed_size,
            vocab_size=vocab_size
        )

        # LSTM 解码器（特有组件，与架构相关）
        self.lstm = nn.LSTM(
            input_size=embed_size + hidden_size,  # 输入：嵌入向量 + 上下文向量
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 保留 LuongAttention 占位（如需兼容其他逻辑）
        self.attention = LuongAttention(hidden_size)

    def forward_step(
        self,
        y_prev: torch.Tensor,  # 上一步输出 token [batch, 1] 或 [batch]
        hidden: torch.Tensor,  # LSTM 隐藏态 [num_layers, batch, hidden_size]
        cell: torch.Tensor,    # LSTM 细胞态 [num_layers, batch, hidden_size]
        encoder_outputs: torch.Tensor,  # 编码器输出 [batch, src_len, hidden_size]
        src_lens: Optional[torch.Tensor] = None,  # 源文本长度 [batch]
        src_ids: Optional[torch.Tensor] = None,   # 源文本词表索引 [batch, src_len]
        src_oov_map: Optional[torch.Tensor] = None,  # OOV映射表 [batch, src_len]
        coverage_vector: Optional[torch.Tensor] = None,  # 覆盖向量 [batch, src_len]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """单步解码（复用通用模块计算注意力和混合分布）"""
        batch_size = y_prev.size(0)
        # 处理输入维度（确保为 [batch]）
        if y_prev.dim() == 2:
            y_prev = y_prev.squeeze(1)

        # 1. 词嵌入 + dropout
        embedded = self.dropout(self.embedding(y_prev))  # [batch, embed_size]

        # 2. 复用 Coverage 机制计算注意力和覆盖损失
        # 取 LSTM 顶层隐藏态作为解码器输出
        context, attn_weights, coverage_loss_t = self.coverage.compute_coverage_attention(
            decoder_output=hidden[-1],  # [batch, hidden_size]
            encoder_outputs=encoder_outputs,
            coverage_vector=coverage_vector,
            src_lens=src_lens
        )

        # 3. LSTM 前向传播
        lstm_input = torch.cat([embedded, context], dim=1).unsqueeze(1)  # [batch, 1, embed+hidden]
        lstm_output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        lstm_output = lstm_output.squeeze(1)  # [batch, hidden_size]

        # 4. 复用 Pointer-Generator 机制计算混合分布
        final_dist = self.pointer.compute_final_dist(
            decoder_output=lstm_output,  # LSTM输出作为解码器状态
            context=context,
            embedded=embedded,
            vocab_size=self.vocab_size,
            src_ids=src_ids,
            src_oov_map=src_oov_map,
            attn_weights=attn_weights
        )

        return final_dist, hidden, cell, attn_weights, coverage_loss_t

    def forward(
        self,
        tgt: torch.Tensor,  # 目标序列 [batch, tgt_len]
        hidden: torch.Tensor,  # LSTM 初始隐藏态 [num_layers, batch, hidden_size]
        cell: torch.Tensor,    # LSTM 初始细胞态 [num_layers, batch, hidden_size]
        encoder_outputs: torch.Tensor,  # 编码器输出 [batch, src_len, hidden_size]
        src_lens: Optional[torch.Tensor] = None,
        src_ids: Optional[torch.Tensor] = None,
        src_oov_map: Optional[torch.Tensor] = None,
        coverage_vector: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """完整解码过程（与原逻辑一致，仅内部调用复用模块）"""
        tgt_len = tgt.size(1)
        batch_size = tgt.size(0)

        all_dists = []
        all_coverage_loss_t = []
        decoder_input = tgt[:, 0].unsqueeze(1)  # 初始输入：<SOS>

        # 初始化覆盖向量（如未提供）
        if coverage_vector is None:
            src_len = encoder_outputs.size(1)
            coverage_vector = torch.zeros(batch_size, src_len, device=tgt.device)
        current_coverage = coverage_vector.clone()

        # 逐时间步解码
        for t in range(1, tgt_len):
            # 调用复用模块的 forward_step
            dist, hidden, cell, attn_weights, coverage_loss_t = self.forward_step(
                decoder_input, hidden, cell,
                encoder_outputs, src_lens,
                src_ids, src_oov_map,
                coverage_vector=current_coverage
            )
            all_dists.append(dist)
            all_coverage_loss_t.append(coverage_loss_t)

            # 更新覆盖向量（累计注意力权重）
            current_coverage = current_coverage + attn_weights

            # 教师强制或自回归输入
            if teacher_forcing:
                decoder_input = tgt[:, t].unsqueeze(1)
            else:
                decoder_input = dist.argmax(dim=1, keepdim=True)
                decoder_input = torch.clamp(decoder_input, 0, self.vocab_size - 1)

        # 聚合结果
        all_dists = torch.stack(all_dists, dim=1)  # [batch, tgt_len-1, extended_vocab_size]
        all_coverage_loss_t = torch.stack(all_coverage_loss_t, dim=0)  # [tgt_len-1, batch]
        coverage_loss = self.cov_loss_weight * all_coverage_loss_t.mean()  # 覆盖损失加权

        return all_dists, hidden, cell, coverage_loss
