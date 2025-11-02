
"""通用 Pointer-Generator 机制实现（与编码器/解码器架构无关）"""
from __future__ import annotations

from typing import Optional, Tuple # 导入 Tuple 以修正返回类型
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerGenerator(nn.Module):
    """实现 Pointer-Generator 核心逻辑：混合词典生成分布与源文本复制分布"""
    def __init__(self, hidden_size: int, embed_size: int, vocab_size: int):
        super().__init__()
        # 生成概率 p_gen 计算层（输入：解码器输出 + 上下文向量 + 嵌入向量）
        self.mid_linear = nn.Linear(hidden_size * 2, hidden_size)
        self.p_gen_linear = nn.Linear(hidden_size * 2 + embed_size, 1)
        # 修正: vocab_linear 的输入应该是 mid_output 的大小，即 hidden_size
        self.vocab_linear = nn.Linear(hidden_size, vocab_size)

    def compute_final_dist(
        self,
        decoder_output: torch.Tensor,  # 解码器输出（LSTM隐藏态或Transformer输出）[batch, hidden_size]
        context: torch.Tensor,         # 上下文向量（注意力加权的编码器输出）[batch, hidden_size]
        embedded: torch.Tensor,        # 解码器输入嵌入 [batch, embed_size]
        vocab_size: int,               # 基础词表大小
        attn_weights: torch.Tensor,     # 注意力权重（用于复制分布）[batch, src_len]
        src_ids: Optional[torch.Tensor] = None,  # 源文本词表索引 [batch, src_len]
        src_oov_map: Optional[torch.Tensor] = None  # OOV映射表 [batch, src_len]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # 修正返回类型
        """
        计算最终混合分布：p_gen * 词典分布 + (1-p_gen) * 复制分布
        完全复用原逻辑，仅通过参数适配不同解码器架构
        """
        batch_size = decoder_output.size(0)

        # 1. 计算生成概率 p_gen（控制生成/复制比例）
        p_gen_input = torch.cat([decoder_output, context, embedded], dim=1)  # [batch, 2*hidden + embed]
        p_gen = torch.sigmoid(self.p_gen_linear(p_gen_input))  # [batch, 1]

        # 2. 计算词典生成分布 P_vocab()
        vocab_input = torch.cat([decoder_output, context], dim=1)  # [batch, 2*hidden]
        mid_output = torch.tanh(self.mid_linear(vocab_input)) # tanh(V [st, h*t] + b) -> [batch, hidden_size]
        vocab_logits = self.vocab_linear(mid_output)  # [batch, hidden_size] -> [batch, vocab_size]
        vocab_logits = torch.clamp(vocab_logits, -50, 50) #防止溢出
        vocab_dist = F.softmax(vocab_logits, dim=1)  # [batch, vocab_size]

        # 3. 处理OOV：构建扩展词表分布（包含源文本中出现的OOV词）
        if src_ids is not None and src_oov_map is not None and src_oov_map.max() >= 0:
            max_oov = src_oov_map.max().item() if src_oov_map.max() >= 0 else -1
            extended_size = vocab_size + max_oov + 1 if max_oov >= 0 else vocab_size # 基础词表 + OOV词数量
        else:
            extended_size = vocab_size

        # 初始化扩展词表分布（基础词表部分先填充P_vocab）
        extended_vocab_dist = torch.zeros(
            batch_size, extended_size,
            device=vocab_dist.device,
            dtype=vocab_dist.dtype
        )
        extended_vocab_dist[:, :vocab_size] = vocab_dist

        # 4. 计算复制分布（基于注意力权重复制源文本词）
        copy_dist = torch.zeros_like(extended_vocab_dist)
        if src_ids is not None:
            # 复制索引映射：将源文本词索引（含OOV）映射到扩展词表
            copy_indices = src_ids.clone()  # [batch, src_len]
            if src_oov_map is not None:
                oov_mask = src_oov_map >= 0  # OOV词掩码
                copy_indices[oov_mask] = vocab_size + src_oov_map[oov_mask]  # OOV词映射到扩展区域
            assert attn_weights.shape == src_ids.shape, f"attn_weights shape {attn_weights.shape} != src_ids shape {src_ids.shape}"
            # 用注意力权重填充复制分布（按索引累加）
            copy_dist.scatter_add_(1, copy_indices, attn_weights)

        # 5. 最终混合分布
        final_dist = p_gen * extended_vocab_dist + (1 - p_gen) * copy_dist
        return final_dist, p_gen.squeeze(-1), vocab_dist
