
"""通用 Coverage 机制实现（与编码器/解码器架构无关）"""
from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoverageMechanism(nn.Module):
    """实现 Coverage 机制核心逻辑：跟踪历史注意力并抑制重复生成"""
    def __init__(self, hidden_size: int):
        super().__init__()
        # 注意力计算相关线性层（遵循 See et al. 2017 公式 9）
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)  # 作用于编码器输出 h_i
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)  # 作用于解码器输出 s_t
        self.W_c = nn.Linear(1, hidden_size, bias=False)            # 作用于覆盖向量 c_{t-1}
        self.V = nn.Linear(hidden_size, 1, bias=False)              # 注意力分数映射
        self.b_attn = nn.Parameter(torch.zeros(1))                  # 偏置项

    def compute_coverage_attention(
        self,
        decoder_output: torch.Tensor,  # 解码器输出（LSTM隐藏态或Transformer输出）[batch, hidden_size]
        encoder_outputs: torch.Tensor, # 编码器输出 [batch, src_len, hidden_size]
        coverage_vector: torch.Tensor, # 覆盖向量（累计历史注意力）[batch, src_len]
        src_lens: Optional[torch.Tensor] = None  # 源文本长度（用于掩码）[batch]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算带覆盖机制的注意力权重、上下文向量和覆盖损失
        完全复用原逻辑，仅通过参数适配不同解码器架构
        """
        batch_size, src_len, _ = encoder_outputs.size()

        # 1. 准备解码器输出（扩展维度以匹配广播）
        s_t = decoder_output.unsqueeze(1)  # [batch, 1, hidden_size]（与源文本序列长度对齐）

        # 2. 计算注意力分数组件（公式 9）
        Wh_h = self.W_h(encoder_outputs)          # [batch, src_len, hidden_size]
        Ws_s = self.W_s(s_t)                      # [batch, 1, hidden_size]（广播到src_len）
        coverage_input = coverage_vector.unsqueeze(-1)  # [batch, src_len, 1]
        Wc_c = self.W_c(coverage_input)           # [batch, src_len, hidden_size]

        # 3. 合并组件计算注意力分数
        attn_input = Wh_h + Ws_s + Wc_c + self.b_attn  # [batch, src_len, hidden_size]
        scores = self.V(torch.tanh(attn_input)).squeeze(-1)  # [batch, src_len]

        # 4. 掩码处理（屏蔽源文本PAD位置）
        if src_lens is not None:
            # 生成掩码：src_len 之外的位置为 True（需要屏蔽）
            mask = torch.arange(src_len, device=scores.device)[None, :] >= src_lens[:, None]
            scores = scores.masked_fill(mask, -1e9)  # 屏蔽位置分数设为负无穷

        # 5. 计算注意力权重（公式 10）
        attn_weights = F.softmax(scores, dim=1)  # [batch, src_len]

        # 6. 计算上下文向量（注意力加权的编码器输出）
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [batch, hidden_size]

        # 7. 计算覆盖损失（公式 12）：L_t^cov = sum(min(a_t^i, c_{t-1}^i))
        coverage_loss_t = torch.sum(torch.min(attn_weights, coverage_vector), dim=1)  # [batch]

        return context, attn_weights, coverage_loss_t
