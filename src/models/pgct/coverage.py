"""
Coverage Mechanism for Pointer-Generator Transformer (PGCT)
------------------------------------------------------------
实现了论文《Get To The Point》中基于加性注意力的 Coverage 机制（公式 9）。
同时支持 Stepwise (推理) 和 Parallel (训练) 模式。
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoverageMechanism(nn.Module):
    # CRITICAL: 隐藏层维度必须与 Transformer 的 d_model 一致
    def __init__(self, hidden_size: int, coverage_loss_weight: float = 1.0):
        super().__init__()
        self.coverage_loss_weight = coverage_loss_weight
        self.hidden_size = hidden_size
        self.MAX_LOGIT = 30.0 # 数值稳定性钳制值

        # 严格遵循论文公式 (9) 实现加性注意力组件
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)  # 作用于编码器输出 h_i
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)  # 作用于解码器输出 s_t
        # Coverage Vector 是 [B, S]，每个元素是标量，W_c 负责将其映射到 H 维
        self.W_c = nn.Linear(1, hidden_size, bias=False)            # 作用于覆盖向量 c_{t-1}
        self.V = nn.Linear(hidden_size, 1, bias=False)              # 注意力分数映射 v^T
        self.b_attn = nn.Parameter(torch.zeros(hidden_size))        # 偏置项 b (作用于 hidden_size 维度)

    # 1. Stepwise Coverage Attention (用于推理 / 循序解码)
    def compute_stepwise_attention(
        self,
        decoder_output: torch.Tensor,  # [B, H] (解码器当前隐藏态 s_t)
        encoder_outputs: torch.Tensor, # [B, S, H] (编码器输出 h_i)
        coverage_vector: torch.Tensor, # [B, S] (历史覆盖向量 c_{t-1})
        src_mask: Optional[torch.Tensor] = None, # [B, S] (源文本 PAD 掩码)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算带覆盖机制的注意力权重、上下文向量、新覆盖向量和覆盖损失（Stepwise）"""
        B, S, H = encoder_outputs.size()

        # 1. 准备解码器输出 s_t (扩展维度以匹配广播)
        s_t_expanded = decoder_output.unsqueeze(1)  # [B, 1, H]

        # 2. 计算 Attention Score 组件 (遵循论文公式 9)
        Wh_h = self.W_h(encoder_outputs)            # [B, S, H]
        Ws_s = self.W_s(s_t_expanded)               # [B, 1, H] (通过广播相加)

        # Wc_c: W_c * c_{t-1}. Input [B, S, 1] -> Output [B, S, H]
        coverage_input = coverage_vector.unsqueeze(-1) 
        Wc_c = self.W_c(coverage_input)             # [B, S, H] 

        # 3. 合并组件计算注意力 Logits (score)
        # attn_input = W_h h_i + W_s s_t + W_c c_{t-1} + b (H-dimensional)
        attn_input = Wh_h + Ws_s + Wc_c + self.b_attn # [B, S, H]
        # attn_logits = v^T * tanh(attn_input) (1-dimensional)
        attn_logits = self.V(torch.tanh(attn_input)).squeeze(-1) # [B, S]

        # 4. CRITICAL FIX: 钳制 Logits 防止 Softmax 溢出
        attn_logits = torch.clamp(attn_logits, min=-self.MAX_LOGIT, max=self.MAX_LOGIT) 

        # 5. 掩码处理 (屏蔽源文本 PAD 位置)
        if src_mask is not None:
            attn_logits = attn_logits.masked_fill(src_mask, float('-inf')) 

        # 6. 计算注意力权重 (公式 10)
        attn_weights = F.softmax(attn_logits, dim=1)  # [B, S]

        # 7. 计算上下文向量
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [B, H]

        # 8. 计算覆盖损失 (公式 12) 和更新覆盖向量
        coverage_loss_t = torch.sum(torch.min(attn_weights, coverage_vector), dim=1) # [B]
        new_coverage_vector = coverage_vector + attn_weights

        cov_loss_t = (self.coverage_loss_weight * coverage_loss_t.mean()).unsqueeze(0)

        # 遵循 PGCTDecoder 的调用约定返回
        return context, attn_weights, new_coverage_vector, cov_loss_t

    # 2. Parallel Coverage Attention (用于训练 / 批处理所有时间步)
    def compute_parallel_training(
        self,
        decoder_outputs: torch.Tensor,  # [B, T, H] (所有 T 步解码器输出 s_t)
        encoder_outputs: torch.Tensor,  # [B, S, H]
        src_mask: Optional[torch.Tensor] = None, # [B, S]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算所有时间步的注意力、上下文向量和总覆盖损失（Parallel）。
        """
        B, T, H = decoder_outputs.size()
        S = encoder_outputs.size(1)

        # --- 第一阶段：计算 Attention Input components (Wh_h, Ws_s, b_attn) ---
        Wh_h = self.W_h(encoder_outputs).unsqueeze(1)    # [B, 1, S, H] (广播 T)
        Ws_s = self.W_s(decoder_outputs).unsqueeze(2)    # [B, T, 1, H] (广播 S)

        # Base Attention Input (Wh_h + Ws_s + b): [B, T, S, H]
        E_base_input = Wh_h + Ws_s + self.b_attn # [B, T, S, H]

        # --- 第二阶段：计算 c_{t-1} 覆盖向量 ---

        # Preliminary Logits E' (不含覆盖项) - 用于计算 c_{t-1}
        # A_prelim = softmax(V * tanh(E_base_input))
        E_prelim = self.V(torch.tanh(E_base_input)).squeeze(-1) # E_prelim [B, T, S]

        # 钳制 Logits 防止 Softmax 溢出
        E_prelim_clamped = torch.clamp(E_prelim, min=-self.MAX_LOGIT, max=self.MAX_LOGIT) 
        if src_mask is not None:
            # 广播 src_mask 以匹配 [B, T, S]
            E_prelim_clamped = E_prelim_clamped.masked_fill(src_mask.unsqueeze(1), float('-inf'))

        A_prelim = F.softmax(E_prelim_clamped, dim=-1) # A_prelim [B, T, S]

        # 计算 c_{t-1}：累加并错位 (c_0 = 0)
        cumulative_attention = torch.cumsum(A_prelim, dim=1) # c_t (包含当前 a_t)

        coverage_vector_shifted = torch.cat([
            torch.zeros(B, 1, S, device=A_prelim.device, dtype=A_prelim.dtype),
            cumulative_attention[:, :-1, :]
        ], dim=1) # C_{t-1} [B, T, S]

        # --- 第三阶段：计算最终 Logits E (含覆盖项 Wc_c) 和 A_final ---

        # Wc_c term: W_c * c_{t-1}. Input [B, T, S, 1] -> Output [B, T, S, H]
        # CRITICAL FIX: 确保 Wc_c 保持 H 维，以便与 E_base_input 相加 (遵循加性注意力公式)
        Wc_c = self.W_c(coverage_vector_shifted.unsqueeze(-1)) # Wc_c [B, T, S, H] 

        # E_final_input = (Wh_h + Ws_s + b) + Wc_c
        E_final_input = E_base_input + Wc_c # [B, T, S, H]

        # E_final = V * tanh(E_final_input)
        E_final = self.V(torch.tanh(E_final_input)).squeeze(-1) # E_final [B, T, S]

        # 钳制和掩码
        E_final_clamped = torch.clamp(E_final, min=-self.MAX_LOGIT, max=self.MAX_LOGIT) 
        if src_mask is not None:
            # 广播 src_mask 以匹配 [B, T, S]
            E_final_clamped = E_final_clamped.masked_fill(src_mask.unsqueeze(1), float('-inf'))

        A_final = F.softmax(E_final_clamped, dim=-1) # A_final [B, T, S]

        # 计算上下文 Context [B, T, H]
        context = torch.einsum('bts,bsh->bth', A_final, encoder_outputs) 

        # 计算总覆盖损失 L_cov (公式 12)
        # Loss = sum(min(A_final, C_{t-1}))
        coverage_loss_t = torch.sum(torch.min(A_final, coverage_vector_shifted), dim=-1) # [B, T]

        # 损失归一化并加权
        cov_loss = (self.coverage_loss_weight * coverage_loss_t.mean()).unsqueeze(0)

        # 针对 Transformer 的 PGCTModel 调用，返回 Context, A_final, Loss
        return context, A_final, cov_loss
