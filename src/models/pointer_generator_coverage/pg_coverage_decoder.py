
"""Pointer-Generator Decoder with OOV handling and Coverage mechanism (Strict Additive Attention)."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 仍然导入 LuongAttention (或确保其存在)
from ..baseline.decoder import LuongAttention 

class PGCoverageDecoder(nn.Module):
    
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        pad_idx: int = 0
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        
        # 1. Attention 模块：Additive Attention (遵循 See et al. 2017 公式 9)
        
        # W_h (作用于 encoder outputs h_i)。
        # encoder_outputs 维度为 [B, L_src, hidden_size]
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False) # 修正: 输入是 hidden_size
        # W_s (作用于 decoder hidden state s_t)
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)
        # W_c (作用于 coverage vector c_{t-1}^i)
        self.W_c = nn.Linear(1, hidden_size, bias=False) 
        # v 和 b_attn (公式 (9) 中的 V^T 和 b_attn)
        self.V = nn.Linear(hidden_size, 1, bias=False) 
        self.b_attn = nn.Parameter(torch.zeros(1)) # 公式 (9) 中的 b_attn

        # LuongAttention 仅作占位或如果其他地方仍需使用
        self.attention = LuongAttention(hidden_size) 
        
        self.lstm = nn.LSTM(
            embed_size + hidden_size, # LSTM Input 仍然是 embedded + context (hidden_size + hidden_size)
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.p_gen_linear = nn.Linear(hidden_size * 2 + embed_size, 1) # hidden_size (LSTM) + hidden_size (Context) + embed_size
        self.out = nn.Linear(hidden_size * 2, vocab_size) # hidden_size (LSTM) + hidden_size (Context)
        self.dropout = nn.Dropout(dropout)
    
    def coverage_attention(
        self,
        decoder_hidden: torch.Tensor, # [batch, hidden_size] (s_t)
        encoder_outputs: torch.Tensor, # [batch, src_len, hidden_size] (h_i)
        coverage_vector: torch.Tensor, # [batch, src_len] (c_{t-1})
        src_lens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算带 Coverage 的 Additive Attention (遵循 See et al. 2017 公式 9, 10)
        """
        batch_size, src_len, enc_hidden_size = encoder_outputs.size()
        
        # 0. 准备 decoder_hidden s_t ( top layer )
        # [batch, 1, hidden]
        s_t = decoder_hidden.unsqueeze(1) 
        
        # 1. 计算 W_h * h_i 
        # Wh_h: [batch, src_len, hidden]
        Wh_h = self.W_h(encoder_outputs) 

        # 2. 计算 W_s * s_t 
        # Ws_s: [batch, 1, hidden]
        Ws_s = self.W_s(s_t) 
        
        # 3. 计算 W_c * c_{t-1}
        # Wc_c: [batch, src_len, hidden]
        coverage_input = coverage_vector.unsqueeze(-1)
        Wc_c = self.W_c(coverage_input) 

        # 4. Attention Input (公式 9: Wh_h + Ws_s + Wc_c + b_attn)
        # Ws_s 需要自动 broadcast 到 [batch, src_len, hidden]
        attn_input = Wh_h + Ws_s + Wc_c + self.b_attn
        
        # 5. Attention Score e_i^t = V^T tanh(...) (公式 9)
        # scores: [batch, src_len]
        scores = self.V(torch.tanh(attn_input)).squeeze(-1) 

        # 6. Masking and Softmax
        if src_lens is not None:
            mask = torch.arange(src_len, device=scores.device)[None, :] >= src_lens[:, None]
            scores = scores.masked_fill(mask, -1e9) 

        # Attention Weights a_t (公式 10)
        attn_weights = F.softmax(scores, dim=1) # [batch, src_len]
        
        # 7. Context Vector 
        # context: [batch, hidden_size]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1) 
        
        # 8. Coverage Loss (L_t^cov = sum_i min(a_t^i, c_{t-1}^i))
        coverage_loss_t = torch.sum(torch.min(attn_weights, coverage_vector), dim=1) # [batch]
        
        return context, attn_weights, coverage_loss_t

    def forward_step(
        self,
        y_prev: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: torch.Tensor, # [batch, src_len, hidden_size]
        src_lens: Optional[torch.Tensor] = None,
        src_ids: Optional[torch.Tensor] = None,
        src_oov_map: Optional[torch.Tensor] = None,
        coverage_vector: Optional[torch.Tensor] = None, 
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        batch_size = y_prev.size(0)
        if y_prev.dim() == 2:
            y_prev = y_prev.squeeze(1) 
            
        embedded = self.dropout(self.embedding(y_prev))
        
        # 1. 计算带 Coverage 的 Attention 和 Loss
        # context: [batch, hidden_size]
        context, attn_weights, coverage_loss_t = self.coverage_attention(
            hidden[-1], encoder_outputs, coverage_vector, src_lens
        ) 
        
        # 2. LSTM 前向传播
        # lstm_input: [batch, embed_size + hidden_size]
        lstm_input = torch.cat([embedded, context], dim=1).unsqueeze(1)
        lstm_output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        lstm_output = lstm_output.squeeze(1) # lstm_output: [batch, hidden_size]

        # 3. P_gen 计算
        # p_gen_input: [batch, hidden_size (LSTM) + hidden_size (Context) + embed_size]
        p_gen_input = torch.cat([lstm_output, context, embedded], dim=1)
        p_gen = torch.sigmoid(self.p_gen_linear(p_gen_input)) # [batch, 1]

        # 4. 词典生成分布 (P_vocab)
        # vocab_input: [batch, hidden_size * 2]
        vocab_input = torch.cat([lstm_output, context], dim=1)
        vocab_logits = self.out(vocab_input)
        vocab_dist = F.softmax(vocab_logits, dim=1) # [batch, vocab_size]
        
        # 5. OOV Handling 和混合分布 (保持不变)
        if src_ids is not None and src_oov_map is not None:
            max_oov = src_oov_map.max().item()
            extended_size = self.vocab_size + max(0, max_oov + 1)
        else:
            extended_size = self.vocab_size
        
        extended_vocab_dist = torch.zeros(
            batch_size, extended_size,
            device=vocab_dist.device,
            dtype=vocab_dist.dtype
        )
        extended_vocab_dist[:, :self.vocab_size] = vocab_dist
        
        if src_ids is not None:
            copy_indices = src_ids.clone()
            
            if src_oov_map is not None:
                oov_mask = src_oov_map >= 0
                copy_indices[oov_mask] = self.vocab_size + src_oov_map[oov_mask]
            
            copy_dist = torch.zeros_like(extended_vocab_dist)
            copy_dist.scatter_add_(1, copy_indices, attn_weights)
        else:
            copy_dist = torch.zeros_like(extended_vocab_dist)
        
        final_dist = p_gen * extended_vocab_dist + (1 - p_gen) * copy_dist
        
        return final_dist, hidden, cell, attn_weights, coverage_loss_t

    # forward 和其他方法保持不变
    def forward(
        self,
        tgt: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_lens: Optional[torch.Tensor] = None,
        src_ids: Optional[torch.Tensor] = None,
        src_oov_map: Optional[torch.Tensor] = None,
        coverage_vector: Optional[torch.Tensor] = None, 
        teacher_forcing: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        tgt_len = tgt.size(1)
        batch_size = tgt.size(0)
        
        all_dists = []
        all_coverage_loss_t = []
        decoder_input = tgt[:, 0].unsqueeze(1) 
        
        if coverage_vector is None:
            src_len = encoder_outputs.size(1)
            coverage_vector = torch.zeros(batch_size, src_len, device=tgt.device)
            
        current_coverage = coverage_vector.clone()
        
        for t in range(1, tgt_len):
            dist, hidden, cell, attn_weights, coverage_loss_t = self.forward_step(
                decoder_input, hidden, cell,
                encoder_outputs, src_lens,
                src_ids, src_oov_map,
                coverage_vector=current_coverage
            )
            all_dists.append(dist)
            all_coverage_loss_t.append(coverage_loss_t)
            
            current_coverage = current_coverage + attn_weights
            
            if teacher_forcing:
                decoder_input = tgt[:, t].unsqueeze(1)
            else:
                decoder_input = dist.argmax(dim=1, keepdim=True)
                decoder_input = torch.clamp(decoder_input, 0, self.vocab_size - 1)
        
        all_dists = torch.stack(all_dists, dim=1)
        
        all_coverage_loss_t = torch.stack(all_coverage_loss_t, dim=0) 
        coverage_loss = COV_LOSS_WEIGHT * all_coverage_loss_t.mean()
        
        return all_dists, hidden, cell, coverage_loss
