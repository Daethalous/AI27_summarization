"""Pointer-Generator Decoder with OOV handling."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..baseline.decoder import LuongAttention

class PointerGeneratorDecoder(nn.Module):

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
        self.attention = LuongAttention(hidden_size)
        self.lstm = nn.LSTM(
            embed_size + hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.p_gen_linear = nn.Linear(hidden_size * 2 + embed_size, 1)
        self.out = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward_step(
        self,
        y_prev: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_lens: Optional[torch.Tensor] = None,
        src_ids: Optional[torch.Tensor] = None,
        src_oov_map: Optional[torch.Tensor] = None,
    )-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        batch_size = y_prev.size(0)
        if y_prev.dim() == 2:
            y_prev = y_prev.squeeze(1) 
        embedded = self.dropout(self.embedding(y_prev))
        context, attn_weights = self.attention(hidden[-1], encoder_outputs, src_lens)

        lstm_input = torch.cat([embedded, context], dim=1).unsqueeze(1)
        lstm_output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        lstm_output = lstm_output.squeeze(1)

        vocab_input = torch.cat([lstm_output, context], dim=1)
        vocab_logits = self.out(vocab_input)
        # 从词典中生成分布
        vocab_dist = F.softmax(vocab_logits, dim=1)

        p_gen_input = torch.cat([lstm_output, context, embedded], dim=1)
        p_gen = torch.sigmoid(self.p_gen_linear(p_gen_input))

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

        if src_ids is not None and src_oov_map is not None:            
            copy_indices = src_ids.clone()
            
            oov_mask = src_oov_map >= 0
            copy_indices[oov_mask] = self.vocab_size + src_oov_map[oov_mask]
            
            copy_dist = torch.zeros_like(extended_vocab_dist)
            # 通过注意力权重分配拷贝概率
            copy_dist.scatter_add_(1, copy_indices, attn_weights)
        else:
            if src_ids is not None:
                copy_dist = torch.zeros_like(extended_vocab_dist)
                copy_dist.scatter_add_(1, src_ids, attn_weights)
            else:
                copy_dist = torch.zeros_like(extended_vocab_dist)
        
        # 混合生成和拷贝分布
        final_dist = p_gen * extended_vocab_dist + (1 - p_gen) * copy_dist
        
        return final_dist, hidden, cell, attn_weights

    def forward(
        self,
        tgt: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_lens: Optional[torch.Tensor] = None,
        src_ids: Optional[torch.Tensor] = None,
        src_oov_map: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass through decoder.
        
        Args:
            tgt: [batch, tgt_len] target sequence
            hidden/cell: encoder states
            encoder_outputs: [batch, src_len, hidden_size]
            src_lens: [batch]
            src_ids: [batch, src_len]
            src_oov_map: [batch, src_len]
            teacher_forcing: if True, always use ground truth; else use predictions
        
        Returns:
            all_dists: [batch, tgt_len, extended_vocab_size]
            hidden: final hidden state
            cell: final cell state
        """
        tgt_len = tgt.size(1)
        
        all_dists = []
        decoder_input = tgt[:, 0].unsqueeze(1)  # Start with <SOS>
        
        for t in range(1, tgt_len):
            # Single step
            dist, hidden, cell, _ = self.forward_step(
                decoder_input, hidden, cell,
                encoder_outputs, src_lens,
                src_ids, src_oov_map
            )
            all_dists.append(dist)
            
            # 训练模式使用真实标签，推理模式使用预测结果
            if teacher_forcing:
                decoder_input = tgt[:, t].unsqueeze(1)
            else:
                decoder_input = dist.argmax(dim=1, keepdim=True)
                decoder_input = torch.clamp(decoder_input, 0, self.vocab_size - 1)
        
        all_dists = torch.stack(all_dists, dim=1)
        
        return all_dists, hidden, cell