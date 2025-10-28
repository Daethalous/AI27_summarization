"""Decoder 模块：Luong Attention + LSTM 解码器."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LuongAttention(nn.Module):
    """Luong General 注意力机制."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_lens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query = decoder_hidden.unsqueeze(1)
        keys = self.W(encoder_outputs)
        scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1)

        if src_lens is not None:
            mask = torch.arange(encoder_outputs.size(1), device=scores.device)[None, :] >= src_lens[:, None]
            scores = scores.masked_fill(mask, -1e9)

        attn_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights


class Decoder(nn.Module):
    """单向 LSTM 解码器，带注意力."""

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
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.attention = LuongAttention(hidden_size)
        self.lstm = nn.LSTM(
            embed_size + hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.out = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_lens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(tgt))
        context, attn_weights = self.attention(hidden[-1], encoder_outputs, src_lens)
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        lstm_output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        lstm_output = lstm_output.squeeze(1)
        output = self.out(torch.cat([lstm_output, context], dim=1))
        return output, hidden, cell, attn_weights
