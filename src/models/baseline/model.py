"""整体 Seq2Seq + Attention 模型."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder


class Seq2Seq(nn.Module):
    """Seq2Seq 模型封装."""

    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 256,
        hidden_size: int = 512,
        num_layers: int = 1,
        dropout: float = 0.1,
        pad_idx: int = 0
    ) -> None:
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_size, hidden_size, num_layers, dropout, pad_idx)
        self.decoder = Decoder(vocab_size, embed_size, hidden_size, num_layers, dropout, pad_idx)
        self.bridge_h = nn.Linear(hidden_size, hidden_size)
        self.bridge_c = nn.Linear(hidden_size, hidden_size)
        self.pad_idx = pad_idx

    def _bridge_states(
        self,
        hidden: torch.Tensor,
        cell: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_layers = hidden.size(0) // 2
        batch_size = hidden.size(1)

        hidden = hidden.view(num_layers, 2, batch_size, -1)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)

        cell = cell.view(num_layers, 2, batch_size, -1)
        cell = torch.cat([cell[:, 0], cell[:, 1]], dim=2)

        hidden = torch.tanh(self.bridge_h(hidden))
        cell = torch.tanh(self.bridge_c(cell))
        return hidden, cell

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_lens: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0
    ) -> torch.Tensor:
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.decoder.out.out_features

        encoder_outputs, (hidden, cell) = self.encoder(src, src_lens)
        hidden, cell = self._bridge_states(hidden, cell)

        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=src.device)
        decoder_input = tgt[:, 0].unsqueeze(1)

        for t in range(1, tgt_len):
            output, hidden, cell, _ = self.decoder(decoder_input, hidden, cell, encoder_outputs, src_lens)
            outputs[:, t] = output

            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            if use_teacher_forcing:
                decoder_input = tgt[:, t].unsqueeze(1)
            else:
                decoder_input = output.argmax(1).unsqueeze(1)

        return outputs
