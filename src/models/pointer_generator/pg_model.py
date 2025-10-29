"""Pointer-Generator Seq2Seq 模型."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..baseline.encoder import Encoder
from .pg_decoder import PointerGeneratorDecoder


class PointerGeneratorSeq2Seq(nn.Module):
    """
    Pointer-Generator Network for text summarization.
    
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 256,
        hidden_size: int = 512,
        num_layers: int = 1,
        dropout: float = 0.1,
        max_oov_size: int = 1000,
        pad_idx: int = 0
    ) -> None:
        """
        Args:
            vocab_size: Size of vocabulary
            embed_size: Embedding dimension
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            pad_idx: Padding token index
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

        # 重用 baseline encoder (BiLSTM)
        self.encoder = Encoder(
            vocab_size, embed_size, hidden_size,
            num_layers, dropout, pad_idx
        )

        # 使用 Pointer-generator decoder
        self.decoder = PointerGeneratorDecoder(
            vocab_size, embed_size, hidden_size,
            num_layers, dropout, pad_idx
        )

        # 桥接层 把bidirectional encoder states 转换到 decoder states
        self.bridge_h = nn.Linear(hidden_size, hidden_size)
        self.bridge_c = nn.Linear(hidden_size, hidden_size)

    def _bridge_states(
        self,
        hidden: torch.Tensor,
        cell: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert bidirectional encoder hidden states to unidirectional decoder states.
        
        Args:
            hidden: [num_layers*2, batch, hidden_size//2]
            cell: [num_layers*2, batch, hidden_size//2]
        
        Returns:
            hidden: [num_layers, batch, hidden_size]
            cell: [num_layers, batch, hidden_size]
        """
        num_layers = hidden.size(0) // 2
        batch_size = hidden.size(1)

        hidden = hidden.view(num_layers, 2, batch_size, -1)

        # [num_layers, batch, hidden]
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)  

        cell = cell.view(num_layers, 2, batch_size, -1)
        cell = torch.cat([cell[:, 0], cell[:, 1]], dim=2)

        # 把双向LSTM的状态转成单向
        hidden = torch.tanh(self.bridge_h(hidden))
        cell = torch.tanh(self.bridge_c(cell))

        return hidden, cell

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_lens: Optional[torch.Tensor] = None,
        src_oov_map: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0
    ) -> torch.Tensor:
        """
        Forward pass with pointer-generator mechanism.
        
        Args:
            src: [batch, src_len] source token IDs
            tgt: [batch, tgt_len] target token IDs (with extended vocab for OOV)
            src_lens: [batch] source sequence lengths
            src_oov_map: [batch, src_len] OOV mapping (-1 for in-vocab, >=0 for OOV id)
            teacher_forcing_ratio: probability of using teacher forcing
        
        Returns:
            outputs: [batch, tgt_len-1, extended_vocab_size] probability distributions
        """
        # 1. 编码
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lens)

        # 2. 桥接
        hidden, cell = self._bridge_states(hidden, cell)
        # hidden/cell: [num_layers, batch, hidden_size]

        # 3. 解码
        use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
        
        outputs, _, _ = self.decoder(
            tgt=tgt,
            hidden=hidden,
            cell=cell,
            encoder_outputs=encoder_outputs,
            src_lens=src_lens,
            src_ids=src,
            src_oov_map=src_oov_map,
            teacher_forcing=use_teacher_forcing
        )
        # outputs: [batch, tgt_len-1, extended_vocab_size]

        return outputs

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
        """
        Greedy decoding for inference.
        
        Args:
            src: [batch, src_len]
            src_lens: [batch]
            src_oov_map: [batch, src_len]
            max_length: maximum generation length
            sos_idx: start of sequence token id
            eos_idx: end of sequence token id
            device: torch device
        
        Returns:
            predictions: [batch, max_length] generated token IDs
            attention_weights: [batch, max_length, src_len]
        """
        if device is None:
            device = src.device

        batch_size = src.size(0)

        encoder_outputs, (hidden, cell) = self.encoder(src, src_lens)
        hidden, cell = self._bridge_states(hidden, cell)

        predictions = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
        all_attentions = []
        
        decoder_input = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(max_length):
            dist, hidden, cell, attn_weights = self.decoder.forward_step(
                decoder_input, hidden, cell,
                encoder_outputs, src_lens,
                src, src_oov_map
            )
            
            # 选择概率最高的(greedy)
            pred_id = dist.argmax(dim=1)
            predictions[:, t] = pred_id
            all_attentions.append(attn_weights)
            
            # Check for EOS 遇到结束符则标记完成
            finished = finished | (pred_id == eos_idx)
            if finished.all():
                break
            
            decoder_input = torch.clamp(pred_id, 0, self.vocab_size - 1).unsqueeze(1)

        # Stack attention weights: [batch, length, src_len]
        attention_weights = torch.stack(all_attentions, dim=1)

        return predictions, attention_weights

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
        """
        Beam search decoding.
        
        Args:
            src: [1, src_len] single source sequence
            src_lens: [1]
            src_oov_map: [1, src_len]
            beam_size: beam width 同时探索5条路径
            max_length: maximum generation length
            sos_idx: start token
            eos_idx: end token
            device: torch device
        
        Returns:
            best_sequence: [1, length] best generated sequence
            best_attention: [1, length, src_len] attention weights
        """
        if device is None:
            device = src.device

        encoder_outputs, (hidden, cell) = self.encoder(src, src_lens)
        hidden, cell = self._bridge_states(hidden, cell)

        # Expand for beam
        # encoder_outputs: [1, src_len, hidden] -> [beam, src_len, hidden]
        encoder_outputs = encoder_outputs.expand(beam_size, -1, -1)
        hidden = hidden.expand(-1, beam_size, -1).contiguous()
        cell = cell.expand(-1, beam_size, -1).contiguous()
        src_expanded = src.expand(beam_size, -1)
        if src_oov_map is not None:
            src_oov_map_expanded = src_oov_map.expand(beam_size, -1)
        else:
            src_oov_map_expanded = None
        if src_lens is not None:
            src_lens_expanded = src_lens.expand(beam_size)
        else:
            src_lens_expanded = None

        sequences = torch.full((beam_size, 1), sos_idx, dtype=torch.long, device=device)
        scores = torch.zeros(beam_size, device=device)
        scores[1:] = -float('inf')  # Only keep first beam active initially

        finished = []

        for t in range(max_length):
            decoder_input = sequences[:, -1].unsqueeze(1)
            
            dist, hidden, cell, _ = self.decoder.forward_step(
                decoder_input, hidden, cell,
                encoder_outputs, src_lens_expanded,
                src_expanded, src_oov_map_expanded
            )

            log_probs = torch.log(dist + 1e-10)  # [beam, vocab]

            next_scores = scores.unsqueeze(1) + log_probs  # [beam, vocab]

            # Flatten and get top-k
            next_scores_flat = next_scores.view(-1)  # [beam * vocab]
            topk_scores, topk_indices = next_scores_flat.topk(beam_size)

            beam_ids = topk_indices // dist.size(1)
            token_ids = topk_indices % dist.size(1)

            # Update sequences
            next_sequences = []
            next_hidden = []
            next_cell = []
            next_scores = []

            for i in range(beam_size):
                beam_id = beam_ids[i]
                token_id = token_ids[i]
                score = topk_scores[i]

                seq = torch.cat([sequences[beam_id], token_id.unsqueeze(0)])

                if token_id == eos_idx:
                    finished.append((seq, score.item()))
                else:
                    next_sequences.append(seq)
                    next_hidden.append(hidden[:, beam_id:beam_id+1, :])
                    next_cell.append(cell[:, beam_id:beam_id+1, :])
                    next_scores.append(score)

            if not next_sequences:
                break

            # Update beams
            sequences = torch.nn.utils.rnn.pad_sequence(
                next_sequences, batch_first=True, padding_value=self.pad_idx
            )
            hidden = torch.cat(next_hidden, dim=1)
            cell = torch.cat(next_cell, dim=1)
            scores = torch.tensor(next_scores, device=device)

            # If we have enough finished sequences, stop
            if len(finished) >= beam_size:
                break

        # Select best sequence
        if finished:
            finished.sort(key=lambda x: x[1], reverse=True)
            best_seq = finished[0][0].unsqueeze(0)
        else:
            best_idx = scores.argmax()
            best_seq = sequences[best_idx].unsqueeze(0)

        return best_seq, torch.zeros(1, best_seq.size(1), src.size(1), device=device)