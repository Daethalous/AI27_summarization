
"""Pointer-Generator Seq2Seq 模型 (带 Coverage)."""
from __future__ import annotations

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入 baseline encoder 和 coverage decoder
from ..baseline.encoder import Encoder
from .pg_coverage_decoder import PGCoverageDecoder


class PGCoverageSeq2Seq(nn.Module):
    """
    Pointer-Generator Network for text summarization with Coverage mechanism.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 256,
        hidden_size: int = 512,
        num_layers: int = 1,
        dropout: float = 0.1,
        pad_idx: int = 0,
        cov_loss_weight: float = 1.0  # 新增：接收覆盖损失权重
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.hidden_size = hidden_size
        self.cov_loss_weight = cov_loss_weight # 新增，保存为模型属性（可选，便于后续调整）

        # 1. 编码器 (BiLSTM, 输出维度为 hidden_size)
        self.encoder = Encoder(
            vocab_size, embed_size, hidden_size,
            num_layers, dropout, pad_idx
        )

        # 2. 解码器 (带 Coverage 的 PG Decoder)
        self.decoder = PGCoverageDecoder(
            vocab_size, embed_size, hidden_size,
            num_layers, dropout, pad_idx,
            cov_loss_weight=cov_loss_weight  # 传递覆盖损失权重到解码器
        )

        # 3. 桥接层: 将 BiLSTM 的 hidden_size//2 * 2 转换为 hidden_size
        self.bridge_h = nn.Linear(hidden_size, hidden_size)
        self.bridge_c = nn.Linear(hidden_size, hidden_size)

    def _bridge_states(
        self,
        hidden: torch.Tensor,
        cell: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert bidirectional encoder hidden states to unidirectional decoder states.
        hidden/cell: [num_layers*2, batch, hidden_size//2] -> [num_layers, batch, hidden_size]
        """
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
        src_oov_map: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # 1. 编码
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lens)
        # encoder_outputs: [batch, src_len, hidden_size]

        # 2. 桥接
        hidden, cell = self._bridge_states(hidden, cell)

        # 3. 解码 (包含 Coverage Loss 计算)
        use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio

        # outputs: [batch, tgt_len-1, extended_vocab_size]
        # coverage_loss: 标量
        outputs, _, _, coverage_loss = self.decoder(
            tgt=tgt,
            hidden=hidden,
            cell=cell,
            encoder_outputs=encoder_outputs,
            src_lens=src_lens,
            src_ids=src,
            src_oov_map=src_oov_map,
            coverage_vector=None, # 在 Decoder 内部初始化
            teacher_forcing=use_teacher_forcing
        )

        return outputs, None, None, coverage_loss

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

        if device is None:
            device = src.device

        batch_size = src.size(0)

        encoder_outputs, (hidden, cell) = self.encoder(src, src_lens)
        hidden, cell = self._bridge_states(hidden, cell)

        predictions = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
        all_attentions = []

        # ⭐ 1. 初始化 Coverage Vector
        src_len = src.size(1)
        current_coverage = torch.zeros(batch_size, src_len, device=device)

        decoder_input = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(max_length):
            # 2. 将当前 coverage 传入 forward_step
            dist, hidden, cell, attn_weights, _ = self.decoder.forward_step(
                decoder_input, hidden, cell,
                encoder_outputs, src_lens,
                src, src_oov_map,
                coverage_vector=current_coverage
            )

            # ⭐ 3. 更新 Coverage Vector: c_t = c_{t-1} + a_t
            current_coverage = current_coverage + attn_weights

            pred_id = dist.argmax(dim=1)
            predictions[:, t] = pred_id
            all_attentions.append(attn_weights)

            finished = finished | (pred_id == eos_idx)
            if finished.all():
                break

            decoder_input = torch.clamp(pred_id, 0, self.vocab_size - 1).unsqueeze(1)

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

        if device is None:
            device = src.device

        # 确保输入是单条序列 (Beam Search 通常用于 batch_size=1)
        assert src.size(0) == 1, "Beam search only supports batch size 1"

        encoder_outputs, (hidden, cell) = self.encoder(src, src_lens)
        hidden, cell = self._bridge_states(hidden, cell)

        # 初始化 Coverage Vector [1, src_len]
        src_len = src.size(1)
        initial_coverage = torch.zeros(1, src_len, device=device)

        # 扩展状态到 Beam
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

        # ⭐ 扩展 Coverage Vector
        coverage_vectors: List[torch.Tensor] = [initial_coverage.squeeze(0).clone()] * beam_size

        sequences = torch.full((beam_size, 1), sos_idx, dtype=torch.long, device=device)
        scores = torch.zeros(beam_size, device=device)
        scores[1:] = -float('inf')

        finished = []

        # 用于存储 Attention Weights (略，这里只关注序列)

        for t in range(max_length):
            decoder_input = sequences[:, -1].unsqueeze(1) # [beam, 1]

            # 聚合当前 beam 的所有 coverage vectors
            current_coverage = torch.stack(coverage_vectors, dim=0) # [beam, src_len]

            dist, hidden, cell, attn_weights, _ = self.decoder.forward_step(
                decoder_input, hidden, cell,
                encoder_outputs, src_lens_expanded,
                src_expanded, src_oov_map_expanded,
                coverage_vector=current_coverage
            ) # attn_weights: [beam, src_len]

            log_probs = torch.log(dist + 1e-10)
            next_scores = scores.unsqueeze(1) + log_probs

            next_scores_flat = next_scores.view(-1)
            topk_scores, topk_indices = next_scores_flat.topk(beam_size)

            beam_ids = topk_indices // dist.size(1) # 来自哪个 beam
            token_ids = topk_indices % dist.size(1) # 哪个 token

            next_sequences = []
            next_hidden = []
            next_cell = []
            next_scores_list = []
            # ⭐ 存储下一时间步的 Coverage
            next_coverage_vectors: List[torch.Tensor] = []

            for i in range(beam_size):
                beam_id = beam_ids[i]
                token_id = token_ids[i]
                score = topk_scores[i]

                seq = torch.cat([sequences[beam_id], token_id.unsqueeze(0)])

                # ⭐ 3. 更新并选择 Coverage
                # c_t = c_{t-1} + a_t
                new_coverage = coverage_vectors[beam_id] + attn_weights[beam_id]

                if token_id == eos_idx:
                    finished.append((seq, score.item()))
                else:
                    next_sequences.append(seq)
                    next_hidden.append(hidden[:, beam_id:beam_id+1, :])
                    next_cell.append(cell[:, beam_id:beam_id+1, :])
                    next_scores_list.append(score)
                    # new
                    next_coverage_vectors.append(new_coverage)

            if not next_sequences:
                break

            # 4. 更新 beams
            sequences = torch.nn.utils.rnn.pad_sequence(
                next_sequences, batch_first=True, padding_value=self.pad_idx
            )
            hidden = torch.cat(next_hidden, dim=1)
            cell = torch.cat(next_cell, dim=1)
            scores = torch.stack(next_scores_list)
            coverage_vectors = next_coverage_vectors # ⭐ 更新 Coverage List

            if len(finished) >= beam_size:
                break

        # 选择最佳序列 (包含长度惩罚等逻辑，这里简化为最高分)
        if finished:
            finished.sort(key=lambda x: x[1], reverse=True)
            best_seq = finished[0][0].unsqueeze(0)
        else:
            best_idx = scores.argmax()
            best_seq = sequences[best_idx].unsqueeze(0)

        # 简化返回 attention weights (Beam Search 的 Attention 追踪较为复杂，这里返回零矩阵)
        return best_seq, torch.zeros(1, best_seq.size(1), src.size(1), device=device)
