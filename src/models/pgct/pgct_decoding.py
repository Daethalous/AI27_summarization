"""Transformer-based PGCT decoding (Greedy + Beam Search)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class BeamNode:
    """束搜索节点：存储序列、分数、注意力、覆盖向量"""
    def __init__(
        self,
        sequence: List[int],
        score: float,
        coverage_vector: torch.Tensor,
        log_prob: float,
        attn_weights_list: Optional[List[torch.Tensor]] = None
    ):
        self.sequence = sequence
        self.score = score
        self.coverage_vector = coverage_vector
        self.log_prob = log_prob
        self.attn_weights_list = attn_weights_list or []

    def __lt__(self, other):
        return self.score < other.score


@torch.no_grad()
def pgct_beam_search_decode(
    model: nn.Module,
    src: torch.Tensor,
    src_lens: Optional[torch.Tensor] = None,
    src_oov_map: Optional[torch.Tensor] = None,
    beam_size: int = 5,
    max_length: int = 100,
    sos_idx: int = 2,
    eos_idx: int = 3,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transformer束搜索（PGCT专用，保存attn_weights_list）"""
    model.eval()
    device = device or src.device
    encoder_outputs, _ = model.encoder(src, src_lens)
    src_mask = (src == model.pad_idx)
    src_len = encoder_outputs.size(1)

    beams = [BeamNode([sos_idx], 0.0, torch.zeros(1, src_len, device=device), 0.0)]
    completed = []

    for _ in range(max_length):
        candidates = []
        if not any(node.sequence[-1] != eos_idx for node in beams):
            completed.extend(beams)
            break

        for node in beams:
            if node.sequence[-1] == eos_idx:
                completed.append(node)
                continue

            decoder_input = torch.tensor([node.sequence], dtype=torch.long, device=device)
            final_dist, attn_weights, _ = model.decoder.forward_step(
                tgt_step=decoder_input,
                encoder_outputs=encoder_outputs,
                src_mask=src_mask,
                coverage_vector=node.coverage_vector,
                src_ids=src,
                src_oov_map=src_oov_map,
            )

            log_probs = torch.log(final_dist + 1e-12).squeeze(0)
            topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)

            for log_prob_t, token_id in zip(topk_log_probs, topk_ids):
                new_seq = node.sequence + [token_id.item()]
                new_cov = node.coverage_vector + attn_weights
                new_attns = node.attn_weights_list + [attn_weights.cpu()]
                candidates.append(
                    BeamNode(new_seq, node.score + log_prob_t.item(), new_cov, node.log_prob + log_prob_t.item(), new_attns)
                )

        beams = sorted(candidates, key=lambda x: x.score, reverse=True)[:beam_size]

    completed.extend(beams)
    completed.sort(key=lambda x: x.score, reverse=True)
    best = completed[0]

    attn_tensor = torch.stack(best.attn_weights_list, dim=0) if best.attn_weights_list else torch.zeros(1, src_len)
    return torch.tensor([best.sequence], device=device), attn_tensor.unsqueeze(0)
