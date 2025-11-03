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

    if src_mask.dim() == 3:  # <--新增：防止被错误扩展
        src_mask = src_mask.squeeze(1)
    src_mask = src_mask.bool()
    
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
            # 修正接口调用：接收 4 个返回值，并忽略第 4 个 (cov_loss_t)
            final_dist, attn_weights, new_coverage_vector, _ = model.decoder.forward_step(
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
                # 使用 forward_step 返回的更新后的覆盖向量
                new_cov = new_coverage_vector 
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

@torch.no_grad()
def pgct_greedy_decode(
    model: nn.Module,
    src: torch.Tensor,
    src_lens: Optional[torch.Tensor] = None,
    src_oov_map: Optional[torch.Tensor] = None,
    max_length: int = 100,
    sos_idx: int = 2,
    eos_idx: int = 3,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transformer贪心解码（PGCT专用, 支持 Pointer-Generator + Coverage）"""
    model.eval()
    device = device or src.device
    encoder_outputs, _ = model.encoder(src, src_lens)
    src_mask = (src == model.pad_idx)

    if src_mask.dim() == 3: # <--新增
        src_mask = src_mask.squeeze(1)
    src_mask = src_mask.bool()
    
    src_len = encoder_outputs.size(1)

    seq = [sos_idx]
    coverage = torch.zeros(1, src_len, device=device)
    attn_list = []

    for _ in range(max_length):
        decoder_input = torch.tensor([seq], dtype=torch.long, device=device)
        # 修正接口调用：接收 4 个返回值，并忽略第 4 个 (cov_loss_t)
        final_dist, attn_weights, new_coverage, _ = model.decoder.forward_step(
            tgt_step=decoder_input,
            encoder_outputs=encoder_outputs,
            src_mask=src_mask,
            coverage_vector=coverage,
            src_ids=src,
            src_oov_map=src_oov_map,
        )

        # 取最后一步输出分布
        # NOTE: final_dist 应该是 [1, 1, extended_vocab_size]，需要正确索引
        logits = final_dist.squeeze(0) 
        next_token = torch.argmax(logits, dim=-1).item()

        seq.append(next_token)
        coverage = new_coverage  # 使用 decoder 返回的更新版覆盖向量
        attn_list.append(attn_weights.cpu())

        if next_token == eos_idx:
            break

    attn_tensor = torch.stack(attn_list, dim=0) if attn_list else torch.zeros(1, src_len)
    return torch.tensor([seq], device=device), attn_tensor.unsqueeze(0)
