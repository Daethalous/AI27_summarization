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

    batch_size = src.size(0)
    src_len = encoder_outputs.size(1)

    sequences = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
    coverage = torch.zeros(batch_size, src_len, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    attn_lists: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]

    for _ in range(max_length):
        prev_finished = finished.clone()

        # 修正接口调用：接收 4 个返回值，并忽略第 4 个 (cov_loss_t)
        final_dist, attn_weights, new_coverage, _ = model.decoder.forward_step(
            tgt_step=sequences,
            encoder_outputs=encoder_outputs,
            src_mask=src_mask,
            coverage_vector=coverage,
            src_ids=src,
            src_oov_map=src_oov_map,
        )

        logits = final_dist  # [B, V_ext]
        next_tokens = torch.argmax(logits, dim=-1)

        # 保留已完成样本的覆盖向量，防止重复更新
        coverage = torch.where(prev_finished.unsqueeze(1), coverage, new_coverage)

        # 只追踪尚未完成样本的注意力
        attn_weights_cpu = attn_weights.detach().cpu()
        for b in range(batch_size):
            if not prev_finished[b]:
                attn_lists[b].append(attn_weights_cpu[b])

        # 对已完成样本强制输出 EOS，避免继续生成
        eos_tensor = torch.full_like(next_tokens, eos_idx)
        next_tokens = torch.where(prev_finished, eos_tensor, next_tokens)

        sequences = torch.cat([sequences, next_tokens.unsqueeze(1)], dim=1)

        finished = prev_finished | (next_tokens == eos_idx)
        if torch.all(finished):
            break

    # 如果尚未生成 EOS，则补充一次 EOS，确保后续处理安全
    if sequences.size(1) == 1:  # 仅包含 SOS 的极端情况
        sequences = torch.cat([sequences, torch.full((batch_size, 1), eos_idx, device=device, dtype=torch.long)], dim=1)

    # 构建注意力张量并自动填充
    max_attn_steps = max((len(lst) for lst in attn_lists), default=0)
    if max_attn_steps == 0:
        attn_tensor = torch.zeros(batch_size, 1, src_len)
    else:
        attn_tensor = torch.zeros(batch_size, max_attn_steps, src_len)
        for b, attns in enumerate(attn_lists):
            if attns:
                stacked = torch.stack(attns, dim=0)
                attn_tensor[b, :stacked.size(0)] = stacked

    return sequences, attn_tensor
