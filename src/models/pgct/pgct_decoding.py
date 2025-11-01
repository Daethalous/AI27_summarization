"""
解码策略：Greedy Decoding 和 Beam Search（适配 PGCTModel 和 LSTM 模型）
"""
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
from src.models.pointer_generator_coverage.pgct_model import PGCTModel
from src.models.pointer_generator_coverage.pg_coverage_model import PGCoverageSeq2Seq  # LSTM模型


def greedy_decode(
    model: Union[PGCTModel, PGCoverageSeq2Seq],
    src: torch.Tensor,
    src_lens: Optional[torch.Tensor] = None,
    src_oov_map: Optional[torch.Tensor] = None,
    max_len: int = 100,
    sos_idx: int = 2,
    eos_idx: int = 3,
    device: Optional[torch.device] = None
) -> Tuple[List[int], List[List[float]]]:
    """贪婪解码（同时适配 Transformer+PG+Coverage 和 LSTM+PG+Coverage 模型）"""
    model.eval()
    if device is None:
        device = src.device

    with torch.no_grad():
        # 1. 编码（适配两种模型）
        if isinstance(model, PGCTModel):
            # Transformer 编码器
            encoder_outputs, _ = model.encoder(src, src_lens)
            src_mask = model.encoder.generate_src_mask(src)
            hidden, cell = None, None  # Transformer 无 LSTM 状态
        else:
            # LSTM 编码器
            encoder_outputs, (hidden, cell) = model.encoder(src, src_lens)
            hidden, cell = model._bridge_states(hidden, cell)
            src_mask = None  # LSTM 无需源掩码

        # 2. 初始化解码状态
        decoded_ids = []
        attention_weights_list = []
        decoder_input = torch.full((1, 1), sos_idx, dtype=torch.long, device=device)
        coverage_vector = torch.zeros(1, src.size(1), device=device) if isinstance(model, PGCTModel) else None

        # 3. 逐步解码（适配两种模型）
        for _ in range(max_len):
            if isinstance(model, PGCTModel):
                # Transformer 解码（调用 decoder.forward_step）
                final_dist, attn_weights, _ = model.decoder.forward_step(
                    tgt_step=decoder_input,
                    encoder_outputs=encoder_outputs,
                    src_mask=src_mask,
                    coverage_vector=coverage_vector,
                    src_ids=src,
                    src_oov_map=src_oov_map
                )
                coverage_vector = coverage_vector + attn_weights  # 更新覆盖向量
            else:
                # LSTM 解码（调用原解码器）
                final_dist, hidden, cell, attn_weights = model.decoder.forward_step(
                    y_prev=decoder_input,
                    hidden=hidden,
                    cell=cell,
                    encoder_outputs=encoder_outputs,
                    src_lens=src_lens,
                    src_ids=src,
                    src_oov_map=src_oov_map,
                    coverage_vector=coverage_vector
                )
                coverage_vector = coverage_vector + attn_weights if coverage_vector is not None else None

            # 记录结果
            pred_id = final_dist.argmax(dim=1).item()
            decoded_ids.append(pred_id)
            attention_weights_list.append(attn_weights.cpu().squeeze().tolist())

            # 遇到 EOS 停止
            if pred_id == eos_idx:
                break

            # 更新解码器输入
            decoder_input = torch.tensor([[pred_id]], dtype=torch.long, device=device)

        return decoded_ids, attention_weights_list


def beam_search_decode(
    model: Union[PGCTModel, PGCoverageSeq2Seq],
    src: torch.Tensor,
    src_lens: Optional[torch.Tensor] = None,
    src_oov_map: Optional[torch.Tensor] = None,
    max_len: int = 100,
    sos_idx: int = 2,
    eos_idx: int = 3,
    beam_size: int = 5,
    device: Optional[torch.device] = None
) -> List[Tuple[List[int], float]]:
    """Beam Search 解码（同时适配两种模型）"""
    model.eval()
    if device is None:
        device = src.device
    assert src.size(0) == 1, "Beam Search 仅支持 batch_size=1"

    with torch.no_grad():
        # 1. 编码（适配两种模型）
        if isinstance(model, PGCTModel):
            encoder_outputs, _ = model.encoder(src, src_lens)
            src_len = encoder_outputs.size(1)
            src_mask = model.encoder.generate_src_mask(src)
            # 扩展到 beam_size 维度
            encoder_outputs = encoder_outputs.expand(beam_size, -1, -1)
            src_expanded = src.expand(beam_size, -1)
            src_mask_expanded = src_mask.expand(beam_size, -1)
            src_oov_expanded = src_oov_map.expand(beam_size, -1) if src_oov_map is not None else None
            hidden, cell = None, None  # Transformer 无 LSTM 状态
        else:
            encoder_outputs, (hidden, cell) = model.encoder(src, src_lens)
            hidden, cell = model._bridge_states(hidden, cell)
            # 扩展到 beam_size 维度
            encoder_outputs = encoder_outputs.expand(beam_size, -1, -1)
            hidden = hidden.expand(-1, beam_size, -1).contiguous()
            cell = cell.expand(-1, beam_size, -1).contiguous()
            src_expanded = src.expand(beam_size, -1)
            src_oov_expanded = src_oov_map.expand(beam_size, -1) if src_oov_map is not None else None
            src_mask_expanded = None

        # 2. 初始化 beam（包含序列、分数、覆盖向量/LSTM状态）
        initial_coverage = torch.zeros(1, src.size(1), device=device) if isinstance(model, PGCTModel) else None
        beams = [([sos_idx], 0.0, initial_coverage, hidden, cell)] if isinstance(model, PGCoverageSeq2Seq) \
            else [([sos_idx], 0.0, initial_coverage)]
        completed = []

        # 3. 逐步扩展 beam
        for _ in range(max_len):
            candidates = []
            for beam in beams:
                if isinstance(model, PGCTModel):
                    seq, score, cov = beam
                else:
                    seq, score, cov, h, c = beam

                # 序列已结束则加入完成列表
                if seq[-1] == eos_idx:
                    completed.append((seq, score))
                    continue

                # 准备解码器输入
                decoder_input = torch.tensor([[seq[-1]]], dtype=torch.long, device=device)

                # 解码下一步（适配两种模型）
                if isinstance(model, PGCTModel):
                    final_dist, attn_weights, _ = model.decoder.forward_step(
                        tgt_step=decoder_input,
                        encoder_outputs=encoder_outputs[0:1],  # 取当前 beam 的编码器输出
                        src_mask=src_mask_expanded[0:1],
                        coverage_vector=cov,
                        src_ids=src_expanded[0:1],
                        src_oov_map=src_oov_expanded[0:1] if src_oov_expanded is not None else None
                    )
                    new_cov = cov + attn_weights  # 更新覆盖向量
                    h, c = None, None  # 无需 LSTM 状态
                else:
                    final_dist, new_h, new_c, attn_weights = model.decoder.forward_step(
                        y_prev=decoder_input,
                        hidden=h,
                        cell=c,
                        encoder_outputs=encoder_outputs[0:1],
                        src_lens=src_lens,
                        src_ids=src_expanded[0:1],
                        src_oov_map=src_oov_expanded[0:1] if src_oov_expanded is not None else None,
                        coverage_vector=cov
                    )
                    new_cov = cov + attn_weights if cov is not None else None
                    h, c = new_h, new_c  # 更新 LSTM 状态

                # 筛选 top-k 候选
                log_probs = F.log_softmax(final_dist, dim=1).squeeze()
                topk_probs, topk_ids = torch.topk(log_probs, beam_size)

                # 扩展候选序列
                for prob, token_id in zip(topk_probs, topk_ids):
                    token_id = token_id.item()
                    new_seq = seq + [token_id]
                    new_score = score + prob.item()
                    if isinstance(model, PGCTModel):
                        candidates.append((new_seq, new_score, new_cov))
                    else:
                        candidates.append((new_seq, new_score, new_cov, new_h, new_c))

            # 筛选下一轮 beam
            if not candidates:
                break
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_size]

            # 所有 beam 结束则停止
            if all(seq[-1] == eos_idx for seq, *_ in beams):
                break

        # 处理未完成的 beam
        for beam in beams:
            seq = beam[0]
            score = beam[1]
            if seq[-1] != eos_idx:
                completed.append((seq, score))

        # 按分数排序并返回
        completed.sort(key=lambda x: x[1], reverse=True)
        return completed[:beam_size]
