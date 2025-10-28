"""
解码策略：Greedy Decoding 和 Beam Search
"""
import torch
import torch.nn.functional as F
from typing import List, Tuple
from models.baseline.model import Seq2Seq


def greedy_decode(
    model: Seq2Seq,
    src: torch.Tensor,
    src_lens: torch.Tensor,
    max_len: int,
    sos_idx: int,
    eos_idx: int,
    device: torch.device
) -> Tuple[List[int], List[float]]:
    """贪婪解码
    
    Args:
        model: Seq2Seq模型
        src: [1, src_len] 源序列
        src_lens: [1] 源序列长度
        max_len: 最大解码长度
        sos_idx: <SOS> 索引
        eos_idx: <EOS> 索引
        device: 设备
        
    Returns:
        decoded_ids: 解码的token ID列表
        attention_weights: 注意力权重列表
    """
    model.eval()
    
    with torch.no_grad():
        # 编码
        encoder_outputs, (hidden, cell) = model.encoder(src, src_lens)
        hidden, cell = model._bridge_states(hidden, cell)
        
        # 初始化
        decoder_input = torch.LongTensor([[sos_idx]]).to(device)
        decoded_ids = []
        attention_weights_list = []
        
        # 逐步解码
        for _ in range(max_len):
            output, hidden, cell, attn_weights = model.decoder(
                decoder_input, hidden, cell, encoder_outputs, src_lens
            )
            
            # 选择概率最大的token
            pred_id = output.argmax(1).item()
            decoded_ids.append(pred_id)
            attention_weights_list.append(attn_weights.cpu().squeeze().tolist())
            
            # 遇到 <EOS> 停止
            if pred_id == eos_idx:
                break
            
            decoder_input = torch.LongTensor([[pred_id]]).to(device)
    
    return decoded_ids, attention_weights_list


def beam_search_decode(
    model: Seq2Seq,
    src: torch.Tensor,
    src_lens: torch.Tensor,
    max_len: int,
    sos_idx: int,
    eos_idx: int,
    beam_size: int,
    device: torch.device
) -> List[Tuple[List[int], float]]:
    """Beam Search 解码
    
    Args:
        model: Seq2Seq模型
        src: [1, src_len] 源序列
        src_lens: [1] 源序列长度
        max_len: 最大解码长度
        sos_idx: <SOS> 索引
        eos_idx: <EOS> 索引
        beam_size: beam大小
        device: 设备
        
    Returns:
        List of (decoded_ids, score) 按分数降序排列
    """
    model.eval()
    
    with torch.no_grad():
        # 编码
        encoder_outputs, (hidden, cell) = model.encoder(src, src_lens)
        hidden, cell = model._bridge_states(hidden, cell)
        
        # 初始化beam
        # 每个beam: (sequence, score, hidden, cell)
        beams = [([sos_idx], 0.0, hidden, cell)]
        completed = []
        
        for _ in range(max_len):
            candidates = []
            
            for seq, score, h, c in beams:
                # 如果已经结束，直接加入completed
                if seq[-1] == eos_idx:
                    completed.append((seq, score))
                    continue
                
                # 解码下一个token
                decoder_input = torch.LongTensor([[seq[-1]]]).to(device)
                output, new_h, new_c, _ = model.decoder(
                    decoder_input, h, c, encoder_outputs, src_lens
                )
                
                # 取top-k
                log_probs = F.log_softmax(output, dim=1)
                topk_probs, topk_ids = torch.topk(log_probs, beam_size)
                
                for prob, token_id in zip(topk_probs.squeeze(), topk_ids.squeeze()):
                    new_seq = seq + [token_id.item()]
                    new_score = score + prob.item()
                    candidates.append((new_seq, new_score, new_h, new_c))
            
            # 选择top beam_size个候选
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_size]
            
            # 如果所有beam都结束了，退出
            if all(seq[-1] == eos_idx for seq, _, _, _ in beams):
                break
        
        # 合并未完成的beam
        for seq, score, _, _ in beams:
            if seq[-1] != eos_idx:
                completed.append((seq, score))
        
        # 按分数排序
        completed.sort(key=lambda x: x[1], reverse=True)
        
        return completed[:beam_size]
