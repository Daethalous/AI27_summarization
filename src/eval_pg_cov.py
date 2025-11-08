"""PG-Coverage 模型评估脚本：计算 ROUGE 指标（支持 OOV 和覆盖机制）."""
from __future__ import annotations

import argparse
import json
import os
import sys
import yaml
from pathlib import Path
from typing import List, Tuple

import torch
from tqdm import tqdm

# 添加 src 到 path（确保能导入模型和工具类）
sys.path.insert(0, str(Path(__file__).parent))

# 导入 PG-Coverage 模型（替换 baseline 模型）
from models.pointer_generator_coverage.pg_coverage_model import PGCoverageSeq2Seq
# 导入数据、词表和指标工具
from datamodules.cnndm import prepare_datasets, get_dataloader
from utils.vocab import Vocab
from utils.metrics import compute_rouge, print_metrics  # 确保有 ROUGE 计算函数


@torch.no_grad()  # 关闭梯度计算，加速评估
def generate_summaries(
    model: PGCoverageSeq2Seq,
    dataloader,
    vocab: Vocab,
    device: torch.device,
    max_tgt_len: int = 100,
    decode_strategy: str = 'greedy',
    beam_size: int = 5
) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    """批量生成摘要（适配 PG-Coverage 模型，返回预测/参考/源文本的 token 列表）"""
    model.eval()  # 推理模式
    all_predictions = []  # 预测摘要的 token 列表（批量）
    all_references = []   # 参考摘要的 token 列表（批量）
    all_sources = []      # 源文本的 token 列表（批量）

    # 批量处理数据
    pbar = tqdm(dataloader, desc=f"生成摘要（策略：{decode_strategy}）")
    for batch in pbar:
        # 提取 batch 数据（适配 dataloader 输出格式）
        src = batch['src'].to(device)  # [batch_size, max_src_len]
        tgt = batch['tgt'].to(device)  # [batch_size, max_tgt_len]
        src_len = batch.get('src_lens', None)  # [batch_size]（实际源文本长度）
        src_oov_map = batch.get('src_oov_map', None)  # [batch_size, max_src_len]（OOV 映射，可选）
        batch_size = src.size(0)

        # 逐个样本解码（适配 PG-Coverage 模型的单样本 Beam Search 限制）
        for i in range(batch_size):
            # 单样本数据（batch_size=1）
            src_i = src[i:i+1]  # [1, max_src_len]
            src_len_i = src_len[i:i+1] if src_len is not None else None
            src_oov_map_i = src_oov_map[i:i+1] if src_oov_map is not None else None
            tgt_i = tgt[i]  # [max_tgt_len]（参考摘要）

            # 1. 解码（调用模型自带方法，自动处理覆盖向量）
            if decode_strategy == 'beam':
                pred_ids, _ = model.beam_search(
                    src=src_i,
                    src_lens=src_len_i,
                    src_oov_map=src_oov_map_i,
                    beam_size=beam_size,
                    max_length=max_tgt_len,
                    sos_idx=vocab.sos_idx,
                    eos_idx=vocab.eos_idx,
                    device=device
                )
                pred_ids = pred_ids[0].tolist()  # 取最佳序列
            else:
                pred_ids, _ = model.generate(
                    src=src_i,
                    src_lens=src_len_i,
                    src_oov_map=src_oov_map_i,
                    max_length=max_tgt_len,
                    sos_idx=vocab.sos_idx,
                    eos_idx=vocab.eos_idx,
                    device=device
                )
                pred_ids = pred_ids[0].tolist()  # 取单样本序列

            # 2. 转为 token 列表（跳过特殊符号）
            pred_tokens = vocab.decode(pred_ids, skip_special=True)  # 预测摘要
            ref_tokens = vocab.decode(tgt_i.cpu().tolist(), skip_s
