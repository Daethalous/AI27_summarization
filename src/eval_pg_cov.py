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


@torch.no_grad()
def generate_summaries(
    model: PGCoverageSeq2Seq,
    dataloader,
    vocab: Vocab,
    device: torch.device,
    max_tgt_len: int = 100,
    decode_strategy: str = 'greedy',
    beam_size: int = 5
) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    """批量生成摘要（适配 PG-Coverage 模型，返回 (预测, 参考, 源文本) token 列表"""
    model.eval()
    all_predictions = []
    all_references = []
    all_sources = []

    pbar = tqdm(dataloader, desc=f"生成摘要（策略：{decode_strategy}）")
    for batch in pbar:
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        src_len = batch.get('src_lens', None)
        src_oov_map = batch.get('src_oov_map', None)
        oov_lists = batch.get("oov_list", None)
        src_tokens_list = batch.get("src_tokens", None)  # 若 dataloader 有存原始 token

        if src_len is not None:
            src_len = src_len.to(device)
        if src_oov_map is not None:
            src_oov_map = src_oov_map.to(device)

        batch_size = src.size(0)

        for i in range(batch_size):
            src_i = src[i:i+1]
            src_len_i = src_len[i:i+1] if src_len is not None else None
            src_oov_map_i = src_oov_map[i:i+1] if src_oov_map is not None else None
            tgt_i = tgt[i]
            oov_list_i = oov_lists[i] if oov_lists is not None else None
            src_tokens_i = src_tokens_list[i] if src_tokens_list is not None else None

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
                pred_ids = pred_ids[0].tolist()
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
                pred_ids = pred_ids[0].tolist()

            # 预测摘要, 参考摘要，源文本转为token并跳过特殊符号
            def decode_with_oov(ids, vocab, oov_list):
                out = []
                for pid in ids:
                    if pid == vocab.eos_idx:
                        break
                    if pid in (vocab.pad_idx, vocab.sos_idx):
                        continue
                    if pid < len(vocab):
                        out.append(vocab.idx2word.get(pid, Vocab.UNK_TOKEN))
                    else:
                        ext_idx = pid - len(vocab)
                        if oov_list and 0 <= ext_idx < len(oov_list):
                            out.append(oov_list[ext_idx])
                        else:
                            out.append(Vocab.UNK_TOKEN)
                return out

            pred_tokens = decode_with_oov(pred_ids, vocab, oov_list_i)
            ref_tokens = decode_with_oov(tgt_i.cpu().tolist(), vocab, None)
            if src_tokens_i is not None and isinstance(src_tokens_i, list):
                src_tokens = src_tokens_i
            else:
                # 或根据索引还原
                src_tokens = decode_with_oov(src_i.squeeze(0).cpu().tolist(), vocab, oov_list_i)

            all_predictions.append(pred_tokens)
            all_references.append(ref_tokens)
            all_sources.append(src_tokens)

    return all_predictions, all_references, all_sources


def main():
    parser = argparse.ArgumentParser(description="PG-Coverage 评估脚本 (支持 OOV/coverage)")
    parser.add_argument('--config', type=str, help='YAML 配置路径（可选）')
    parser.add_argument('--data_dir', type=str, default='./data/raw', help='原始数据目录')
    parser.add_argument('--vocab_path', type=str, default='./data/processed/vocab.json', help='词表文件')
    parser.add_argument('--processed_dir', type=str, default='./data/processed', help='处理后数据目录')
    parser.add_argument('--split', type=str, default='val', help='评估数据划分 val/test')
    parser.add_argument('--checkpoint', type=str, required=True, help='PG-Coverage 检查点路径')
    parser.add_argument('--max_src_len', type=int, default=400)
    parser.add_argument('--max_tgt_len', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--decode_strategy', type=str, choices=['greedy', 'beam'], default='greedy')
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--save_path', type=str, default='./docs/pg_coverage_eval_results.json', help='评估预测输出文件')
    args = parser.parse_args()

    # YAML 配置合并
    config = {}
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    for arg_k, arg_v in vars(args).items():
        if arg_v is not None:
            config[arg_k] = arg_v

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 词表及数据集准备
    vocab = Vocab.load(config["vocab_path"])
    print(f"词表大小: {len(vocab)}")

    dataloader = get_dataloader(
        processed_dir=config["processed_dir"],
        batch_size=config["batch_size"],
        split=config["split"],
        shuffle=False,
        num_workers=0
    )
    print(f"评估集样本数: {len(dataloader.dataset)}")

    # 恢复模型
    # 检查点应存有训练config
    checkpoint = torch.load(config["checkpoint"], map_location=device)
    train_conf = checkpoint.get("config", {})
    model = PGCoverageSeq2Seq(
        vocab_size=len(vocab),
        embed_size=train_conf.get("embed_size", 256),
        hidden_size=train_conf.get("hidden_size", 256),
        num_layers=train_conf.get("num_layers", 1),
        dropout=train_conf.get("dropout", 0.1),
        pad_idx=vocab.pad_idx,
        cov_loss_weight=train_conf.get("coverage_loss_weight", 1.0)
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("模型已加载 ✔️")

    # 生成摘要
    predictions, references, sources = generate_summaries(
        model=model,
        dataloader=dataloader,
        vocab=vocab,
        device=device,
        max_tgt_len=config["max_tgt_len"],
        decode_strategy=config["decode_strategy"],
        beam_size=config.get("beam_size", 5),
    )

    # 保存输出样例
    eval_output = {
        "predictions": [" ".join(x) for x in predictions],
        "references": [" ".join(y) for y in references],
        "sources": [" ".join(z) for z in sources],
    }
    with open(config["save_path"], "w", encoding="utf-8") as f:
        json.dump(eval_output, f, ensure_ascii=False, indent=2)
    print(f"已保存预测结果到: {config['save_path']}")

    # 计算 ROUGE (字符串版本)
    preds_strs = eval_output["predictions"]
    refs_strs = eval_output["references"]
    rouge_scores = compute_rouge(preds_strs, refs_strs)
    print_metrics(rouge_scores)


if __name__ == "__main__":
    main()
