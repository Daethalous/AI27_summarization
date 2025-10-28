"""Baseline 推理脚本：从文本文件生成摘要."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import torch
import yaml

# 添加 src 到 path
sys.path.insert(0, str(Path(__file__).parent))

from datamodules.cnndm import prepare_datasets
from models.baseline.model import Seq2Seq
from utils.decoding import beam_search_decode, greedy_decode

try:
    import nltk
    from nltk.tokenize import word_tokenize
except ImportError as exc:  # pragma: no cover
    raise ImportError("请先安装 nltk (pip install nltk)") from exc


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    if lowercase:
        text = text.lower()
    return word_tokenize(text)


def load_model(checkpoint_path: str, vocab_size: int, pad_idx: int, device: torch.device) -> Seq2Seq:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    model = Seq2Seq(
        vocab_size=vocab_size,
        embed_size=config.get('embed_size', 256),
        hidden_size=config.get('hidden_size', 512),
        num_layers=config.get('num_layers', 1),
        dropout=config.get('dropout', 0.1),
        pad_idx=pad_idx
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def summarize_text(
    model: Seq2Seq,
    vocab,
    article: str,
    device: torch.device,
    max_src_len: int,
    max_tgt_len: int,
    decode_strategy: str,
    beam_size: int
) -> List[str]:
    tokens = tokenize(article)[:max_src_len]
    src_indices = vocab.encode(tokens, max_len=max_src_len)
    src_tensor = torch.LongTensor([src_indices]).to(device)
    src_len = torch.LongTensor([min(len(tokens), max_src_len)]).to(device)

    if decode_strategy == 'beam':
        beams = beam_search_decode(
            model,
            src_tensor,
            src_len,
            max_tgt_len,
            vocab.sos_idx,
            vocab.eos_idx,
            beam_size,
            device
        )
        pred_ids = beams[0][0] if beams else []
    else:
        pred_ids, _ = greedy_decode(
            model,
            src_tensor,
            src_len,
            max_tgt_len,
            vocab.sos_idx,
            vocab.eos_idx,
            device
        )

    return vocab.decode(pred_ids, skip_special=True)


def collect_inputs(input_path: str) -> List[Path]:
    path = Path(input_path)
    if path.is_dir():
        return sorted([p for p in path.glob('*.txt') if p.is_file()])
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"未找到输入路径: {input_path}")


def main(args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    raw_data_dir = config.get('data_dir', './data/raw')
    vocab_path = config.get('vocab_path', './data/processed/vocab.json')
    processed_dir = config.get('processed_dir', os.path.dirname(vocab_path))
    checkpoint_path = args.checkpoint or config.get('checkpoint_path', './checkpoints/best_model.pt')

    max_src_len = config.get('max_src_len', 512)
    max_tgt_len = config.get('max_tgt_len', 512)
    decode_strategy = args.decode_strategy
    beam_size = args.beam_size

    # 准备词表和数据缓存（若已存在则直接读取）
    vocab = prepare_datasets(
        raw_dir=raw_data_dir,
        processed_dir=processed_dir,
        vocab_path=vocab_path,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        max_vocab_size=config.get('max_vocab_size', 50000),
        min_freq=config.get('min_freq', 5)
    )

    model = load_model(checkpoint_path, len(vocab), vocab.pad_idx, device)
    print(f"✓ 模型已加载: {checkpoint_path}")

    input_files = collect_inputs(args.input)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results = []
    for idx, filepath in enumerate(input_files, start=1):
        with filepath.open('r', encoding='utf-8') as f:
            article = f.read().strip()

        summary_tokens = summarize_text(
            model,
            vocab,
            article,
            device,
            max_src_len,
            max_tgt_len,
            decode_strategy,
            beam_size
        )
        summary = ' '.join(summary_tokens)

        results.append(
            {
                'id': idx,
                'file': str(filepath),
                'summary': summary
            }
        )

        print(f"\n--- 样例 {idx} ({filepath.name}) ---")
        print(summary)

    with open(args.output, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(f"=== SAMPLE {item['id']} ===\n")
            f.write(f"File: {item['file']}\n")
            f.write(f"Summary: {item['summary']}\n\n")

    print(f"\n✓ 推理结果已保存到: {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Seq2Seq Baseline 推理脚本')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, help='模型检查点路径')
    parser.add_argument('--input', type=str, required=True, help='输入文本文件或目录')
    parser.add_argument('--output', type=str, default='../docs/baseline_samples.txt', help='输出结果文件')
    parser.add_argument('--decode_strategy', type=str, default='greedy', choices=['greedy', 'beam'], help='解码策略')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam Search 的 beam 大小')

    main(parser.parse_args())
