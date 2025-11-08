"""PG-Coverage 模型推理脚本：从文本文件生成摘要（支持 OOV 和覆盖机制）."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import torch
import yaml

# 添加 src 到 path（确保能导入模型和工具类）
sys.path.insert(0, str(Path(__file__).parent))

# 导入 PG-Coverage 模型（替换 baseline 模型）
from models.pointer_generator_coverage.pg_coverage_model import PGCoverageSeq2Seq
# 导入数据预处理和词表工具
from datamodules.cnndm import prepare_datasets
from utils.vocab import Vocab  # 确保 Vocab 类有 encode/decode 方法

try:
    import nltk
    from nltk.tokenize import word_tokenize
    nltk.download('punkt', quiet=True)  # 自动下载分词所需资源
except ImportError as exc:
    raise ImportError("请先安装 nltk (pip install nltk)") from exc


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    """文本分词（与训练时预处理逻辑一致）"""
    if lowercase:
        text = text.lower()
    return word_tokenize(text)


def load_pg_coverage_model(
    checkpoint_path: str, 
    vocab_size: int, 
    pad_idx: int, 
    device: torch.device
) -> PGCoverageSeq2Seq:
    """加载 PG-Coverage 模型（适配模型初始化参数）"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})  # 从 checkpoint 读取训练时的配置
    
    # 初始化 PG-Coverage 模型（匹配 __init__ 方法参数）
    model = PGCoverageSeq2Seq(
        vocab_size=vocab_size,
        embed_size=config.get('embed_size', 256),  # 默认与训练一致
        hidden_size=config.get('hidden_size', 256),
        num_layers=config.get('num_layers', 1),
        dropout=config.get('dropout', 0.1),
        pad_idx=pad_idx,
        cov_loss_weight=config.get('coverage_loss_weight', 1.0)  # 推理时不影响，仅为初始化
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 推理模式（关闭 dropout）
    return model


def summarize_text(
    model: PGCoverageSeq2Seq,
    vocab: Vocab,
    article: str,
    device: torch.device,
    max_src_len: int = 400,
    max_tgt_len: int = 100,
    decode_strategy: str = 'greedy',
    beam_size: int = 5
) -> List[str]:
    """生成摘要（调用 PG-Coverage 模型自带的解码方法，适配覆盖向量）"""
    # 1. 文本预处理（与训练一致：分词、截断、编码）
    tokens = tokenize(article)[:max_src_len]  # 截断过长的源文本
    src_indices = vocab.encode(tokens, max_len=max_src_len)  # 转为词表索引（补零/截断）
    src_tensor = torch.LongTensor([src_indices]).to(device)  # [1, max_src_len]（batch_size=1）
    src_len = torch.LongTensor([min(len(tokens), max_src_len)]).to(device)  # 实际文本长度（用于掩码）
    
    # 2. 解码（调用模型自带的 generate/beam_search 方法，自动处理覆盖向量）
    if decode_strategy == 'beam':
        # Beam Search 解码（模型自带，支持覆盖机制）
        pred_ids, _ = model.beam_search(
            src=src_tensor,
            src_lens=src_len,
            src_oov_map=None,  # 推理时无 OOV 映射（若输入有 OOV，需补充，此处简化）
            beam_size=beam_size,
            max_length=max_tgt_len,
            sos_idx=vocab.sos_idx,
            eos_idx=vocab.eos_idx,
            device=device
        )
        pred_ids = pred_ids[0].tolist()  # 取最佳序列（batch_size=1）
    else:
        # Greedy 解码（模型自带，自动更新覆盖向量）
        pred_ids, _ = model.generate(
            src=src_tensor,
            src_lens=src_len,
            src_oov_map=None,
            max_length=max_tgt_len,
            sos_idx=vocab.sos_idx,
            eos_idx=vocab.eos_idx,
            device=device
        )
        pred_ids = pred_ids[0].tolist()  # 取第一个样本的序列
    
    # 3. 解码为文本（跳过特殊符号：<PAD>、<SOS>、<EOS>）
    return vocab.decode(pred_ids, skip_special=True)


def collect_inputs(input_path: str) -> List[Path]:
    """收集输入文本文件（支持单个文件或目录下的所有 .txt 文件）"""
    path = Path(input_path)
    if path.is_dir():
        return sorted([p for p in path.glob('*.txt') if p.is_file()])
    if path.is_file() and path.suffix == '.txt':
        return [path]
    raise FileNotFoundError(f"未找到有效输入：{input_path}（仅支持 .txt 文件或包含 .txt 的目录）")


def main(args: argparse.Namespace) -> None:
    # 1. 设备初始化（优先 GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ 使用设备: {device}")

    # 2. 加载配置（支持 YAML 配置文件或命令行参数）
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # 3. 关键参数配置（优先级：命令行参数 > 配置文件 > 默认值）
    raw_data_dir = args.data_dir or config.get('data_dir', './data/raw')
    vocab_path = args.vocab_path or config.get('vocab_path', './data/processed/vocab.json')
    processed_dir = config.get('processed_dir', os.path.dirname(vocab_path))
    checkpoint_path = args.checkpoint or config.get('checkpoint_path', './checkpoints/best_model.pt')
    max_src_len = args.max_src_len or config.get('max_src_len', 400)
    max_tgt_len = args.max_tgt_len or config.get('max_tgt_len', 100)
    decode_strategy = args.decode_strategy
    beam_size = args.beam_size

    # 4. 加载词表（若未预处理，自动生成；若已存在，直接加载）
    print(f"📥 加载词表: {vocab_path}")
    vocab = prepare_datasets(
        raw_dir=raw_data_dir,
        processed_dir=processed_dir,
        vocab_path=vocab_path,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        max_vocab_size=config.get('max_vocab_size', 50000),
        min_freq=config.get('min_freq', 5)
    )
    print(f"✅ 词表加载完成（大小：{len(vocab)}）")

    # 5. 加载 PG-Coverage 模型
    print(f"📥 加载模型: {checkpoint_path}")
    model = load_pg_coverage_model(
        checkpoint_path=checkpoint_path,
        vocab_size=len(vocab),
        pad_idx=vocab.pad_idx,
        device=device
    )
    print(f"✅ 模型加载完成（推理模式）")

    # 6. 收集输入文本
    print(f"📥 收集输入文件: {args.input}")
    input_files = collect_inputs(args.input)
    if not input_files:
        raise FileNotFoundError(f"未找到任何 .txt 文件：{args.input}")
    print(f"✅ 共收集到 {len(input_files)} 个输入文件")

    # 7. 创建输出目录（确保输出路径存在）
    output_path = Path(args.output)
    os.makedirs(output_path.parent, exist_ok=True)

    # 8. 批量推理并保存结果
    results = []
    print(f"\n🚀 开始生成摘要（策略：{decode_strategy}，最大长度：{max_tgt_len}）")
    for idx, filepath in enumerate(input_files, start=1):
        # 读取输入文本
        with filepath.open('r', encoding='utf-8') as f:
            article = f.read().strip()
        if not article:
            print(f"⚠️ 跳过空文件：{filepath.name}")
            continue

        # 生成摘要（无梯度计算，加速）
        with torch.no_grad():
            summary_tokens = summarize_text(
                model=model,
                vocab=vocab,
                article=article,
                device=device,
                max_src_len=max_src_len,
                max_tgt_len=max_tgt_len,
                decode_strategy=decode_strategy,
                beam_size=beam_size
            )
        summary = ' '.join(summary_tokens)  # 转为字符串

        # 保存结果
        results.append({
            'id': idx,
            'file': str(filepath),
            'article_length': len(article),
            'summary': summary
        })

        # 打印进度
        print(f"\n--- 样本 {idx}（{filepath.name}）---")
        print(f"源文本（前100字符）: {article[:100]}...")
        print(f"生成摘要: {summary}")

    # 9. 保存所有结果到文件
    with output_path.open('w', encoding='utf-8') as f:
        for item in results:
            f.write(f"=== SAMPLE {item['id']} ===\n")
            f.write(f"File: {item['file']}\n")
            f.write(f"Article Length: {item['article_length']} characters\n")
            f.write(f"Summary: {item['summary']}\n\n")

    print(f"\n🎉 推理完成！结果已保存到: {output_path}")
    print(f"📊 统计：共处理 {len(results)} 个有效文件，生成 {len(results)} 条摘要")


if __name__ == '__main__':
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='PG-Coverage 模型推理脚本（支持 OOV 和覆盖机制）')
    # 配置与模型
    parser.add_argument('--config', type=str, help='YAML 配置文件路径（可选）')
    parser.add_argument('--checkpoint', type=str, help='PG-Coverage 模型检查点路径（必填或在配置文件中指定）')
    parser.add_argument('--vocab_path', type=str, help='词表路径（必填或在配置文件中指定）')
    parser.add_argument('--data_dir', type=str, help='原始数据目录（用于生成词表，可选）')
    # 输入输出
    parser.add_argument('--input', type=str, required=True, help='输入：.txt 文件路径或包含 .txt 的目录（必填）')
    parser.add_argument('--output', type=str, default='../docs/pg_coverage_infer_results.txt', help='输出结果文件路径（默认：../docs/pg_coverage_infer_results.txt）')
    # 推理参数
    parser.add_argument('--max_src_len', type=int, help='源文本最大长度（默认：400）')
    parser.add_argument('--max_tgt_len', type=int, help='摘要最大长度（默认：100）')
    parser.add_argument('--decode_strategy', type=str, default='greedy', choices=['greedy', 'beam'], help='解码策略（默认：greedy）')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam Search 大小（仅 beam 策略需要，默认：5）')

    args = parser.parse_args()
    main(args)
