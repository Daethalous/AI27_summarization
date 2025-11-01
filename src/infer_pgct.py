"""PGCT 推理脚本：从文本文件生成摘要（适配 Transformer+Pointer-Generator+Coverage 模型）"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import yaml

# 添加 src 到 path
sys.path.insert(0, str(Path(__file__).parent))

from datamodules.cnndm import prepare_datasets
from models.pointer_generator_coverage.pgct_model import PGCTModel  # 导入PGCT模型
from utils.decoding import beam_search_decode, greedy_decode
from utils.vocab import Vocab

try:
    import nltk
    from nltk.tokenize import word_tokenize
except ImportError as exc:  # pragma: no cover
    raise ImportError("请先安装 nltk (pip install nltk)") from exc


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    """文本分词（保持与训练数据一致的预处理）"""
    if lowercase:
        text = text.lower()
    return word_tokenize(text)


def process_oov(tokens: List[str], vocab: Vocab) -> Tuple[List[int], Dict[int, str], List[int]]:
    """
    处理OOV词汇，生成带OOV映射的源序列
    
    Args:
        tokens: 分词后的文本
        vocab: 词表
        
    Returns:
        src_indices: 源序列索引（含OOV标记）
        oov_dict: OOV词汇映射表（索引->词）
        src_oov_map: 源序列OOV映射（原始索引->OOV索引）
    """
    src_indices = []
    oov_dict = {}  # 存储OOV词：{oov_idx: token}
    src_oov_map = []  # 记录每个位置是否为OOV，以及对应的OOV索引
    
    for token in tokens:
        if token in vocab.word2idx:
            # 常规词
            src_indices.append(vocab.word2idx[token])
            src_oov_map.append(0)  # 0表示非OOV
        else:
            # OOV词：分配临时索引（从vocab_size开始）
            if token not in oov_dict.values():
                oov_idx = len(oov_dict) + 1  # 1-based索引
                oov_dict[oov_idx] = token
            else:
                oov_idx = [k for k, v in oov_dict.items() if v == token][0]
            src_indices.append(vocab.unk_idx)  # 用UNK标记占位
            src_oov_map.append(oov_idx)  # 记录OOV索引
    
    return src_indices, oov_dict, src_oov_map


def load_model(
    checkpoint_path: str, 
    vocab_size: int, 
    pad_idx: int, 
    device: torch.device
) -> PGCTModel:
    """加载PGCT模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    model = PGCTModel(
        vocab_size=vocab_size,
        embed_size=config.get('embed_size', 256),
        hidden_size=config.get('hidden_size', 256),
        num_encoder_layers=config.get('num_encoder_layers', 3),
        num_decoder_layers=config.get('num_decoder_layers', 3),
        nhead=config.get('nhead', 8),  # Transformer多头注意力头数
        dropout=config.get('dropout', 0.1),
        pad_idx=pad_idx,
        cov_loss_weight=config.get('cov_loss_weight', 1.0),
        max_src_len=config.get('max_src_len', 400),
        max_tgt_len=config.get('max_tgt_len', 100)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def summarize_text(
    model: PGCTModel,
    vocab: Vocab,
    article: str,
    device: torch.device,
    max_src_len: int,
    max_tgt_len: int,
    decode_strategy: str,
    beam_size: int
) -> List[str]:
    """生成摘要（支持OOV处理）"""
    # 1. 文本预处理与分词
    tokens = tokenize(article)[:max_src_len]  # 截断过长文本
    src_len = len(tokens)
    
    # 2. 处理OOV词汇
    src_indices, oov_dict, src_oov_map = process_oov(tokens, vocab)
    
    # 3. 补齐序列长度
    if len(src_indices) < max_src_len:
        pad_length = max_src_len - len(src_indices)
        src_indices += [vocab.pad_idx] * pad_length
        src_oov_map += [0] * pad_length  # 补齐部分OOV映射为0
    
    # 4. 转换为Tensor
    src_tensor = torch.LongTensor([src_indices]).to(device)  # [1, max_src_len]
    src_len_tensor = torch.LongTensor([src_len]).to(device)   # [1]
    src_oov_map_tensor = torch.LongTensor([src_oov_map]).to(device)  # [1, max_src_len]
    
    # 5. 解码生成摘要
    with torch.no_grad():
        if decode_strategy == 'beam':
            beams = beam_search_decode(
                model,
                src=src_tensor,
                src_lens=src_len_tensor,
                src_oov_map=src_oov_map_tensor,  # 传入OOV映射
                max_len=max_tgt_len,
                sos_idx=vocab.sos_idx,
                eos_idx=vocab.eos_idx,
                beam_size=beam_size,
                device=device
            )
            pred_ids = beams[0][0] if beams else []
        else:
            pred_ids, _ = greedy_decode(
                model,
                src=src_tensor,
                src_lens=src_len_tensor,
                src_oov_map=src_oov_map_tensor,  # 传入OOV映射
                max_len=max_tgt_len,
                sos_idx=vocab.sos_idx,
                eos_idx=vocab.eos_idx,
                device=device
            )
    
    # 6. 转换为文本（处理OOV）
    summary_tokens = []
    for idx in pred_ids:
        idx_val = idx if isinstance(idx, int) else idx.item()
        if idx_val < len(vocab):
            # 常规词
            token = vocab.idx2word.get(idx_val, vocab.unk_token)
        else:
            # OOV词：从oov_dict中查找
            oov_idx = idx_val - len(vocab)
            token = oov_dict.get(oov_idx, vocab.unk_token)
        
        # 过滤特殊符号
        if token not in [vocab.pad_token, vocab.sos_token, vocab.eos_token]:
            summary_tokens.append(token)
    
    return summary_tokens


def collect_inputs(input_path: str) -> List[Path]:
    """收集输入文件（支持目录或单个文件）"""
    path = Path(input_path)
    if path.is_dir():
        return sorted([p for p in path.glob('*.txt') if p.is_file()])
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"未找到输入路径: {input_path}")


def main(args: argparse.Namespace) -> None:
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载配置
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # 参数配置
    raw_data_dir = config.get('data_dir', './data/raw')
    vocab_path = config.get('vocab_path', './data/processed/vocab.json')
    processed_dir = config.get('processed_dir', os.path.dirname(vocab_path))
    checkpoint_path = args.checkpoint or config.get('checkpoint_path', './checkpoints/pgct_best_model.pt')

    max_src_len = config.get('max_src_len', 400)
    max_tgt_len = config.get('max_tgt_len', 100)
    decode_strategy = args.decode_strategy
    beam_size = args.beam_size

    # 准备词表
    print(f"加载词表: {vocab_path}")
    vocab = prepare_datasets(
        raw_dir=raw_data_dir,
        processed_dir=processed_dir,
        vocab_path=vocab_path,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        max_vocab_size=config.get('max_vocab_size', 50000),
        min_freq=config.get('min_freq', 5)
    )
    if not isinstance(vocab, Vocab):
        vocab = Vocab.load(vocab_path)  # 确保正确加载词表

    # 加载模型
    model = load_model(checkpoint_path, len(vocab), vocab.pad_idx, device)
    print(f"✓ 模型已加载: {checkpoint_path}")

    # 收集输入文件
    input_files = collect_inputs(args.input)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 生成摘要
    results = []
    for idx, filepath in enumerate(input_files, start=1):
        with filepath.open('r', encoding='utf-8') as f:
            article = f.read().strip()

        print(f"\n--- 处理文件 {idx}/{len(input_files)}: {filepath.name} ---")
        print(f"源文本长度: {len(article)} 字符")

        # 生成摘要
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

        results.append({
            'id': idx,
            'file': str(filepath),
            'summary': summary
        })

        # 打印生成结果
        print(f"生成摘要: {summary[:200]}...")  # 显示前200字符

    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(f"=== 样本 {item['id']} ===\n")
            f.write(f"文件路径: {item['file']}\n")
            f.write(f"生成摘要: {item['summary']}\n\n")

    print(f"\n✓ 所有结果已保存到: {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PGCT 模型推理脚本（Transformer+Pointer-Generator+Coverage）')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, help='模型检查点路径')
    parser.add_argument('--input', type=str, required=True, help='输入文本文件或目录（仅支持txt）')
    parser.add_argument('--output', type=str, default='../docs/pgct_samples.txt', help='输出结果文件')
    parser.add_argument('--decode_strategy', type=str, default='greedy', choices=['greedy', 'beam'], help='解码策略')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam Search 的 beam 大小')

    main(parser.parse_args())
