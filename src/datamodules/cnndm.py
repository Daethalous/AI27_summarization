"""
CNN/DailyMail 数据处理与数据集模块（Baseline 版本）
负责：
1. 使用 NLTK 对 raw 文本进行分词与截断；
2. 构建词表并将样本转换为索引形式；
3. 将结果缓存为 train/val/test 的 pickle 文件；
4. 提供基于缓存数据的 Dataset/DataLoader。
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nltk
from nltk.tokenize import word_tokenize
import torch
from torch.utils.data import DataLoader, Dataset

from utils.vocab import Vocab

# 子目录与文件映射
SPLIT_DIR_MAP = {
    'train': 'train',
    'val': 'validation',
    'test': 'test'
}

PKL_FILENAMES = {
    'train': 'train.pkl',
    'val': 'val.pkl',
    'test': 'test.pkl'
}


def _ensure_nltk_resource(resource: str) -> None:
    """确保指定的 NLTK 资源已安装。"""
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:  # pragma: no cover - 仅在缺资源时执行
        nltk.download(resource)


def _parse_raw_story(filepath: Path) -> Tuple[str, str]:
    """解析 raw 文本文件，返回文章与摘要的原始字符串。"""
    article_lines: List[str] = []
    summary_lines: List[str] = []
    current_section: Optional[str] = None

    with filepath.open('r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if line == '=== ARTICLE ===':
                current_section = 'article'
                continue
            if line == '=== SUMMARY ===':
                current_section = 'summary'
                continue
            if not line:
                continue

            if current_section == 'article':
                article_lines.append(line)
            elif current_section == 'summary':
                summary_lines.append(line)

    return ' '.join(article_lines), ' '.join(summary_lines)


def _tokenize(text: str, lowercase: bool = True) -> List[str]:
    """使用 NLTK 分词，并可选地转为小写。"""
    if lowercase:
        text = text.lower()
    return word_tokenize(text)


def _encode_sample(
    vocab: Vocab,
    article_tokens: List[str],
    summary_tokens: List[str],
    max_src_len: int,
    max_tgt_len: int
) -> Dict[str, List[int]]:
    """将分词后的样本编码为索引，并返回必要字段。"""
    src_tokens = article_tokens[:max_src_len]
    tgt_tokens = [Vocab.SOS_TOKEN] + summary_tokens[: max_tgt_len - 2] + [Vocab.EOS_TOKEN]

    src_len = len(src_tokens)
    tgt_len = len(tgt_tokens)

    src_indices = vocab.encode(src_tokens, max_len=max_src_len)
    tgt_indices = vocab.encode(tgt_tokens, max_len=max_tgt_len)

    return {
        'src': src_indices,
        'tgt': tgt_indices,
        'src_len': src_len,
        'tgt_len': tgt_len
    }


def prepare_datasets(
    raw_dir: str,
    processed_dir: str,
    vocab_path: str,
    max_src_len: int = 512,
    max_tgt_len: int = 512,
    max_vocab_size: int = 50000,
    min_freq: int = 5,
    limit_per_split: Optional[int] = None,
    progress_step: int = 10000
) -> Vocab:
    """处理 raw 数据，生成 pickle 文件与词表。

    若缓存已存在，则直接加载词表。"""
    for resource in ('punkt', 'punkt_tab'):
        _ensure_nltk_resource(resource)

    raw_dir_path = Path(raw_dir)
    processed_dir_path = Path(processed_dir)
    processed_dir_path.mkdir(parents=True, exist_ok=True)

    processed_paths = {split: processed_dir_path / fname for split, fname in PKL_FILENAMES.items()}
    vocab_file = Path(vocab_path)

    cache_exists = vocab_file.exists() and all(path.exists() for path in processed_paths.values())
    if cache_exists:
        return Vocab.load(vocab_path)

    # 收集训练集 token 用于构建词表
    training_token_corpus: List[List[str]] = []
    split_samples: Dict[str, List[Dict[str, List[str]]]] = {split: [] for split in SPLIT_DIR_MAP}

    for split, split_dir_name in SPLIT_DIR_MAP.items():
        split_raw_dir = raw_dir_path / split_dir_name
        if not split_raw_dir.is_dir():
            raise ValueError(f"未找到原始数据目录: {split_raw_dir}")

        filepaths = sorted(split_raw_dir.glob('*.txt'))
        if limit_per_split is not None:
            filepaths = filepaths[:limit_per_split]
        if not filepaths:
            raise ValueError(f"目录 {split_raw_dir} 中没有 .txt 文件")

        print(f"[{split}] 发现 {len(filepaths)} 个样本，开始分词与编码...")
        encoded_entries: List[Dict[str, List[str]]] = []

        log_interval = max(1, progress_step)

        for idx, filepath in enumerate(filepaths, start=1):
            article_text, summary_text = _parse_raw_story(filepath)
            article_tokens = _tokenize(article_text)
            summary_tokens = _tokenize(summary_text)

            if split == 'train':
                training_token_corpus.append(article_tokens[:max_src_len])
                training_token_corpus.append(summary_tokens[: max_tgt_len - 2])

            encoded_entries.append({
                'article_tokens': article_tokens,
                'summary_tokens': summary_tokens
            })

            if idx % log_interval == 0 or idx == len(filepaths):
                print(f"  [{split}] 已处理 {idx}/{len(filepaths)} 个样本")

        split_samples[split] = encoded_entries
        print(f"[{split}] 分词完成，共收集 {len(encoded_entries)} 条样本")

    # 构建词表
    print("开始构建词表...")
    vocab = Vocab(max_vocab_size=max_vocab_size, min_freq=min_freq)
    vocab.build_vocab(training_token_corpus)
    print("词表构建完成，开始写入缓存文件...")
    vocab.save(vocab_path)

    # 编码并写入 pickle
    for split, entries in split_samples.items():
        serialized: List[Dict[str, List[int]]] = []
        for entry in entries:
            encoded_entry = _encode_sample(
                vocab,
                entry['article_tokens'],
                entry['summary_tokens'],
                max_src_len,
                max_tgt_len
            )
            serialized.append(encoded_entry)

        with processed_paths[split].open('wb') as f:
            pickle.dump(serialized, f)
        print(f"[{split}] 已写入 {processed_paths[split]}")

    return vocab


class CNNDMDataset(Dataset):
    """基于 pickle 缓存的 CNN/DailyMail 数据集。"""

    def __init__(self, processed_dir: str, split: str):
        processed_path = Path(processed_dir) / PKL_FILENAMES[split]
        if not processed_path.exists():
            raise ValueError(f"未找到预处理后的数据文件: {processed_path}")

        with processed_path.open('rb') as f:
            self.samples: List[Dict[str, List[int]]] = pickle.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            'src': torch.LongTensor(sample['src']),
            'tgt': torch.LongTensor(sample['tgt']),
            'src_len': torch.LongTensor([sample['src_len']]),
            'tgt_len': torch.LongTensor([sample['tgt_len']])
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    src = torch.stack([item['src'] for item in batch])
    tgt = torch.stack([item['tgt'] for item in batch])
    src_len = torch.cat([item['src_len'] for item in batch])
    tgt_len = torch.cat([item['tgt_len'] for item in batch])

    return {
        'src': src,
        'tgt': tgt,
        'src_len': src_len,
        'tgt_len': tgt_len
    }


def get_dataloader(
    processed_dir: str,
    batch_size: int = 32,
    split: str = 'train',
    num_workers: int = 0,
    shuffle: Optional[bool] = None
) -> DataLoader:
    """基于预处理数据创建 DataLoader。"""
    if split not in PKL_FILENAMES:
        raise ValueError(f"Unknown split: {split}")

    dataset = CNNDMDataset(processed_dir, split)
    if shuffle is None:
        shuffle = (split == 'train')

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


def build_vocab_from_data(
    data_dir: str,
    max_vocab_size: int = 50000,
    min_freq: int = 5,
    max_src_len: int = 512,
    max_tgt_len: int = 512,
    processed_dir: Optional[str] = None,
    vocab_path: Optional[str] = None,
    limit_per_split: Optional[int] = None,
    progress_step: int = 10000
) -> Vocab:
    """兼容旧接口，调用 `prepare_datasets` 并返回 Vocab。"""
    if processed_dir is None or vocab_path is None:
        raise ValueError("需要提供 processed_dir 与 vocab_path 以构建词表")

    return prepare_datasets(
        raw_dir=data_dir,
        processed_dir=processed_dir,
        vocab_path=vocab_path,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        max_vocab_size=max_vocab_size,
        min_freq=min_freq,
        limit_per_split=limit_per_split,
        progress_step=progress_step
    )
