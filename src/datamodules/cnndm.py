"""
CNN/DailyMail 数据处理与数据集模块（Pointer-Generator 兼容版本）
负责：
1. 使用 NLTK 对 raw 文本进行分词与截断；
2. 构建词表并将样本转换为索引形式；
3. 新增：在 pickle 缓存中保存原始 token (用于 OOV 映射)；
4. 提供基于缓存数据的 Dataset/DataLoader，支持 Pointer-Generator 机制所需的 OOV 映射。
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nltk
from nltk.tokenize import word_tokenize
import torch
from torch.utils.data import DataLoader, Dataset

from utils.vocab import Vocab # 假设这里的 Vocab 类来自你的 utils/vocab.py (其中使用 word2idx)

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
) -> Dict[str, List]:
    """
    将分词后的样本编码为索引，并返回 PG 机制所需的原始 tokens。
    返回类型中的 List 包含 List[int] 和 List[str]
    """
    src_tokens = article_tokens[:max_src_len]
    tgt_tokens = [Vocab.SOS_TOKEN] + summary_tokens[: max_tgt_len - 2] + [Vocab.EOS_TOKEN]

    src_len = len(src_tokens)
    tgt_len = len(tgt_tokens)

    # 1. 编码为固定词表索引
    src_indices = vocab.encode(src_tokens, max_len=max_src_len)
    tgt_indices = vocab.encode(tgt_tokens, max_len=max_tgt_len)

    raw_summary_text = ' '.join(summary_tokens) #<---新增：保存原始参考摘要文本（用于 ROUGE 计算，不含 SOS/EOS）

    return {
        'src': src_indices,
        'tgt': tgt_indices,
        # 2. 存储原始 tokens 供 collate_fn_pg 使用
        'src_tokens': src_tokens, 
        'tgt_tokens': tgt_tokens,
        'src_len': src_len,
        'tgt_len': tgt_len
    }


def _is_pg_cache_compatible(processed_paths: Dict[str, Path]) -> bool:
    """稳健检查：验证缓存文件是否存在且包含 Pointer-Generator 所需的原始 tokens (src_tokens)。"""
    train_path = processed_paths.get('train')
    if not train_path or not train_path.exists():
        return False
    
    try:
        with train_path.open('rb') as f:
            samples = pickle.load(f)

        # 检查：1. 文件中是否有样本； 2. 样本是否为字典； 3. 样本中是否包含 PG 必需的 'src_tokens' 键；4.新增 'tgt_text' 检查
        if samples and isinstance(samples, list) and isinstance(samples[0], dict) and 'src_tokens' in samples[0] and 'tgt_text' in samples[0]:
            return True
        return False
    except Exception as e:
        # 捕获文件读取错误、pickle 反序列化错误等
        print(f"警告：无法加载或检查训练集缓存文件结构（{train_path}）：{e}")
        return False


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
    """处理 raw 数据，生成 pickle 文件与词表。"""
    for resource in ('punkt', 'punkt_tab'):
        _ensure_nltk_resource(resource)

    raw_dir_path = Path(raw_dir)
    processed_dir_path = Path(processed_dir)
    processed_dir_path.mkdir(parents=True, exist_ok=True)

    processed_paths = {split: processed_dir_path / fname for split, fname in PKL_FILENAMES.items()}
    vocab_file = Path(vocab_path)

    # 1. 检查所有文件是否存在
    all_files_exist = vocab_file.exists() and all(path.exists() for path in processed_paths.values())
    
    if all_files_exist:
        vocab: Optional[Vocab] = None
        # 2. 稳健检查词表是否能加载
        try:
            vocab = Vocab.load(vocab_path)
            print(f"词表已加载: {len(vocab)} 个词")
        except Exception as e:
            print(f"警告：词表文件加载失败 ({e})，将重新生成所有数据。")
            all_files_exist = False

        # 3. 稳健检查缓存结构是否兼容 PG
        if all_files_exist and vocab is not None:
            if _is_pg_cache_compatible(processed_paths):
                print("发现 PG 兼容的缓存文件和词表，跳过数据准备步骤。")
                return vocab
            else:
                print("警告：缓存文件存在，但结构与 PG 模型不兼容（缺少原始 token 数据）。将重新生成数据。")
                
    else:
        print("未发现完整的缓存文件（或词表），开始生成数据...")


    # --- 数据生成流程开始 ---

    # 收集训练集 token 用于构建词表
    training_token_corpus: List[List[str]] = []
    # 存储包含原始 tokens 的结构
    split_samples: Dict[str, List[Dict[str, List]]] = {split: [] for split in SPLIT_DIR_MAP}

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
        encoded_entries: List[Dict[str, List]] = []

        log_interval = max(1, progress_step)

        # 第一次迭代：收集 Tokens 用于构建词表
        for idx, filepath in enumerate(filepaths, start=1):
            article_text, summary_text = _parse_raw_story(filepath)
            article_tokens = _tokenize(article_text)
            summary_tokens = _tokenize(summary_text)

            if split == 'train':
                training_token_corpus.append(article_tokens[:max_src_len])
                training_token_corpus.append(summary_tokens[: max_tgt_len - 2])

            # 临时编码 (使用一个空的 vocab 对象，以便获取 tokens 和长度)
            encoded_entry = _encode_sample(
                vocab=Vocab(),
                article_tokens=article_tokens,
                summary_tokens=summary_tokens,
                max_src_len=max_src_len,
                max_tgt_len=max_tgt_len
            )
            encoded_entries.append(encoded_entry)

            if idx % log_interval == 0 or idx == len(filepaths):
                print(f"  [{split}] 已处理 {idx}/{len(filepaths)} 个样本")

        split_samples[split] = encoded_entries
        print(f"[{split}] 分词完成，共收集 {len(encoded_entries)} 条样本")

    # 构建词表 (现在使用完整的 corpus)
    print("开始构建词表...")
    vocab = Vocab(max_vocab_size=max_vocab_size, min_freq=min_freq)
    vocab.build_vocab(training_token_corpus)
    print("词表构建完成，开始写入词表文件...")
    vocab.save(vocab_path)
    
    # 第二次迭代：使用构建好的词表进行最终编码并写入 pickle
    print("使用最终词表重新编码固定索引并写入 pickle...")
    for split, entries in split_samples.items():
        serialized: List[Dict[str, List]] = []
        for entry in entries:
            # 重新编码固定词表索引
            src_indices = vocab.encode(entry['src_tokens'], max_len=max_src_len)
            tgt_indices = vocab.encode(entry['tgt_tokens'], max_len=max_tgt_len)
            
            # 存储最终数据结构（包含固定索引和原始 tokens）
            serialized.append({
                'src': src_indices,
                'tgt': tgt_indices,
                'src_tokens': entry['src_tokens'],
                'tgt_tokens': entry['tgt_tokens'],
                'src_len': entry['src_len'],
                'tgt_len': entry['tgt_len'],
                'tgt_text': ' '.join(entry['tgt_tokens'][1:-1])  # 去掉 <SOS> 和 <EOS>，保存原始参考摘要文本
            })

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
            # samples 包含: 'src', 'tgt' (固定索引) 和 'src_tokens', 'tgt_tokens' (原始 tokens)
            self.samples: List[Dict[str, List]] = pickle.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | List]:
        sample = self.samples[idx]
        
        # 返回基础张量和原始 tokens
        return {
            'src': torch.LongTensor(sample['src']),
            'tgt': torch.LongTensor(sample['tgt']),
            'src_len': torch.LongTensor([sample['src_len']]),
            'tgt_len': torch.LongTensor([sample['tgt_len']]),
            'src_tokens': sample['src_tokens'],
            'tgt_tokens': sample['tgt_tokens'],
            'tgt_text': sample['tgt_text']  # 新增：返回原始参考摘要文本
        }


class PGCollateFn:
    """
    Pointer-Generator 专用的 collate_fn。
    负责在批次创建时计算 OOV 映射 (src_oov_map) 和 
    扩展词表目标索引 (tgt_ext)。
    """
    def __init__(self, vocab: Vocab):
        self.vocab = vocab
        self.pad_idx = vocab.pad_idx
        
    def __call__(self, batch: List[Dict[str, torch.Tensor | List]]) -> Dict[str, torch.Tensor]:
        # 1. 基础张量堆叠
        src = torch.stack([item['src'] for item in batch])
        tgt = torch.stack([item['tgt'] for item in batch])
        src_len = torch.cat([item['src_len'] for item in batch])
        tgt_len = torch.cat([item['tgt_len'] for item in batch])
        
        # 2. 原始 tokens (用于 OOV 映射计算)
        src_tokens_batch = [item['src_tokens'] for item in batch]
        tgt_tokens_batch = [item['tgt_tokens'] for item in batch]

        B, T_src = src.shape
        T_tgt = tgt.shape[1]
        
        # max_oov_len = 0 # <--- 新增: 初始化批次中最大的 OOV 数量
        tgt_text_batch = [item['tgt_text'] for item in batch]  #<---新增：收集参考摘要文本

        # 3. 计算 OOV 映射 (src_oov_map)
        # src_oov_map: [B, T_src] 形状, 包含扩展词表索引
        oov_dicts_batch: List[Dict[int, str]] = []  # <---新增：存储每个样本的 OOV 映射
        src_oov_maps: List[List[int]] = []
        
        for tokens in src_tokens_batch:
            oov_dict: Dict[str, int] = {}
            oov_map: List[int] = []
            rev_oov_dict: Dict[int, str] = {}  # <---新增：相对索引→词（用于还原 OOV 词）
            
            for token in tokens:
                idx = self.vocab.word2idx.get(token) 
                if idx is None:
                    # OOV 词汇
                    if token not in oov_dict:
                        # 分配一个新的 OOV 索引 (vocab_size + 0, 1, 2, ...)
                        new_oov_idx = len(self.vocab) + len(oov_dict) # 使用 len(self.vocab) 作为词表大小
                        oov_dict[token] = new_oov_idx
                    oov_map.append(oov_dict[token])
                else:
                    oov_map.append(idx)
            
            # max_oov_len = max(max_oov_len, len(oov_dict))
            src_oov_maps.append(oov_map)
            oov_dicts_batch.append(rev_oov_dict)  # <---新增：保存当前样本的 OOV 映射

        # 转换为张量
        # src_oov_map 的形状与 src 相同，但包含扩展索引
        src_oov_map = torch.zeros_like(src)
        for i, oov_map in enumerate(src_oov_maps):
            # 确保只复制源文本有效长度
            src_oov_map[i, :len(oov_map)] = torch.tensor(oov_map[:T_src])
        
        # 4. 计算扩展目标索引 (tgt_ext)
        # tgt_ext: [B, T_tgt] 形状
        tgt_ext = torch.zeros_like(tgt)
        
        for i, (src_tokens, tgt_tokens) in enumerate(zip(src_tokens_batch, tgt_tokens_batch)):
            oov_dict: Dict[str, int] = {}
            vocab_size = len(self.vocab)
            
            # 重新构建该样本的 OOV 字典 (与上面计算 src_oov_map 保持一致)
            for token in src_tokens:
                if token not in self.vocab.word2idx: 
                    if token not in oov_dict:
                        new_oov_idx = vocab_size + len(oov_dict)
                        oov_dict[token] = new_oov_idx

            # 遍历目标序列，查找其在扩展词表中的索引
            for t, token in enumerate(tgt_tokens):
                if t >= T_tgt: continue # 目标序列已被截断
                
                # 如果是固定词表内的词或特殊符号
                if tgt[i, t].item() < vocab_size:
                    tgt_ext[i, t] = tgt[i, t]
                else:
                    # 查找 OOV 词汇
                    oov_idx = oov_dict.get(token)
                    if oov_idx is not None:
                        # OOV 词汇, 使用扩展索引
                        tgt_ext[i, t] = oov_idx
                    else:
                        # 目标 OOV 词汇未出现在源文本中，映射为 UNK
                        tgt_ext[i, t] = self.vocab.unk_idx
        
        # 最终返回的 batch 字典
        return {
            'src': src,
            'tgt': tgt_ext, # 使用扩展目标索引
            'src_len': src_len,
            'tgt_len': tgt_len,
            'src_oov_map': src_oov_map, # 源文本的扩展索引
            # 'max_oov_len': torch.LongTensor([max_oov_len]), # <--- 新增: 返回批次中最大的 OOV 数量
            'oov_dicts': oov_dicts_batch,  # <--- 新增：每个样本的 OOV 映射（相对索引→词）
            'tgt_text': tgt_text_batch     # <--- 新增：参考摘要文本列表
        }


def collate_fn_baseline(batch: List[Dict[str, torch.Tensor | List]]) -> Dict[str, torch.Tensor]:
    """基线模型专用的 collate_fn (仅堆叠张量)。"""
    src = torch.stack([item['src'] for item in batch])
    # 注意: tgt 仍然是固定词表索引
    tgt = torch.stack([item['tgt'] for item in batch]) 
    src_len = torch.cat([item['src_len'] for item in batch])
    tgt_len = torch.cat([item['tgt_len'] for item in batch])

    return {
        'src': src,
        'tgt': tgt,
        'src_len': src_len,
        'tgt_len': tgt_len,
    }


def get_dataloader(
    processed_dir: str,
    batch_size: int = 32,
    split: str = 'train',
    num_workers: int = 0,
    shuffle: Optional[bool] = None,
    vocab: Optional[Vocab] = None,
    include_oov: bool = False
) -> DataLoader:
    """
    基于预处理数据创建 DataLoader。
    新增 vocab 和 include_oov 参数用于 Pointer-Generator 机制。
    """
    if split not in PKL_FILENAMES:
        raise ValueError(f"Unknown split: {split}")
    
    if include_oov and vocab is None:
        raise ValueError("使用 include_oov=True 时，必须提供 vocab 对象。")

    # CNNDMDataset 现在可以加载 PG 所需的原始 tokens
    dataset = CNNDMDataset(processed_dir, split)
    if shuffle is None:
        shuffle = (split == 'train')
        
    if include_oov:
        # 使用 PG 专用的 collate_fn (传入 vocab 对象)
        collate = PGCollateFn(vocab)
    else:
        # 使用基线 collate_fn
        collate = collate_fn_baseline

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
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
