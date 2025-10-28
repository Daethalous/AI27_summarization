"""
词表构建与序列化模块
"""
import json
from collections import Counter
from typing import List, Dict, Optional


class Vocab:
    """词表类，用于单词和索引之间的映射"""
    
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    SOS_TOKEN = '<SOS>'
    EOS_TOKEN = '<EOS>'
    
    def __init__(self, max_vocab_size: int = 50000, min_freq: int = 1):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        
        # 特殊符号
        self.special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]
        
        # 词表映射
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freq: Counter = Counter()
        
        # 初始化特殊符号
        for idx, token in enumerate(self.special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token
    
    @property
    def pad_idx(self) -> int:
        return self.word2idx[self.PAD_TOKEN]
    
    @property
    def unk_idx(self) -> int:
        return self.word2idx[self.UNK_TOKEN]
    
    @property
    def sos_idx(self) -> int:
        return self.word2idx[self.SOS_TOKEN]
    
    @property
    def eos_idx(self) -> int:
        return self.word2idx[self.EOS_TOKEN]
    
    def __len__(self) -> int:
        return len(self.word2idx)
    
    def build_vocab(self, texts: List[List[str]]):
        """从文本列表构建词表
        
        Args:
            texts: 分词后的文本列表，每个元素是一个单词列表
        """
        # 统计词频
        for text in texts:
            self.word_freq.update(text)
        
        # 按词频排序，选取高频词
        sorted_words = sorted(
            self.word_freq.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 构建词表（跳过特殊符号）
        idx = len(self.special_tokens)
        for word, freq in sorted_words:
            if freq < self.min_freq:
                break
            if len(self.word2idx) >= self.max_vocab_size:
                break
            if word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
        print(f"词表构建完成: {len(self)} 个词 (包含特殊符号)")
        print(f"  - 最小词频: {self.min_freq}")
        print(f"  - 总词数: {len(self.word_freq)}")
    
    def encode(self, words: List[str], max_len: Optional[int] = None) -> List[int]:
        """将单词列表转换为索引列表
        
        Args:
            words: 单词列表
            max_len: 最大长度，如果指定则截断或填充
            
        Returns:
            索引列表
        """
        indices = [self.word2idx.get(w, self.unk_idx) for w in words]
        
        if max_len is not None:
            if len(indices) > max_len:
                indices = indices[:max_len]
            elif len(indices) < max_len:
                indices = indices + [self.pad_idx] * (max_len - len(indices))
        
        return indices
    
    def decode(self, indices: List[int], skip_special: bool = True) -> List[str]:
        """将索引列表转换为单词列表
        
        Args:
            indices: 索引列表
            skip_special: 是否跳过特殊符号
            
        Returns:
            单词列表
        """
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, self.UNK_TOKEN)
            if skip_special and word in self.special_tokens:
                continue
            words.append(word)
        return words
    
    def save(self, filepath: str):
        """保存词表到文件"""
        data = {
            'word2idx': self.word2idx,
            'idx2word': {str(k): v for k, v in self.idx2word.items()},
            'word_freq': dict(self.word_freq),
            'max_vocab_size': self.max_vocab_size,
            'min_freq': self.min_freq
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"词表已保存到: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'Vocab':
        """从文件加载词表"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vocab = cls(
            max_vocab_size=data['max_vocab_size'],
            min_freq=data['min_freq']
        )
        vocab.word2idx = data['word2idx']
        vocab.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        vocab.word_freq = Counter(data['word_freq'])
        
        print(f"词表已加载: {len(vocab)} 个词")
        return vocab
