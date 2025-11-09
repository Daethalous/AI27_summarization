from __future__ import annotations
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Transformer位置编码（正弦余弦编码）"""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PGCT_layer_Encoder(nn.Module):
    """Transformer编码器（适配PG+Coverage机制）"""
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 512,
        hidden_size: int = 512,
        num_layers: int = 3,
        nhead: int = 8,
        dropout: float = 0.1,
        pad_idx: int = 0,
        max_src_len: int = 400
    ):
        super().__init__()
        self.d_model = hidden_size
        self.pad_idx = pad_idx
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers  # 新增：记录层数
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.embed_proj = nn.Linear(embed_size, hidden_size) if embed_size != hidden_size else nn.Identity()
        self.pos_encoding = PositionalEncoding(d_model=hidden_size, max_len=max_src_len, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        # 官方的 TransformerEncoder 里面有 layers 和 norm 两个属性
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def generate_src_mask(self, src: torch.Tensor, src_lens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """生成源序列掩码（屏蔽 <PAD>）。True 表示 mask"""
        if src_lens is not None:
            batch_size, seq_len = src.size()
            mask = torch.arange(seq_len, device=src.device).unsqueeze(0).expand(batch_size, seq_len)
            mask = mask >= src_lens.unsqueeze(1)
            return mask
        return (src == self.pad_idx)

    def forward(
        self,
        src: torch.Tensor,
        src_lens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        原有接口：只返回最后一层的编码结果和 src_mask。
        旧代码（如解码模块）可以继续用这个接口，不受影响。
        """
        embed = self.embedding(src)
        embed = self.embed_proj(embed)
        embed = self.pos_encoding(embed)
        
        src_mask = self.generate_src_mask(src, src_lens)
        encoder_outputs = self.transformer_encoder(embed, src_key_padding_mask=src_mask)
        return encoder_outputs, src_mask

    def forward_with_layers(
        self,
        src: torch.Tensor,
        src_lens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        新增接口：除了最后一层编码输出，还返回每一层的输出列表，用于层间注意力。
        返回：
          - encoder_outputs: 最后一层（norm 后）的输出 [B, T, d_model]
          - src_mask: 源序列 key_padding_mask [B, T]
          - all_layer_outputs: 长度 = num_layers 的 list，每个元素形状 [B, T, d_model]
        """
        embed = self.embedding(src)
        embed = self.embed_proj(embed)
        embed = self.pos_encoding(embed)

        src_mask = self.generate_src_mask(src, src_lens)

        x = embed
        all_layer_outputs = []
        # 逐层跑 TransformerEncoderLayer
        for layer in self.transformer_encoder.layers:
            x = layer(x, src_key_padding_mask=src_mask)
            all_layer_outputs.append(x)

        # 应用可选的层归一化（如果在构造 TransformerEncoder 时指定了 norm）
        if self.transformer_encoder.norm is not None:
            x = self.transformer_encoder.norm(x)

        encoder_outputs = x  # [B, T, d_model]
        return encoder_outputs, src_mask, all_layer_outputs
