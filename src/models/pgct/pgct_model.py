
"""完整的Transformer+Pointer-Generator+Coverage模型 (Model Aggregator)."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入组件
from .pgct_encoder import PGCTEncoder
from .pgct_decoder import PGCTDecoder
# 导入PGCT专用的解码工具函数
from .pgct_decoding import pgct_greedy_decode, pgct_beam_search_decode 


class PGCTModel(nn.Module):
    """完整的Transformer+Pointer-Generator+Coverage模型"""
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 256,
        hidden_size: int = 256,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        nhead: int = 8,
        dropout: float = 0.1,
        pad_idx: int = 0,
        cov_loss_weight: float = 1.0,
        max_src_len: int = 400,
        max_tgt_len: int = 100
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.max_tgt_len = max_tgt_len
        
        self.encoder = PGCTEncoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_encoder_layers,
            nhead=nhead,
            dropout=dropout,
            pad_idx=pad_idx,
            max_src_len=max_src_len
        )
        
        self.decoder = PGCTDecoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_decoder_layers,
            nhead=nhead,
            dropout=dropout,
            pad_idx=pad_idx,
            cov_loss_weight=cov_loss_weight,
            max_tgt_len=max_tgt_len
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        src_lens: Optional[torch.Tensor] = None,
        src_oov_map: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        训练阶段前向传播或推理阶段调用。
        - 当 tgt 提供时，按 teacher_forcing_ratio 决定是否使用 teacher forcing。
        - 当 tgt 为 None 时，为推理阶段，需调用 generate() 或 beam_search()。
        """
        # 编码器前向
        encoder_outputs, _ = self.encoder(src, src_lens)
        src_mask = self.encoder.generate_src_mask(src)
        
        if tgt is not None:
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            outputs, coverage_loss = self.decoder(
                tgt=tgt,
                encoder_outputs=encoder_outputs,
                src=src,
                src_mask=src_mask,
                src_oov_map=src_oov_map,
                teacher_forcing=use_teacher_forcing
            )
            return outputs, coverage_loss
        else:
            # 推理阶段，用户需调用 generate() 或 beam_search()
            raise ValueError("tgt is None, use generate() or beam_search() for inference.")

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        src_lens: Optional[torch.Tensor] = None,
        src_oov_map: Optional[torch.Tensor] = None,
        max_length: int = 100,
        sos_idx: int = 2,
        eos_idx: int = 3,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """调用PGCT专用的贪婪解码函数"""
        return pgct_greedy_decode(
            model=self,
            src=src,
            src_lens=src_lens,
            src_oov_map=src_oov_map,
            max_length=max_length,
            sos_idx=sos_idx,
            eos_idx=eos_idx,
            device=device
        )

    @torch.no_grad()
    def beam_search(
        self,
        src: torch.Tensor,
        src_lens: Optional[torch.Tensor] = None,
        src_oov_map: Optional[torch.Tensor] = None,
        beam_size: int = 5,
        max_length: int = 100,
        sos_idx: int = 2,
        eos_idx: int = 3,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """调用PGCT专用的束搜索函数"""
        return pgct_beam_search_decode(
            model=self,
            src=src,
            src_lens=src_lens,
            src_oov_map=src_oov_map,
            beam_size=beam_size,
            max_length=max_length,
            sos_idx=sos_idx,
            eos_idx=eos_idx,
            device=device
        )
