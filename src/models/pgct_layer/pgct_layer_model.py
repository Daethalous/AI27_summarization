"""完整的Transformer+Pointer-Generator+Coverage模型 (Model Aggregator)."""
from __future__ import annotations
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
from .pgct_layer_encoder import PGCT_layer_Encoder
from .pgct_decoder import PGCTDecoder
from .pgct_decoding import pgct_greedy_decode, pgct_beam_search_decode 


class LayerAttention(nn.Module):
    """
    层间注意力模块：对 encoder 每一层的输出做 softmax 加权融合。
    输入：all_layer_outputs: List[Tensor]，长度 = L，每个 [B, T, d_model]
    输出：
      - fused: [B, T, d_model] 融合后的表示
      - alpha: [L] 每一层的注意力权重（可视化用）
    """
    def __init__(self, num_layers: int):
        super().__init__()
        # 每一层一个可学习的标量得分
        self.layer_scores = nn.Parameter(torch.zeros(num_layers))

    def forward(self, all_layer_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # all_layer_outputs 堆叠成 [L, B, T, d_model]
        stacked = torch.stack(all_layer_outputs, dim=0)  # [L, B, T, d]

        # 对层维度做 softmax，得到 [L]
        alpha = torch.softmax(self.layer_scores, dim=0)  # [L]

        # 加权和：sum_l alpha_l * H^{(l)} -> [B, T, d]
        # 'l,lbtf->btf'：l 对应层，b/t/f 分别是 batch / time / feature
        fused = torch.einsum('l,lbtf->btf', alpha, stacked)

        return fused, alpha


class PGCT_layer_Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 512,
        hidden_size: int = 512,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        nhead: int = 8,
        dropout: float = 0.1,
        pad_idx: int = 0,
        cov_loss_weight: float = 1.0,
        max_src_len: int = 400,
        max_tgt_len: int = 100,
        use_layer_attention: bool = True,   # 是否启用层间注意力
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.max_tgt_len = max_tgt_len
        self.hidden_size = hidden_size
        self.use_layer_attention = use_layer_attention
        
        self.encoder = PGCT_layer_Encoder(
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

        # 如果 embed_size != hidden_size，这里做一层投影（双保险）
        self.encoder_proj = nn.Linear(hidden_size, hidden_size) if embed_size != hidden_size else nn.Identity()

        # 新增：层间注意力模块（可选）
        if self.use_layer_attention:
            self.layer_attn = LayerAttention(num_encoder_layers)
        else:
            self.layer_attn = None

    def encode(
        self,
        src: torch.Tensor,
        src_lens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 启用层间注意力：调用 encoder.forward_with_layers() + LayerAttention
        if self.use_layer_attention:
            encoder_outputs, src_mask, all_layer_outputs = self.encoder.forward_with_layers(
                src, src_lens
            )
            fused_outputs, _ = self.layer_attn(all_layer_outputs)  # [B, T, d]
            encoder_outputs = fused_outputs
        else:
            encoder_outputs, src_mask = self.encoder(src, src_lens)

        if encoder_outputs.size(-1) != self.hidden_size:
            encoder_outputs = self.encoder_proj(encoder_outputs)

        if src_lens is None:
            src_mask = self.encoder.generate_src_mask(src, src_lens)

        return encoder_outputs, src_mask

    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        src_lens: Optional[torch.Tensor] = None,
        src_oov_map: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        # 不再直接 self.encoder(...)，而是走 self.encode()（包含层间注意力）
        encoder_outputs, src_mask = self.encode(src, src_lens)
        
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
            return outputs, None, None, coverage_loss
        else:
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
        # 这里暂时保持不变，pgct_greedy_decode 里目前还是直接访问 model.encoder。
        # 之后你可以在 pgct_decoding.py 里改成使用 model.encode()，
        # 这样推理时也会走层间注意力。
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
