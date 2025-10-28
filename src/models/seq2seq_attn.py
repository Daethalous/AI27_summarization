"""
Seq2Seq + Attention 模型实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LuongAttention(nn.Module):
    """Luong Attention 机制（General 形式）"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
    
    def forward(
        self, 
        decoder_hidden: torch.Tensor, 
        encoder_outputs: torch.Tensor, 
        src_lens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_hidden: [B, hidden_size]
            encoder_outputs: [B, src_len, hidden_size]
            src_lens: [B] 源序列实际长度
            
        Returns:
            context: [B, hidden_size]
            attention_weights: [B, src_len]
        """
        # 计算注意力分数
        # [B, hidden_size] -> [B, 1, hidden_size]
        query = decoder_hidden.unsqueeze(1)
        
        # [B, src_len, hidden_size] -> [B, src_len, hidden_size]
        keys = self.W(encoder_outputs)
        
        # [B, 1, hidden_size] × [B, hidden_size, src_len] -> [B, 1, src_len]
        scores = torch.bmm(query, keys.transpose(1, 2))
        scores = scores.squeeze(1)  # [B, src_len]
        
        # 掩码padding部分
        if src_lens is not None:
            mask = torch.arange(encoder_outputs.size(1), device=scores.device)[None, :] >= src_lens[:, None]
            scores = scores.masked_fill(mask, -1e9)
        
        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=1)  # [B, src_len]
        
        # 计算上下文向量
        # [B, 1, src_len] × [B, src_len, hidden_size] -> [B, 1, hidden_size]
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)  # [B, hidden_size]
        
        return context, attention_weights


class Encoder(nn.Module):
    """BiLSTM 编码器"""
    
    def __init__(
        self, 
        vocab_size: int, 
        embed_size: int, 
        hidden_size: int, 
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_size, 
            hidden_size // 2,  # 双向，每个方向 hidden_size//2
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        src: torch.Tensor, 
        src_lens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            src: [B, src_len]
            src_lens: [B] 源序列实际长度
            
        Returns:
            outputs: [B, src_len, hidden_size]
            (hidden, cell): 各为 [num_layers*2, B, hidden_size//2]
        """
        # 嵌入
        embedded = self.dropout(self.embedding(src))  # [B, src_len, embed_size]
        
        # 打包序列（可选，提高效率）
        if src_lens is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False
            )
            outputs, (hidden, cell) = self.lstm(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, (hidden, cell) = self.lstm(embedded)
        
        return outputs, (hidden, cell)


class Decoder(nn.Module):
    """LSTM 解码器 + Attention"""
    
    def __init__(
        self, 
        vocab_size: int, 
        embed_size: int, 
        hidden_size: int, 
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.attention = LuongAttention(hidden_size)
        
        # LSTM 输入：embedding + context
        self.lstm = nn.LSTM(
            embed_size + hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层：concat(hidden, context) -> vocab
        self.out = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        tgt: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_lens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            tgt: [B, 1] 当前时间步输入
            hidden: [num_layers, B, hidden_size]
            cell: [num_layers, B, hidden_size]
            encoder_outputs: [B, src_len, hidden_size]
            src_lens: [B]
            
        Returns:
            output: [B, vocab_size]
            hidden: [num_layers, B, hidden_size]
            cell: [num_layers, B, hidden_size]
            attention_weights: [B, src_len]
        """
        # 嵌入
        embedded = self.dropout(self.embedding(tgt))  # [B, 1, embed_size]
        
        # 计算注意力上下文
        # 使用最后一层的hidden state
        context, attention_weights = self.attention(
            hidden[-1], encoder_outputs, src_lens
        )  # context: [B, hidden_size], attn: [B, src_len]
        
        # 拼接 embedding 和 context
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)  # [B, 1, embed+hidden]
        
        # LSTM
        lstm_output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # lstm_output: [B, 1, hidden_size]
        
        # 输出层：concat(lstm_output, context)
        lstm_output = lstm_output.squeeze(1)  # [B, hidden_size]
        output = self.out(torch.cat([lstm_output, context], dim=1))  # [B, vocab_size]
        
        return output, hidden, cell, attention_weights


class Seq2Seq(nn.Module):
    """Seq2Seq 模型"""
    
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 256,
        hidden_size: int = 512,
        num_layers: int = 1,
        dropout: float = 0.1,
        pad_idx: int = 0
    ):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.pad_idx = pad_idx
        
        # 将encoder的双向hidden合并为decoder的单向hidden
        self.bridge_h = nn.Linear(hidden_size, hidden_size)
        self.bridge_c = nn.Linear(hidden_size, hidden_size)
    
    def _bridge_states(
        self, 
        hidden: torch.Tensor, 
        cell: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将编码器的双向状态转换为解码器的单向状态
        
        Args:
            hidden: [num_layers*2, B, hidden_size//2]
            cell: [num_layers*2, B, hidden_size//2]
            
        Returns:
            hidden: [num_layers, B, hidden_size]
            cell: [num_layers, B, hidden_size]
        """
        num_layers = hidden.size(0) // 2
        batch_size = hidden.size(1)
        
        # 合并前向和后向
        # [num_layers*2, B, hidden//2] -> [num_layers, B, hidden]
        hidden = hidden.view(num_layers, 2, batch_size, -1)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)
        
        cell = cell.view(num_layers, 2, batch_size, -1)
        cell = torch.cat([cell[:, 0], cell[:, 1]], dim=2)
        
        # 线性变换
        hidden = torch.tanh(self.bridge_h(hidden))
        cell = torch.tanh(self.bridge_c(cell))
        
        return hidden, cell
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_lens: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0
    ) -> torch.Tensor:
        """
        Args:
            src: [B, src_len]
            tgt: [B, tgt_len]
            src_lens: [B]
            teacher_forcing_ratio: teacher forcing 概率
            
        Returns:
            outputs: [B, tgt_len, vocab_size]
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.decoder.out.out_features
        
        # 编码
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lens)
        
        # 转换状态
        hidden, cell = self._bridge_states(hidden, cell)
        
        # 准备输出
        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=src.device)
        
        # 第一个输入是 <SOS>
        decoder_input = tgt[:, 0].unsqueeze(1)  # [B, 1]
        
        # 逐步解码
        for t in range(1, tgt_len):
            output, hidden, cell, _ = self.decoder(
                decoder_input, hidden, cell, encoder_outputs, src_lens
            )
            
            outputs[:, t] = output
            
            # Teacher forcing
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            if use_teacher_forcing:
                decoder_input = tgt[:, t].unsqueeze(1)
            else:
                decoder_input = output.argmax(1).unsqueeze(1)
        
        return outputs
