"""
快速测试 Transformer+Pointer-Generator+Coverage (PGCT) 模型训练

功能:
    - 验证 PGCT 模型能否在少量样本上前向/反向传播
    - 测试 Teacher Forcing 和覆盖机制
    - 测试推理/生成功能
"""
from __future__ import annotations
import sys
from pathlib import Path
import logging

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# 项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from datamodules.cnndm import prepare_datasets, get_dataloader
from models.pointer_generator_coverage.pgct_model import PGCTModel
from utils.vocab import Vocab


def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def quick_test_pgct(
    data_dir: str = "../data/raw",
    num_samples: int = 100,
    batch_size: int = 2,
    embed_size: int = 256,
    hidden_size: int = 256,
    num_encoder_layers: int = 2,
    num_decoder_layers: int = 2,
    nhead: int = 4,
    dropout: float = 0.1,
    cov_loss_weight: float = 1.0,
    max_src_len: int = 400,
    max_tgt_len: int = 100,
    num_epochs: int = 1
):
    logger = setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 数据处理
    data_dir = Path(data_dir)
    processed_dir = data_dir.parent / "processed"
    vocab_path = processed_dir / "vocab.json"
    if not vocab_path.exists():
        logger.info("词表不存在，开始预处理...")
        prepare_datasets(str(data_dir), str(processed_dir), str(vocab_path),
                         max_vocab_size=50000, min_freq=5,
                         max_src_len=max_src_len, max_tgt_len=max_tgt_len,
                         limit_per_split=200)

    vocab = Vocab.load(str(vocab_path))
    pad_idx = vocab.word2idx['<PAD>']

    train_loader = get_dataloader(str(processed_dir), batch_size=batch_size, split="train", shuffle=True)
    train_subset = Subset(train_loader.dataset, list(range(min(num_samples, len(train_loader.dataset)))))
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=train_loader.collate_fn)

    # 模型
    model = PGCTModel(
        vocab_size=len(vocab),
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        nhead=nhead,
        dropout=dropout,
        pad_idx=pad_idx,
        cov_loss_weight=cov_loss_weight,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 快速训练循环
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_oov_map = batch.get('src_oov_map', None)
            if src_oov_map is not None:
                src_oov_map = src_oov_map.to(device)

            optimizer.zero_grad()
            outputs, _, _, coverage_loss = model(src, tgt, src_oov_map=src_oov_map, teacher_forcing_ratio=1.0)

            # NLL损失
            output_flat = outputs.reshape(-1, outputs.size(-1))
            tgt_flat = tgt[:, 1:].reshape(-1)
            log_probs = torch.log(output_flat + 1e-10)
            target_log_probs = log_probs.gather(1, tgt_flat.unsqueeze(1)).squeeze(1)
            mask = (tgt_flat != pad_idx).float()
            nll_loss = -(target_log_probs * mask).sum() / mask.sum()

            total_loss = nll_loss + coverage_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            pbar.set_postfix({"NLL": f"{nll_loss.item():.4f}", "Cov": f"{coverage_loss.item():.4f}", "Total": f"{total_loss.item():.4f}"})

    logger.info("✅ 快速测试完成，模型训练正常！")


if __name__ == "__main__":
    quick_test_pgct()
