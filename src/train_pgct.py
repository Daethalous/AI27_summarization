"""
Transformer+Pointer-Generator+Coverage (PGCT) 模型正式训练脚本
"""
from __future__ import annotations
import sys
from pathlib import Path
import logging
from typing import Optional, List

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from datamodules.cnndm import prepare_datasets, get_dataloader
from models.pointer_generator_coverage.pgct_model import PGCTModel
from utils.vocab import Vocab

try:
    from utils.metrics import compute_rouge
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False


def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def calculate_nll_loss(predictions: torch.Tensor, targets: torch.Tensor, pad_idx: int) -> torch.Tensor:
    B, T, V = predictions.shape
    preds_flat = predictions.reshape(-1, V)
    targs_flat = targets.reshape(-1)
    log_probs = torch.log(preds_flat + 1e-10)
    picked = log_probs.gather(1, targs_flat.unsqueeze(1)).squeeze(1)
    mask = (targs_flat != pad_idx).float()
    loss = -(picked * mask).sum() / mask.sum()
    return loss


def main():
    logger = setup_logger()

    # 配置参数
    data_dir = "../data/raw"
    save_dir = "../checkpoints_pgct"
    num_epochs = 10
    batch_size = 8
    embed_size = 256
    hidden_size = 256
    num_encoder_layers = 3
    num_decoder_layers = 3
    nhead = 8
    dropout = 0.1
    cov_loss_weight = 1.0
    max_src_len = 400
    max_tgt_len = 100
    teacher_forcing_ratio = 0.5
    learning_rate = 1e-4
    grad_clip = 5.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 数据预处理
    data_dir = Path(data_dir)
    processed_dir = data_dir.parent / "processed"
    vocab_path = processed_dir / "vocab.json"
    if not vocab_path.exists():
        prepare_datasets(str(data_dir), str(processed_dir), str(vocab_path),
                         max_vocab_size=50000, min_freq=5,
                         max_src_len=max_src_len, max_tgt_len=max_tgt_len)

    vocab = Vocab.load(str(vocab_path))
    pad_idx = vocab.pad_idx
    sos_idx = vocab.sos_idx
    eos_idx = vocab.eos_idx

    train_loader = get_dataloader(str(processed_dir), batch_size=batch_size, split="train", shuffle=True, include_oov=True)
    val_loader = get_dataloader(str(processed_dir), batch_size=batch_size, split="val", shuffle=False, include_oov=True)

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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    tb_writer = SummaryWriter(log_dir=Path(save_dir)/"runs")

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs+1):
        model.train()
        running_nll = 0.0
        running_cov = 0.0
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}/{num_epochs}")
        for batch in pbar:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_oov_map = batch.get('src_oov_map', None)
            if src_oov_map is not None:
                src_oov_map = src_oov_map.to(device)

            optimizer.zero_grad()
            outputs, _, _, coverage_loss = model(src, tgt, src_oov_map=src_oov_map, teacher_forcing_ratio=teacher_forcing_ratio)
            nll_loss = calculate_nll_loss(outputs, tgt[:, 1:], pad_idx)
            total_loss = nll_loss + coverage_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            running_nll += nll_loss.item()
            running_cov += coverage_loss.item()
            pbar.set_postfix({"NLL": f"{nll_loss.item():.4f}", "Cov": f"{coverage_loss.item():.4f}"})

        avg_train_nll = running_nll / len(train_loader)
        avg_train_cov = running_cov / len(train_loader)
        avg_train_total = avg_train_nll + avg_train_cov
        logger.info(f"Epoch {epoch} Train Loss: Total={avg_train_total:.4f} (NLL={avg_train_nll:.4f}, Cov={avg_train_cov:.4f})")
        tb_writer.add_scalar("Train/TotalLoss", avg_train_total, epoch)

        # 验证
        model.eval()
        val_total_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                src = batch['src'].to(device)
                tgt = batch['tgt'].to(device)
                src_oov_map = batch.get('src_oov_map', None)
                if src_oov_map is not None:
                    src_oov_map = src_oov_map.to(device)

                outputs, _, _, coverage_loss = model(src, tgt, src_oov_map=src_oov_map, teacher_forcing_ratio=1.0)
                nll_loss = calculate_nll_loss(outputs, tgt[:, 1:], pad_idx)
                val_total_loss += (nll_loss + coverage_loss).item()

        avg_val_loss = val_total_loss / len(val_loader)
        logger.info(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f}")
        tb_writer.add_scalar("Val/TotalLoss", avg_val_loss, epoch)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), Path(save_dir)/"best.pt")
            logger.info(f"✨ 新最佳模型保存: {save_dir}/best.pt")

    tb_writer.close()
    logger.info("✅ 正式训练完成！")


if __name__ == "__main__":
    main()
