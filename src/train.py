"""Seq2Seq + Attention Baseline 训练脚本."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 添加src到path
sys.path.insert(0, str(Path(__file__).parent))

from datamodules.cnndm import get_dataloader, prepare_datasets
from models.baseline.model import Seq2Seq
from utils.metrics import batch_compute_metrics, print_metrics


def setup_logger(log_path: str) -> logging.Logger:
    """配置日志记录器，输出到终端与文件。"""
    logger = logging.getLogger('baseline')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def compute_loss(model_output: torch.Tensor, targets: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """按 token 交叉熵计算损失，忽略 padding。"""
    output = model_output[:, 1:, :].contiguous().view(-1, model_output.size(-1))
    targets = targets[:, 1:].contiguous().view(-1)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    return criterion(output, targets)


def train_epoch(
    model: Seq2Seq,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pad_idx: int,
    epoch: int,
    teacher_forcing_ratio: float,
    writer: Optional[SummaryWriter] = None
) -> float:
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        src_len = batch['src_len'].squeeze().to(device)

        optimizer.zero_grad()
        output = model(src, tgt, src_len, teacher_forcing_ratio=teacher_forcing_ratio)
        loss = compute_loss(output, tgt, pad_idx)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate(model, dataloader, device, vocab):
    """评估模型
    
    Args:
        model: Seq2Seq模型
        dataloader: 数据加载器
        device: 设备
        vocab: 词表
        
    Returns:
        avg_loss, metrics字典
    """
    model.eval()
    total_loss = 0.0
    
    all_predictions = []
    all_references = []
    
    pbar = tqdm(dataloader, desc="Evaluating")
    for batch in pbar:
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        src_len = batch['src_len'].squeeze().to(device)
        
        # 前向传播
        output = model(src, tgt, src_len, teacher_forcing_ratio=0.0)
        
        # 计算损失
        loss = compute_loss(output, tgt, vocab.pad_idx)
        total_loss += loss.item()
        
        # 解码预测（用于ROUGE计算）
        pred_ids = output.argmax(dim=-1)  # [B, tgt_len]
        
        for i in range(pred_ids.size(0)):
            pred_tokens = vocab.decode(pred_ids[i].cpu().tolist(), skip_special=True)
            ref_tokens = vocab.decode(tgt[i].cpu().tolist(), skip_special=True)
            
            all_predictions.append(pred_tokens)
            all_references.append(ref_tokens)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    
    # 计算ROUGE等指标
    vocab_set = set(vocab.word2idx.keys())
    metrics = batch_compute_metrics(all_predictions, all_references, vocab_set)
    metrics['loss'] = avg_loss
    
    return avg_loss, metrics


def main(args):
    """主训练函数"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载配置
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # 合并命令行参数（优先级更高）
    raw_data_dir = args.data_dir or config.get('data_dir', './data/raw')
    vocab_path = args.vocab_path or config.get('vocab_path', './data/processed/vocab.json')
    processed_dir = config.get('processed_dir', os.path.dirname(vocab_path))
    save_dir = args.save_dir or config.get('save_dir', './checkpoints')
    log_path = config.get('log_path', './logs/baseline.log')

    batch_size = args.batch_size or config.get('batch_size', 32)
    max_src_len = args.max_src_len or config.get('max_src_len', 512)
    max_tgt_len = args.max_tgt_len or config.get('max_tgt_len', 512)

    embed_size = config.get('embed_size', 256)
    hidden_size = config.get('hidden_size', 512)
    num_layers = config.get('num_layers', 1)
    dropout = config.get('dropout', 0.1)
    
    num_epochs = args.epochs or config.get('num_epochs', 10)
    lr = args.lr or config.get('learning_rate', 1e-4)
    teacher_forcing_ratio = config.get('teacher_forcing_ratio', 0.5)
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    logger = setup_logger(log_path)
    logger.info("===== Baseline Training Start =====")
    logger.info("Raw data dir: %s", raw_data_dir)
    logger.info("Processed dir: %s", processed_dir)
    logger.info("Vocab path: %s", vocab_path)

    vocab = prepare_datasets(
        raw_dir=raw_data_dir,
        processed_dir=processed_dir,
        vocab_path=vocab_path,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        max_vocab_size=config.get('max_vocab_size', 50000),
        min_freq=config.get('min_freq', 5)
    )
    logger.info("Vocabulary size: %d", len(vocab))

    print("创建数据加载器...")
    train_loader = get_dataloader(
        processed_dir, batch_size, split='train', num_workers=args.num_workers
    )
    val_loader = get_dataloader(
        processed_dir, batch_size, split='val', num_workers=args.num_workers
    )
    
    # 创建模型
    print("创建模型...")
    model = Seq2Seq(
        vocab_size=len(vocab),
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        pad_idx=vocab.pad_idx
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    logger.info("Model parameters: %s", f"{total_params:,}")
    
    # 优化器和调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'runs'))
    
    # 训练循环
    best_val_loss = float('inf')
    rouge_l_history: List[float] = []
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'=' * 60}")
        
        # 训练
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            vocab.pad_idx,
            epoch,
            teacher_forcing_ratio,
            writer
        )
        print(f"Train Loss: {train_loss:.4f}")
        logger.info("Epoch %d Train Loss: %.4f", epoch, train_loss)

        val_loss, val_metrics = evaluate(model, val_loader, device, vocab)
        print_metrics(val_metrics, prefix="Validation")

        rouge_l = val_metrics.get('rougeL_f', 0.0)
        rouge_l_history.append(rouge_l)
        delta = None
        if len(rouge_l_history) > 1:
            delta = abs(rouge_l_history[-1] - rouge_l_history[-2])

        logger.info(
            "Epoch %d Val Loss: %.4f | ROUGE-L: %.4f%s",
            epoch,
            val_loss,
            rouge_l,
            f" | ΔROUGE-L: {delta:.4f}" if delta is not None else ""
        )

        writer.add_scalar('Val/Loss', val_loss, epoch)
        for key, value in val_metrics.items():
            if 'rouge' in key:
                writer.add_scalar(f'Val/{key}', value, epoch)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f"✓ 保存最佳模型: {checkpoint_path}")
            logger.info("Saved best checkpoint: %s", checkpoint_path)
        
        # 定期保存检查点
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f"✓ 保存检查点: {checkpoint_path}")
            logger.info("Saved periodic checkpoint: %s", checkpoint_path)
    
    writer.close()
    print(f"\n训练完成！最佳验证loss: {best_val_loss:.4f}")
    logger.info("Training finished. Best Val Loss: %.4f", best_val_loss)

    if rouge_l_history:
        epochs_axis = list(range(1, len(rouge_l_history) + 1))
        plt.figure(figsize=(8, 5))
        plt.plot(epochs_axis, rouge_l_history, marker='o', color='#1f77b4', label='ROUGE-L (F1)')
        plt.title('ROUGE-L Trend')
        plt.xlabel('Epoch')
        plt.ylabel('ROUGE-L F1')
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        plt.xticks(epochs_axis)
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(save_dir, 'rouge_l_trend.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logger.info("Saved ROUGE-L trend plot: %s", plot_path)

    if len(rouge_l_history) > 1:
        rouge_deltas = [abs(rouge_l_history[i] - rouge_l_history[i - 1]) for i in range(1, len(rouge_l_history))]
        logger.info("ROUGE-L deltas: %s", [f"{delta:.4f}" for delta in rouge_deltas])
        if rouge_deltas:
            if len(rouge_deltas) >= 3:
                stable = all(delta < 0.01 for delta in rouge_deltas[-3:])
            else:
                stable = rouge_deltas[-1] < 0.01
            logger.info("ROUGE-L stability (<1%%) satisfied: %s", stable)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练 Seq2Seq + Attention Baseline')
    
    # 数据相关
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--data_dir', type=str, help='原始数据目录（包含train/validation/test子目录）')
    parser.add_argument('--vocab_path', type=str, help='词表保存/加载路径')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    
    # 训练超参数
    parser.add_argument('--batch_size', type=int, help='批次大小')
    parser.add_argument('--max_src_len', type=int, help='最大源文本长度')
    parser.add_argument('--max_tgt_len', type=int, help='最大目标文本长度')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--lr', type=float, help='学习率')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')
    
    args = parser.parse_args()
    main(args)
