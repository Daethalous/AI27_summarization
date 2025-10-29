"""
Pointer-Generator Network 训练脚本

用法示例：
    python train_pg.py --data_dir ../data/raw --num_epochs 2 --num_samples 200
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent))

from datamodules.cnndm import prepare_datasets, get_dataloader
from models.pointer_generator import PointerGeneratorSeq2Seq
from utils.vocab import Vocab

try:
    from utils.metrics import calculate_rouge
    HAS_ROUGE = True
except Exception:
    HAS_ROUGE = False


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def calculate_loss_with_extended_vocab(
    predictions: torch.Tensor,   # [B, T-1, ext_vocab]
    targets: torch.Tensor,       # [B, T-1]
    pad_idx: int
) -> torch.Tensor:
    """与 quick_test_pg 中一致的 NLL 计算（支持扩展词表id）。"""
    B, Tm1, V = predictions.shape
    preds_flat = predictions.reshape(-1, V)             # [B*(T-1), V]
    targs_flat = targets.reshape(-1)                    # [B*(T-1)]
    log_probs = torch.log(preds_flat + 1e-10)           # 数值稳定
    picked = log_probs.gather(1, targs_flat.unsqueeze(1)).squeeze(1)
    mask = (targs_flat != pad_idx).float()
    loss = -(picked * mask).sum() / (mask.sum() + 1e-8)
    return loss


def decode_with_oov(
    ids: List[int],
    vocab: Vocab,
    oov_list: Optional[List[str]],
    eos_idx: int,
    pad_idx: int,
    sos_idx: int
) -> List[str]:
    """把可能含扩展词表 id 的序列转为 token 列表。"""
    out = []
    for pid in ids:
        if pid == eos_idx:
            break
        if pid in (pad_idx, sos_idx):
            continue
        if pid < len(vocab):
            out.append(vocab.idx2word.get(pid, "<UNK>"))
        else:
            # 扩展词表：映射回该样本的 OOV 列表
            ext_idx = pid - len(vocab)
            if oov_list and 0 <= ext_idx < len(oov_list):
                out.append(oov_list[ext_idx])
            else:
                out.append("<UNK>")
    return out


def main():
    logger = setup_logger()

    parser = argparse.ArgumentParser(description="Train Pointer-Generator (aligned with quick_test)")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    parser.add_argument("--grad_clip", type=float, default=5.0)
    # 数据 & 采样
    parser.add_argument("--data_dir", type=str, default="../data/raw", help="原始数据目录")
    parser.add_argument("--num_samples", type=int, default=0, help="训练样本（0=使用完整数据集，>0 = 使用数据集子集）")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)

    # 词表/预处理
    parser.add_argument("--max_vocab_size", type=int, default=50000)
    parser.add_argument("--min_freq", type=int, default=5)
    parser.add_argument("--max_src_len", type=int, default=400)
    parser.add_argument("--max_tgt_len", type=int, default=100)

    # 模型
    parser.add_argument("--embed_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_oov_size", type=int, default=1000)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)

    # 训练
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="../checkpoints_pg")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # —— 数据与词表 —— #
    data_dir = Path(args.data_dir)
    processed_dir = data_dir.parent / "processed"
    vocab_path = processed_dir / "vocab.json"

    if not vocab_path.exists():
        logger.info("词表不存在，开始预处理数据…")
        processed_dir.mkdir(parents=True, exist_ok=True)
        prepare_datasets(
            raw_dir=str(data_dir),
            output_dir=str(processed_dir),
            max_vocab_size=args.max_vocab_size,
            min_freq=args.min_freq,
            max_src_len=args.max_src_len,
            max_tgt_len=args.max_tgt_len,
        )

    vocab = Vocab.load(str(vocab_path))
    logger.info(f"词表大小: {len(vocab)}")

    # 与 quick_test_pointer_generator 完全一致的 dataloader 接口
    train_loader = get_dataloader(
        processed_dir=str(processed_dir),
        batch_size=args.batch_size,
        split="train",
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_loader = get_dataloader(
        processed_dir=str(processed_dir),
        batch_size=args.batch_size,
        split="val",
        num_workers=args.num_workers,
        shuffle=False,
    )

    # 处理训练样本数量
    total_train_samples = len(train_loader.dataset)
    if args.num_samples > 0 and args.num_samples < total_train_samples:
        # 使用子集（快速验证）
        logger.info(f"⚠️ 使用训练子集: {args.num_samples}/{total_train_samples} 样本")
        train_indices = list(range(args.num_samples))
        train_subset = Subset(train_loader.dataset, train_indices)
        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=train_loader.collate_fn,
            pin_memory=True,
        )   
        logger.info(f"训练样本数: {len(train_subset)}")
    else:
        # 使用全部数据（正式训练）
        logger.info(f"使用完整训练集: {total_train_samples} 样本")
        logger.info(f"训练样本数: {total_train_samples}")

    logger.info(f"验证样本数: {len(val_loader.dataset)}")

    pad_idx = vocab.word2idx["<PAD>"]
    sos_idx = vocab.word2idx["<SOS>"]
    eos_idx = vocab.word2idx["<EOS>"]

    # —— 模型 —— #
    model = PointerGeneratorSeq2Seq(
        vocab_size=len(vocab),
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_oov_size=args.max_oov_size,
        pad_idx=pad_idx,
    ).to(device)
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")

    # —— 训练 —— #
    for epoch in range(1, args.num_epochs + 1):
        tqdm.write("\n" + "=" * 50)
        tqdm.write(f"Epoch {epoch}/{args.num_epochs}")
        tqdm.write("=" * 50)

        model.train()
        running = 0.0
        pbar = tqdm(
            train_loader, 
            desc=f"Train {epoch}/{args.num_epochs}",
            ncols=100,           
            dynamic_ncols=False,  
            leave=True,          
            file=sys.stdout     
        )
        
        for batch in pbar:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            src_lens = batch.get("src_lens", None)
            src_oov_map = batch.get("src_oov_map", None)

            if src_lens is not None:
                src_lens = src_lens.to(device)
            if src_oov_map is not None:
                src_oov_map = src_oov_map.to(device)

            optimizer.zero_grad()
            # 模型输出 [B, T-1, ext_vocab]
            outputs = model(
                src=src,
                tgt=tgt,
                src_lens=src_lens,
                src_oov_map=src_oov_map,
                teacher_forcing_ratio=args.teacher_forcing_ratio,
            )

            loss = calculate_loss_with_extended_vocab(
                predictions=outputs,
                targets=tgt[:, 1:],   # 去掉 <SOS>
                pad_idx=pad_idx
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = running / max(1, len(train_loader))
        logger.info(f"Train Loss: {train_loss:.4f}")

        # —— 验证 —— #
        model.eval()
        val_running = 0.0
        all_preds: List[str] = []
        all_refs: List[str] = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                src = batch["src"].to(device)
                tgt = batch["tgt"].to(device)
                src_lens = batch.get("src_lens", None)
                src_oov_map = batch.get("src_oov_map", None)
                oov_lists = batch.get("oov_list", None)  # 每样本 OOV 原词列表

                if src_lens is not None:
                    src_lens = src_lens.to(device)
                if src_oov_map is not None:
                    src_oov_map = src_oov_map.to(device)

                outputs = model(
                    src=src,
                    tgt=tgt,
                    src_lens=src_lens,
                    src_oov_map=src_oov_map,
                    teacher_forcing_ratio=1.0,
                )
                loss = calculate_loss_with_extended_vocab(
                    predictions=outputs,
                    targets=tgt[:, 1:],
                    pad_idx=pad_idx,
                )
                val_running += loss.item()

                # 生成以便可选地计算 ROUGE
                try:
                    preds, _ = model.generate(
                        src=src,
                        src_lens=src_lens,
                        src_oov_map=src_oov_map,
                        max_length= args.max_tgt_len,
                        sos_idx=sos_idx,
                        eos_idx=eos_idx,
                    )
                except TypeError:
                    # 如果 generate 不支持这些关键字，退化成只传必须项
                    preds, _ = model.generate(src=src, max_length=args.max_tgt_len)

                # 解码预测和参考（含 OOV 回填）
                for i in range(len(preds)):
                    pred_ids = preds[i].tolist()
                    ref_ids = tgt[i].tolist()
                    oov_list_i = oov_lists[i] if oov_lists else None

                    pred_tokens = decode_with_oov(pred_ids, vocab, oov_list_i, eos_idx, pad_idx, sos_idx)
                    ref_tokens = decode_with_oov(ref_ids, vocab, None, eos_idx, pad_idx, sos_idx)

                    all_preds.append(" ".join(pred_tokens))
                    all_refs.append(" ".join(ref_tokens))

        val_loss = val_running / max(1, len(val_loader))
        logger.info(f"Val Loss: {val_loss:.4f}")

        if HAS_ROUGE:
            rouge = calculate_rouge(all_preds, all_refs)
            logger.info(f"Val ROUGE-1: {rouge['rouge1']:.4f} | ROUGE-2: {rouge['rouge2']:.4f} | ROUGE-L: {rouge['rougeL']:.4f}")
        else:
            logger.info("未检测到 calculate_rouge，跳过 ROUGE 计算。")

        # 保存模型
        ckpt_path = Path(args.save_dir) / f"epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }, ckpt_path.as_posix())
        logger.info(f"保存检查点: {ckpt_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = Path(args.save_dir) / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, best_path.as_posix())
            logger.info(f"✨ 新最佳模型（Val Loss {best_val_loss:.4f}）：{best_path}")

    logger.info("\n" + "=" * 50)
    logger.info("训练完成！")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()


