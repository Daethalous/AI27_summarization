"""
快速测试训练脚本 - 只使用少量数据验证代码能正常运行

使用方法:
    python quick_test_train.py --model baseline   # 测试baseline模型
    python quick_test_train.py --model pg         # 测试pointer-generator模型
    python quick_test_train.py --model pg_cov     # 测试pointer-generator with coverage模型 (新增)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent))
# 自动定位项目根目录（假设当前文件是 src/quick_test_train.py）
project_root = Path(__file__).resolve().parents[1]
default_data_dir = project_root / 'data' / 'raw'


from datamodules.cnndm import prepare_datasets, get_dataloader
from models.baseline.model import Seq2Seq
from models.pointer_generator import PointerGeneratorSeq2Seq
# =======================================================
# 🚀 关键修改 1: 导入 PG with Coverage 模型
from models.pointer_generator_coverage.pg_coverage_model import PGCoverageSeq2Seq
# =======================================================
from utils.vocab import Vocab
from utils.metrics import batch_compute_metrics


def setup_logger():
    """设置简单的日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def quick_test_baseline(args):
    """快速测试baseline模型"""
    logger = setup_logger()
    logger.info("=" * 50)
    logger.info("快速测试 Baseline 模型（Seq2Seq + Attention）")
    logger.info("=" * 50)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 1. 准备数据（只使用很少的样本）
    logger.info("\n[1/5] 准备数据...")
    data_dir = Path(args.data_dir)
    processed_dir = data_dir.parent / 'processed'

    # 准备数据集（如果需要）
    vocab_path = processed_dir / 'vocab.json'
    if not vocab_path.exists():
        logger.info("词表不存在，开始预处理数据...")
        # =======================================================
        # 🐛 修复 prepare_datasets 调用参数
        # =======================================================
        prepare_datasets(
            raw_dir=str(data_dir),
            processed_dir=str(processed_dir), 
            vocab_path=str(vocab_path),      
            max_vocab_size=args.max_vocab_size,
            min_freq=args.min_freq,
            max_src_len=args.max_src_len,
            max_tgt_len=args.max_tgt_len
        )
        # =======================================================

    # 加载词表
    vocab = Vocab.load(str(vocab_path))
    logger.info(f"词表大小: {len(vocab)}")

    # 加载数据（使用小批量）
    train_loader = get_dataloader(
        processed_dir=str(processed_dir),
        batch_size=args.batch_size,
        split='train',
        shuffle=True
    )

    val_loader = get_dataloader(
        processed_dir=str(processed_dir),
        batch_size=args.batch_size,
        split='val',
        shuffle=False
    )

    # 只使用少量数据
    train_subset_indices = list(range(min(args.num_samples, len(train_loader.dataset))))
    train_subset = Subset(train_loader.dataset, train_subset_indices)
    train_loader_small = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_loader.collate_fn
    )

    val_subset_indices = list(range(min(args.num_samples // 2, len(val_loader.dataset))))
    val_subset = Subset(val_loader.dataset, val_subset_indices)
    val_loader_small = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=val_loader.collate_fn
    )

    logger.info(f"训练样本数: {len(train_subset)}")
    logger.info(f"验证样本数: {len(val_subset)}")

    # 2. 创建模型
    logger.info("\n[2/5] 创建模型...")
    model = Seq2Seq(
        vocab_size=len(vocab),
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_idx=vocab.word2idx['<PAD>']
    ).to(device)

    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 3. 优化器
    logger.info("\n[3/5] 创建优化器...")
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx['<PAD>'])

    # 4. 训练几个epoch
    logger.info(f"\n[4/5] 开始训练 {args.num_epochs} 个epoch...")

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader_small, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)

            optimizer.zero_grad()

            # 前向传播
            output = model(src, tgt[:, :-1])  # teacher forcing

            # 计算损失
            output_flat = output.reshape(-1, output.size(-1))
            tgt_flat = tgt[:, 1:].reshape(-1)
            loss = criterion(output_flat, tgt_flat)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader_small)
        logger.info(f"Epoch {epoch+1} 训练损失: {avg_loss:.4f}")

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader_small:
                src = batch['src'].to(device)
                tgt = batch['tgt'].to(device)

                output = model(src, tgt[:, :-1])
                output_flat = output.reshape(-1, output.size(-1))
                tgt_flat = tgt[:, 1:].reshape(-1)
                loss = criterion(output_flat, tgt_flat)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader_small)
        logger.info(f"Epoch {epoch+1} 验证损失: {avg_val_loss:.4f}")

    # 5. 测试推理
    logger.info("\n[5/5] 测试推理功能...")
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(val_loader_small))
        src = test_batch['src'][:1].to(device)  # 只取一个样本

        # Greedy decoding
        predictions = model.greedy_decode(
            src,
            max_len=50,
            sos_idx=vocab.word2idx['<SOS>'],
            eos_idx=vocab.word2idx['<EOS>']
        )

        # 转换为文本
        pred_tokens = [vocab.idx2word.get(idx.item(), Vocab.UNK_TOKEN)
                      for idx in predictions[0]
                      if idx.item() not in [vocab.word2idx['<PAD>'],
                                           vocab.word2idx['<SOS>'],
                                           vocab.word2idx['<EOS>']]]

        logger.info(f"生成的摘要样例: {' '.join(pred_tokens[:20])}...")

    logger.info("\n" + "=" * 50)
    logger.info("✅ 测试完成！Baseline模型运行正常！")
    logger.info("=" * 50)


def quick_test_pointer_generator(args):
    """快速测试Pointer-Generator模型"""
    logger = setup_logger()
    logger.info("=" * 50)
    logger.info("快速测试 Pointer-Generator 模型")
    logger.info("=" * 50)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 1. 准备数据
    logger.info("\n[1/5] 准备数据...")
    data_dir = Path(args.data_dir)
    processed_dir = data_dir.parent / 'processed'

    vocab_path = processed_dir / 'vocab.json'
    if not vocab_path.exists():
        logger.info("词表不存在，开始预处理数据...")
        from datamodules.cnndm import prepare_datasets
        # =======================================================
        # 🐛 修复 prepare_datasets 调用参数
        # =======================================================
        prepare_datasets(
            raw_dir=str(data_dir),
            processed_dir=str(processed_dir), 
            vocab_path=str(vocab_path),      
            max_vocab_size=args.max_vocab_size,
            min_freq=args.min_freq,
            max_src_len=args.max_src_len,
            max_tgt_len=args.max_tgt_len
        )
        # =======================================================

    # 加载词表
    vocab = Vocab.load(str(vocab_path))
    logger.info(f"词表大小: {len(vocab)}")

    # 加载数据
    train_loader = get_dataloader(
        processed_dir=str(processed_dir),
        batch_size=args.batch_size,
        split='train',
        shuffle=True
    )

    # 只使用少量数据
    train_subset_indices = list(range(min(args.num_samples, len(train_loader.dataset))))
    train_subset = Subset(train_loader.dataset, train_subset_indices)
    train_loader_small = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_loader.collate_fn
    )

    logger.info(f"训练样本数: {len(train_subset)}")

    # 2. 创建Pointer-Generator模型
    logger.info("\n[2/5] 创建Pointer-Generator模型...")
    model = PointerGeneratorSeq2Seq(
        vocab_size=len(vocab),
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_oov_size=args.max_oov_size,
        pad_idx=vocab.word2idx['<PAD>']
    ).to(device)

    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 3. 优化器
    logger.info("\n[3/5] 创建优化器...")
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 4. 训练几个epoch
    logger.info(f"\n[4/5] 开始训练 {args.num_epochs} 个epoch...")

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader_small, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_lens = batch.get('src_lens', None)
            src_oov_map = batch.get('src_oov_map', None)

            optimizer.zero_grad()

            # 前向传播: PG模型返回 1 个输出
            output = model(
                src=src,
                tgt=tgt,
                src_lens=src_lens,
                src_oov_map=src_oov_map
            )

            # 计算损失（处理扩展词表）
            batch_size, seq_len, vocab_size = output.shape
            output_flat = output.reshape(-1, vocab_size)
            tgt_flat = tgt[:, 1:].reshape(-1)

            # 计算负对数似然
            log_probs = torch.log(output_flat + 1e-10)
            target_log_probs = log_probs.gather(1, tgt_flat.unsqueeze(1)).squeeze(1)

            # Mask padding
            mask = (tgt_flat != vocab.word2idx['<PAD>']).float()
            loss = -(target_log_probs * mask).sum() / mask.sum()

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader_small)
        logger.info(f"Epoch {epoch+1} 训练损失: {avg_loss:.4f}")

    # 5. 测试推理
    logger.info("\n[5/5] 测试推理功能...")
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(train_loader_small))
        src = test_batch['src'][:1].to(device)
        src_lens = test_batch.get('src_lens', None)
        if src_lens is not None:
            src_lens = src_lens[:1]
        src_oov_map = test_batch.get('src_oov_map', None)
        if src_oov_map is not None:
             src_oov_map = src_oov_map[:1].to(device)
        oov_lists = test_batch.get('oov_list', [[]])[:1]

        # Greedy decoding
        predictions, _ = model.generate(
            src=src,
            src_lens=src_lens,
            src_oov_map=src_oov_map,
            max_length=50,
            sos_idx=vocab.word2idx['<SOS>'],
            eos_idx=vocab.word2idx['<EOS>'],
        )

        # 转换为文本（处理OOV）
        pred_tokens = []
        for idx in predictions[0]:
            idx_val = idx.item()
            if idx_val < len(vocab):
                token = vocab.idx2word.get(idx_val, Vocab.UNK_TOKEN)
            else:
                # OOV token
                oov_idx = idx_val - len(vocab)
                if oov_idx < len(oov_lists[0]):
                    token = oov_lists[0][oov_idx]
                else:
                    token = Vocab.UNK_TOKEN

            if token not in [Vocab.PAD_TOKEN, Vocab.SOS_TOKEN, Vocab.EOS_TOKEN]:
                pred_tokens.append(token)

        logger.info(f"生成的摘要样例: {' '.join(pred_tokens[:20])}...")

    logger.info("\n" + "=" * 50)
    logger.info("✅ 测试完成！Pointer-Generator模型运行正常！")
    logger.info("=" * 50)


# =======================================================
# 🚀 关键修改 2: 新增 quick_test_pg_coverage 函数
# =======================================================
def quick_test_pg_coverage(args):
    """快速测试Pointer-Generator with Coverage模型"""
    logger = setup_logger()
    logger.info("=" * 50)
    logger.info("快速测试 Pointer-Generator with Coverage 模型")
    logger.info("=" * 50)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 1. 准备数据
    logger.info("\n[1/5] 准备数据...")
    data_dir = Path(args.data_dir)
    processed_dir = data_dir.parent / 'processed'

    vocab_path = processed_dir / 'vocab.json'
    if not vocab_path.exists():
        logger.info("词表不存在，开始预处理数据...")
        from datamodules.cnndm import prepare_datasets
        # =======================================================
        # 🐛 修复 prepare_datasets 调用参数
        # =======================================================
        prepare_datasets(
            raw_dir=str(data_dir),
            processed_dir=str(processed_dir), 
            vocab_path=str(vocab_path),      
            max_vocab_size=args.max_vocab_size,
            min_freq=args.min_freq,
            max_src_len=args.max_src_len,
            max_tgt_len=args.max_tgt_len,
            limit_per_split=100  # 限制数据，否则数据量太大
        )
        # =======================================================

    # 加载词表
    vocab = Vocab.load(str(vocab_path))
    logger.info(f"词表大小: {len(vocab)}")

    # 加载数据
    train_loader = get_dataloader(
        processed_dir=str(processed_dir),
        batch_size=args.batch_size,
        split='train',
        shuffle=True
    )

    # 只使用少量数据
    train_subset_indices = list(range(min(args.num_samples, len(train_loader.dataset))))
    train_subset = Subset(train_loader.dataset, train_subset_indices)
    train_loader_small = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_loader.collate_fn
    )

    logger.info(f"训练样本数: {len(train_subset)}")

    # 2. 创建Pointer-Generator with Coverage模型
    logger.info("\n[2/5] 创建Pointer-Generator with Coverage 模型...")
    # ******* [关键修改 1: 模型类切换] *******
    model = PGCoverageSeq2Seq(
        vocab_size=len(vocab),
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_idx=vocab.word2idx['<PAD>'],
        cov_loss_weight=1.0  # 关键补充：传递覆盖损失权重
    ).to(device)

    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 3. 优化器
    logger.info("\n[3/5] 创建优化器...")
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 4. 训练几个epoch
    logger.info(f"\n[4/5] 开始训练 {args.num_epochs} 个epoch...")

    for epoch in range(args.num_epochs):
        model.train()
        total_nll_loss = 0
        total_cov_loss = 0

        pbar = tqdm(train_loader_small, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_lens = batch.get('src_lens', None)
            src_oov_map = batch.get('src_oov_map', None)

            optimizer.zero_grad()

            # ******* [关键修改 2: 接收 4 个输出] *******
            output, _, _, raw_coverage_loss = model(
                src=src,
                tgt=tgt,
                src_lens=src_lens,
                src_oov_map=src_oov_map,
                teacher_forcing_ratio=1.0 # 快速测试中使用 1.0 简化
            )

            # 计算 NLL 损失（与 PG 模型逻辑相同）
            batch_size, seq_len, vocab_size = output.shape
            output_flat = output.reshape(-1, vocab_size)
            tgt_flat = tgt[:, 1:].reshape(-1)

            log_probs = torch.log(output_flat + 1e-10)
            target_log_probs = log_probs.gather(1, tgt_flat.unsqueeze(1)).squeeze(1)

            mask = (tgt_flat != vocab.word2idx['<PAD>']).float()
            nll_loss = -(target_log_probs * mask).sum() / mask.sum()

            # ******* [关键修改 3: 计算 Total Loss] *******
            # 在快速测试中，我们假设 coverage_loss_weight (lambda) = 1.0
            total_loss = nll_loss + 1.0 * raw_coverage_loss

            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_nll_loss += nll_loss.item()
            total_cov_loss += raw_coverage_loss.item()

            # ******* [关键修改 4: 进度条显示] *******
            pbar.set_postfix({
                'nll': f'{nll_loss.item():.4f}',
                'cov': f'{raw_coverage_loss.item():.4f}',
                'total': f'{total_loss.item():.4f}'
            })

        avg_nll_loss = total_nll_loss / len(train_loader_small)
        avg_cov_loss = total_cov_loss / len(train_loader_small)
        avg_total_loss = avg_nll_loss + avg_cov_loss
        logger.info(f"Epoch {epoch+1} 训练损失: Total={avg_total_loss:.4f} (NLL={avg_nll_loss:.4f}, Cov={avg_cov_loss:.4f})")

    # 5. 测试推理（与PG模型逻辑相同）
    logger.info("\n[5/5] 测试推理功能...")
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(train_loader_small))
        src = test_batch['src'][:1].to(device)
        src_lens = test_batch.get('src_lens', None)
        if src_lens is not None:
            src_lens = src_lens[:1]
        src_oov_map = test_batch.get('src_oov_map', None)
        if src_oov_map is not None:
             src_oov_map = src_oov_map[:1].to(device)

        oov_lists = test_batch.get('oov_list', [[]])[:1]

        # Greedy decoding
        # PGCoverageSeq2Seq 的 generate 方法应自动处理 Coverage 向量
        predictions, _ = model.generate(
            src=src,
            src_lens=src_lens,
            src_oov_map=src_oov_map,
            max_length=50,
            sos_idx=vocab.word2idx['<SOS>'],
            eos_idx=vocab.word2idx['<EOS>'],
        )

        # 转换为文本（处理OOV）
        pred_tokens = []
        for idx in predictions[0]:
            idx_val = idx.item()
            if idx_val < len(vocab):
                token = vocab.idx2word.get(idx_val, Vocab.UNK_TOKEN)
            else:
                oov_idx = idx_val - len(vocab)
                if oov_idx < len(oov_lists[0]):
                    token = oov_lists[0][oov_idx]
                else:
                    token = Vocab.UNK_TOKEN

            if token not in [Vocab.PAD_TOKEN, Vocab.SOS_TOKEN, Vocab.EOS_TOKEN]:
                pred_tokens.append(token)

        logger.info(f"生成的摘要样例: {' '.join(pred_tokens[:20])}...")

    logger.info("\n" + "=" * 50)
    logger.info("✅ 测试完成！Pointer-Generator with Coverage 模型运行正常！")
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='快速测试训练脚本')

    # 模型选择
    # 🚀 关键修改 3: 增加 'pg_cov' 选项
    parser.add_argument('--model', type=str, default='baseline',
                       choices=['baseline', 'pg', 'pg_cov'],
                       help='选择模型: baseline, pg (pointer-generator) 或 pg_cov (PG with Coverage)')

    # 数据参数
    # ⭐⭐⭐ 关键修复: 将默认值从 '../data/raw' 更改为 './data/raw' ⭐⭐⭐
    # parser.add_argument('--data_dir', type=str, default='../data/raw',
    #                    help='原始数据目录')
    parser.add_argument('--data_dir', type=str, default=str(default_data_dir),
                    help='原始数据目录 (自动定位)')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='使用的训练样本数量（默认100）')
    parser.add_argument('--max_vocab_size', type=int, default=50000)
    parser.add_argument('--min_freq', type=int, default=5)
    parser.add_argument('--max_src_len', type=int, default=400)
    parser.add_argument('--max_tgt_len', type=int, default=100)

    # 模型参数
    parser.add_argument('--embed_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_oov_size', type=int, default=1000,
                       help='Pointer-Generator最大OOV词汇数')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=2,
                       help='训练轮数（默认2，只为验证）')
    parser.add_argument('--learning_rate', type=float, default=1e-4)

    args = parser.parse_args()

    # 根据模型类型调用相应的测试函数
    # 🚀 关键修改 4: 增加 'pg_cov' 调用逻辑
    if args.model == 'baseline':
        quick_test_baseline(args)
    elif args.model == 'pg':
        quick_test_pointer_generator(args)
    elif args.model == 'pg_cov':
        quick_test_pg_coverage(args)
    else:
        # Should not happen due to argparse choices
        pass


if __name__ == '__main__':
    main()
