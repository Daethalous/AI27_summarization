"""
快速测试训练脚本 - 验证 Transformer+Pointer-Generator+Coverage (PGCT) 模型

使用方法:
    python quick_test_train_pgct.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))
project_root = Path(__file__).resolve().parents[1]
default_data_dir = project_root / 'data' / 'raw'


from datamodules.cnndm import prepare_datasets, get_dataloader
from models.pointer_generator_coverage.pgct_model import PGCTModel  # 导入PGCT模型
from utils.vocab import Vocab


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def quick_test_pgct(args):
    """快速    快速测试 Transformer+Pointer-Generator+Coverage (PGCT) 模型
    """
    logger = setup_logger()
    logger.info("=" * 50)
    logger.info("快速测试 Transformer+Pointer-Generator+Coverage (PGCT) 模型")
    logger.info("=" * 50)

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 1. 准备数据
    logger.info("\n[1/5] 准备数据...")
    data_dir = Path(args.data_dir)
    processed_dir = data_dir.parent / 'processed'

    vocab_path = processed_dir / 'vocab.json'
    if not vocab_path.exists():
        logger.info("词表不存在，开始预处理数据...")
        prepare_datasets(
            raw_dir=str(data_dir),
            processed_dir=str(processed_dir),
            vocab_path=str(vocab_path),
            max_vocab_size=args.max_vocab_size,
            min_freq=args.min_freq,
            max_src_len=args.max_src_len,
            max_tgt_len=args.max_tgt_len,
            limit_per_split=100  # 限制数据量，加速测试
        )

    # 加载词表
    vocab = Vocab.load(str(vocab_path))
    logger.info(f"词表大小: {len(vocab)}")

    # 加载数据加载器
    train_loader = get_dataloader(
        processed_dir=str(processed_dir),
        batch_size=args.batch_size,
        split='train',
        shuffle=True
    )

    # 只使用少量样本进行测试
    train_subset_indices = list(range(min(args.num_samples, len(train_loader.dataset))))
    train_subset = Subset(train_loader.dataset, train_subset_indices)
    train_loader_small = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_loader.collate_fn
    )

    logger.info(f"训练样本数: {len(train_subset)}")

    # 2. 创建PGCT模型
    logger.info("\n[2/5] 创建PGCT模型...")
    model = PGCTModel(
        vocab_size=len(vocab),
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        nhead=args.nhead,  # Transformer多头注意力头数
        dropout=args.dropout,
        pad_idx=vocab.word2idx['<PAD>'],
        cov_loss_weight=args.cov_loss_weight,  # 覆盖损失权重
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len
    ).to(device)

    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 3. 优化器
    logger.info("\n[3/5] 创建优化器...")
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 4. 训练几个epoch
    logger.info(f"\n[4/5] 开始训练 {args.num_epochs} 个epoch...")

    for epoch in range(args.num_epochs):
        model.train()
        total_nll_loss = 0.0
        total_cov_loss = 0.0

        pbar = tqdm(train_loader_small, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_lens = batch.get('src_lens', None)
            src_oov_map = batch.get('src_oov_map', None)
            if src_oov_map is not None:
                src_oov_map = src_oov_map.to(device)

            optimizer.zero_grad()

            # 前向传播：PGCT模型返回 (输出分布, None, None, 覆盖损失)
            output, _, _, coverage_loss = model(
                src=src,
                tgt=tgt,
                src_lens=src_lens,
                src_oov_map=src_oov_map,
                teacher_forcing_ratio=1.0  # 测试阶段使用完全教师强制
            )

            # 计算负对数似然损失（NLL Loss）
            batch_size, seq_len, vocab_size = output.shape
            output_flat = output.reshape(-1, vocab_size)  # [batch*seq_len, vocab_size]
            tgt_flat = tgt[:, 1:].reshape(-1)  # 目标序列右移一位 [batch*seq_len]

            # 计算对数概率（加小epsilon避免log(0)）
            log_probs = torch.log(output_flat + 1e-10)
            # 提取目标token对应的概率
            target_log_probs = log_probs.gather(1, tgt_flat.unsqueeze(1)).squeeze(1)

            # 屏蔽PAD token
            mask = (tgt_flat != vocab.word2idx['<PAD>']).float()
            nll_loss = -(target_log_probs * mask).sum() / mask.sum()  # 平均每个有效token的损失

            # 总损失 = NLL损失 + 覆盖损失（已加权）
            total_loss = nll_loss + coverage_loss

            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 梯度裁剪
            optimizer.step()

            # 累计损失
            total_nll_loss += nll_loss.item()
            total_cov_loss += coverage_loss.item()

            # 进度条显示
            pbar.set_postfix({
                'nll': f'{nll_loss.item():.4f}',
                'cov': f'{coverage_loss.item():.4f}',
                'total': f'{total_loss.item():.4f}'
            })

        # 计算平均损失
        avg_nll = total_nll_loss / len(train_loader_small)
        avg_cov = total_cov_loss / len(train_loader_small)
        avg_total = avg_nll + avg_cov
        logger.info(
            f"Epoch {epoch+1} 训练损失: "
            f"Total={avg_total:.4f} (NLL={avg_nll:.4f}, Cov={avg_cov:.4f})"
        )

    # 5. 测试推理功能
    logger.info("\n[5/5] 测试推理功能...")
    model.eval()
    with torch.no_grad():
        # 取一个测试样本
        test_batch = next(iter(train_loader_small))
        src = test_batch['src'][:1].to(device)  # [1, src_len]
        src_lens = test_batch.get('src_lens', None)
        if src_lens is not None:
            src_lens = src_lens[:1]
        src_oov_map = test_batch.get('src_oov_map', None)
        if src_oov_map is not None:
            src_oov_map = src_oov_map[:1].to(device)
        oov_lists = test_batch.get('oov_list', [[]])[:1]  # OOV词列表

        # 贪婪解码
        predictions, _ = model.generate(
            src=src,
            src_lens=src_lens,
            src_oov_map=src_oov_map,
            max_length=50,
            sos_idx=vocab.word2idx['<SOS>'],
            eos_idx=vocab.word2idx['<EOS>'],
            device=device
        )

        # 转换为文本（处理OOV）
        pred_tokens = []
        for idx in predictions[0]:
            idx_val = idx.item()
            if idx_val < len(vocab):
                # 常规词
                token = vocab.idx2word.get(idx_val, Vocab.UNK_TOKEN)
            else:
                # OOV词（索引 = 词表大小 + OOV在源文中的索引）
                oov_idx = idx_val - len(vocab)
                if oov_idx < len(oov_lists[0]):
                    token = oov_lists[0][oov_idx]
                else:
                    token = Vocab.UNK_TOKEN

            # 过滤特殊符号
            if token not in [Vocab.PAD_TOKEN, Vocab.SOS_TOKEN, Vocab.EOS_TOKEN]:
                pred_tokens.append(token)

        logger.info(f"生成的摘要样例: {' '.join(pred_tokens[:20])}...")

    logger.info("\n" + "=" * 50)
    logger.info("✅ 测试完成！PGCT模型运行正常！")
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='快速测试PGCT模型训练')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default=str(default_data_dir),
                       help='原始数据目录')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='使用的训练样本数量')
    parser.add_argument('--max_vocab_size', type=int, default=50000)
    parser.add_argument('--min_freq', type=int, default=5)
    parser.add_argument('--max_src_len', type=int, default=400)
    parser.add_argument('--max_tgt_len', type=int, default=100)

    # 模型参数（Transformer特有）
    parser.add_argument('--embed_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=256)  # Transformer的d_model
    parser.add_argument('--num_encoder_layers', type=int, default=2)
    parser.add_argument('--num_decoder_layers', type=int, default=2)
    parser.add_argument('--nhead', type=int, default=4)  # 多头注意力头数
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--cov_loss_weight', type=float, default=1.0,
                       help='覆盖损失权重')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-4)

    args = parser.parse_args()
    quick_test_pgct(args)


if __name__ == '__main__':
    main()
