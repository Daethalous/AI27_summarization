"""
Transformer+Pointer-Generator+Coverage (PGCT_layer) 模型正式训练脚本
支持定期保存 checkpoint，新增配置文件参数加载（优先级：命令行>配置文件>默认值）
"""
from __future__ import annotations
import sys
from pathlib import Path
import logging
from typing import Optional, List
import argparse
import yaml

import torch
import torch.nn as nn  # [NEW] 引入 nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# [MODIFIED] 移除 torch.optim as optim，改用 AdamW
from torch.optim import AdamW
# -------------------------------------------------------------------------
# [MODIFIED] 引入学习率调度器：LambdaLR 用于自定义 Warmup+Cosine
from torch.optim.lr_scheduler import LambdaLR

# [MODIFIED] 移除 from torch.optim.lr_scheduler import ReduceLROnPlateau
# -------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent.parent))

from datamodules.cnndm import prepare_datasets, get_dataloader
from utils.vocab import Vocab
# -------------------------------------------------------------------------
from models.pgct_layer.pgct_layer_model import PGCT_layer_Model
from models.pgct_layer.pgct_decoding import pgct_greedy_decode

# -------------------------------------------------------------------------

try:
    from utils.metrics import compute_rouge

    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False


def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


# [NEW] Warmup + Cosine Annealing 调度器函数
def get_optimizer_and_scheduler(model: nn.Module, args: argparse.Namespace, total_steps: int):
    """
    初始化 AdamW 优化器和 Warmup + Cosine Annealing 学习率调度器。
    """
    base_lr = args.learning_rate
    weight_decay = args.weight_decay
    warmup_steps = args.warmup_steps

    # 1. 初始化优化器 (AdamW)
    optimizer = AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay
    )

    # 2. 定义学习率调度函数 (lr_lambda)
    def lr_lambda(step):
        step = max(step, 0)

        # a. Warmup 阶段: 线性增加
        if step < warmup_steps:
            # 学习率从 0 线性增加到 1.0 (即 base_lr)
            return float(step) / float(warmup_steps)

        # b. Cosine Annealing 阶段: 余弦退火衰减
        decay_steps = total_steps - warmup_steps
        current_decay_step = step - warmup_steps

        # 如果训练已经超过总步数或衰减步数不合理
        if current_decay_step >= decay_steps or decay_steps <= 0:
            return 0.0

        # 计算衰减进度 (0.0 到 1.0)
        progress = float(current_decay_step) / float(decay_steps)

        # 余弦衰减公式: 0.5 * (1 + cos(pi * progress))
        return 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)))

    # 3. 初始化调度器
    # LambdaLR 使用 lr_lambda 函数来计算乘数因子，乘以 base_lr 得到实际学习率
    scheduler = LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def calculate_nll_loss(predictions: torch.Tensor, targets: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    计算负对数似然损失 (NLL)。
    此函数现在必须能够处理 targets 中的扩展词表索引 (OOV 词)。
    """
    # predictions: [B, T_out, V_ext] (V_ext = vocab_size + max_oov_size)
    # targets: [B, T_out] (T_out = T - 1, 因为目标序列移位了，targets 包含扩展索引)
    B, T, V = predictions.shape
    preds_flat = predictions.reshape(-1, V)
    targs_flat = targets.reshape(-1)

    # 使用 log(P) 以确保数值稳定性
    log_probs = torch.log(preds_flat + 1e-12)  # 避免 log(0)

    # 针对目标索引 targs_flat 收集对应的 log 概率
    # targs_flat 的值可以大于 vocab_size (对应 OOV 词)
    picked = log_probs.gather(1, targs_flat.unsqueeze(1)).squeeze(1)

    # 计算有效词的掩码 (非 PAD 词)
    mask = (targs_flat != pad_idx).float()

    # NLL 损失: -log(P) 的平均值
    loss = -(picked * mask).sum() / mask.sum()
    return loss


def generate_val_summaries(model, val_loader, vocab, device, max_tgt_len):
    """生成验证集摘要（用于计算 ROUGE）"""
    model.eval()
    generated_summaries = []
    reference_summaries = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="生成验证集摘要"):
            src = batch['src'].to(device)
            src_len = batch['src_len'].to(device)
            src_oov_map = batch['src_oov_map'].to(device)
            oov_dicts = batch['oov_dicts']  # 每个样本的 OOV 词映射
            references = batch['tgt_text']  # 参考摘要文本

            # 贪心解码生成摘要
            pred_ids, _ = pgct_greedy_decode(
                model=model,
                src=src,
                src_lens=src_len,
                src_oov_map=src_oov_map,
                max_length=max_tgt_len,
                sos_idx=vocab.sos_idx,
                eos_idx=vocab.eos_idx,
                device=device
            )

            # 将预测索引转换为文本（处理 OOV）
            for i in range(len(pred_ids)):
                pred_tokens = []
                oov_dict = oov_dicts[i]  # 当前样本的 OOV 映射
                for idx in pred_ids[i].tolist():
                    if idx < len(vocab):
                        # 使用 vocab.UNK_TOKEN (如果存在)，否则默认为 '<unk>'
                        token = vocab.idx2word.get(idx, vocab.UNK_TOKEN)
                    else:
                        oov_rel_idx = idx - len(vocab)
                        token = oov_dict.get(oov_rel_idx, vocab.UNK_TOKEN)
                    if token == vocab.EOS_TOKEN:
                        break  # 遇到 EOS 停止
                    if token not in [vocab.PAD_TOKEN, vocab.SOS_TOKEN]:
                        pred_tokens.append(token)
                generated_summaries.append(' '.join(pred_tokens))
                reference_summaries.append(references[i])

    return generated_summaries, reference_summaries


def main():
    parser = argparse.ArgumentParser()
    # 新增：添加 --config 参数，用于指定YAML配置文件路径
    parser.add_argument("--config", type=str, help="YAML配置文件路径（例如 ../configs/pgct_layer.yaml）")
    parser.add_argument(
        "--resume_ckpt_path",
        type=str,
        default=None,
        help="用于恢复训练的 Checkpoint 文件路径"
    )
    # 原有参数保留，默认值将作为最低优先级
    parser.add_argument("--data_dir", type=str, default="../data/raw")
    parser.add_argument("--save_dir", type=str, default="../checkpoints_pgct_layer")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--embed_size", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cov_loss_weight", type=float, default=1.0)
    parser.add_argument("--max_src_len", type=int, default=400)
    parser.add_argument("--max_tgt_len", type=int, default=100)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    # [NEW] 优化器和调度器参数
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW 权重衰减")
    parser.add_argument("--warmup_steps", type=int, default=4000, help="Warmup 步数")
    # -------------------------------------------------------------------------
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--save_every", type=int, default=2, help="隔多少个 epoch 保存一次 checkpoint")
    parser.add_argument("--num_samples", type=int, default=None, help="限制训练集使用的样本数量 (None表示使用全部)")
    # -------------------------------------------------------------------------
    args = parser.parse_args()

    logger = setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # -------------------------------------------------------------------------
    # 新增：加载YAML配置文件（如果指定）
    config = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"指定的配置文件不存在: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"✅  成功加载配置文件: {args.config}")
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # 新增：参数优先级处理（命令行 > 配置文件 > 脚本默认值）
    data_config = config.get("data", {})
    # ... (数据参数加载)
    args.data_dir = args.data_dir or data_config.get("data_dir", "../data/raw")
    args.max_src_len = args.max_src_len or data_config.get("max_src_len", 400)
    args.max_tgt_len = args.max_tgt_len or data_config.get("max_tgt_len", 100)
    # [FIX] 强制转换 min_freq 和 max_vocab_size 为整数，并保持配置文件优先级

    # 1. 处理 max_vocab_size
    config_mvs = data_config.get("max_vocab_size")
    if config_mvs is not None:
        try:
            # 尝试将配置文件中的值转换为 int
            max_vocab_size = int(config_mvs)
        except (TypeError, ValueError):
            # 如果转换失败（值是无效字符串），则回退到硬编码默认值
            logger.warning(f"配置文件中的 max_vocab_size: '{config_mvs}' 无效，使用默认值 50000。")
            max_vocab_size = 50000
    else:
        # 如果配置文件中未设置该键，则使用默认值
        max_vocab_size = 50000

    # 2. 处理 min_freq
    config_mf = data_config.get("min_freq")
    if config_mf is not None:
        try:
            # 尝试将配置文件中的值转换为 int
            min_freq = int(config_mf)
        except (TypeError, ValueError):
            # 如果转换失败（值是无效字符串），则回退到硬编码默认值
            logger.warning(f"配置文件中的 min_freq: '{config_mf}' 无效，使用默认值 5。")
            min_freq = 5
    else:
        # 如果配置文件中未设置该键，则使用默认值
        min_freq = 5

    model_config = config.get("model", {})
    # ... (模型参数加载)
    args.embed_size = args.embed_size or model_config.get("embed_size", 512)
    args.hidden_size = args.hidden_size or model_config.get("hidden_size", 512)
    args.num_encoder_layers = args.num_encoder_layers or model_config.get("num_encoder_layers", 3)
    args.num_decoder_layers = args.num_decoder_layers or model_config.get("num_decoder_layers", 3)
    args.nhead = args.nhead or model_config.get("nhead", 8)
    args.dropout = args.dropout or model_config.get("dropout", 0.1)
    args.cov_loss_weight = args.cov_loss_weight or model_config.get("cov_loss_weight", 1.0)

    train_config = config.get("train", {})
    args.save_dir = args.save_dir or train_config.get("save_dir", "../checkpoints_pgct_layer")
    args.num_epochs = args.num_epochs or train_config.get("num_epochs", 10)
    args.batch_size = args.batch_size or train_config.get("batch_size", 8)
    args.learning_rate = args.learning_rate or train_config.get("learning_rate", 1e-4)
    args.teacher_forcing_ratio = args.teacher_forcing_ratio or train_config.get("teacher_forcing_ratio", 0.5)
    args.grad_clip = args.grad_clip or train_config.get("grad_clip", 5.0)
    args.save_every = args.save_every or train_config.get("save_every", 2)
    args.num_samples = args.num_samples or train_config.get("num_samples", None)

    # [NEW] 从配置或命令行加载新的优化器/调度器参数
    args.weight_decay = args.weight_decay or train_config.get("weight_decay", 0.01)
    args.warmup_steps = args.warmup_steps or train_config.get("warmup_steps", 4000)
    # -------------------------------------------------------------------------

    # 打印最终生效的核心参数（方便验证优先级）
    logger.info(f"🔧  最终生效的核心参数:")
    logger.info(f"  - 模型参数: hidden_size={args.hidden_size}, embed_size={args.embed_size}, nhead={args.nhead}")
    logger.info(
        f"  - 训练参数: batch_size={args.batch_size}, num_epochs={args.num_epochs}, peak_lr={args.learning_rate}, warmup_steps={args.warmup_steps}, weight_decay={args.weight_decay}")  # [MODIFIED]
    logger.info(
        f"  - 数据参数: max_src_len={args.max_src_len}, max_tgt_len={args.max_tgt_len}, max_vocab_size={max_vocab_size}")
    # -------------------------------------------------------------------------

    # 数据预处理
    # ... (数据加载和词表初始化保持不变)
    data_dir = Path(args.data_dir)
    processed_dir = data_dir.parent / "processed"
    vocab_path = data_config.get("vocab_path", processed_dir / "vocab.json")
    vocab_path = Path(vocab_path)

    logger.warning(f"确保 {processed_dir} 中的缓存文件包含 PG 所需的原始 tokens，若没有，将重新生成数据...")
    prepare_datasets(
        str(data_dir),
        str(processed_dir),
        str(vocab_path),
        max_vocab_size=max_vocab_size,
        min_freq=min_freq,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len
    )

    vocab = Vocab.load(str(vocab_path))
    pad_idx = vocab.pad_idx
    sos_idx = vocab.sos_idx
    eos_idx = vocab.eos_idx
    logger.info(f"词表已加载: {len(vocab)} 个词")

    full_train_loader = get_dataloader(
        str(processed_dir),
        batch_size=args.batch_size,
        split="train",
        shuffle=True,
        vocab=vocab,
        include_oov=True
    )

    if args.num_samples is not None and args.num_samples < len(full_train_loader.dataset):
        indices = list(range(args.num_samples))
        subset_dataset = Subset(full_train_loader.dataset, indices)
        train_loader = DataLoader(
            subset_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=full_train_loader.collate_fn,
            num_workers=full_train_loader.num_workers
        )
        logger.info(f"🚧  限制训练集大小为 {args.num_samples} 个样本。")
    else:
        train_loader = full_train_loader

    val_loader = get_dataloader(
        str(processed_dir),
        batch_size=args.batch_size,
        split="val",
        shuffle=False,
        vocab=vocab,
        include_oov=True
    )

    # 模型
    model = PGCT_layer_Model(
        vocab_size=len(vocab),
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        nhead=args.nhead,
        dropout=args.dropout,
        pad_idx=pad_idx,
        cov_loss_weight=args.cov_loss_weight,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len
    ).to(device)
    logger.info("PGCT_layer_Model 初始化完成")

    # 创建保存目录
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # [NEW] 计算总步数
    total_steps = len(train_loader) * args.num_epochs

    # [MODIFIED] 使用新的函数初始化 AdamW 优化器和 Warmup+Cosine 调度器
    optimizer, scheduler = get_optimizer_and_scheduler(model, args, total_steps)  # [MODIFIED]

    tb_writer = SummaryWriter(log_dir=Path(args.save_dir) / "runs")

    start_epoch = 1
    best_val_loss = float("inf")
    best_rouge_l = -float("inf")

    # [NEW/MODIFIED] Checkpoint 恢复机制
    if args.resume_ckpt_path and Path(args.resume_ckpt_path).exists():
        ckpt_path = Path(args.resume_ckpt_path)
        logger.info(f"💾  尝试从 Checkpoint 恢复训练: {ckpt_path}")

        try:
            checkpoint = torch.load(ckpt_path, map_location=device)

            # 恢复模型和优化器状态
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # 恢复训练进度
            start_epoch = checkpoint['epoch'] + 1
            # 恢复最佳指标
            best_val_loss = checkpoint.get('best_val_loss', checkpoint.get('avg_val_loss', float("inf")))
            best_rouge_l = checkpoint.get('best_rouge_l', -float("inf"))

            # [NEW] 恢复调度器进度 (Warmup/Cosine 必须快进)
            steps_per_epoch = len(train_loader)
            steps_completed = (start_epoch - 1) * steps_per_epoch

            # 使用 scheduler.step() 快进到正确的位置
            for _ in range(steps_completed):
                scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                f"✅  成功恢复！从 Epoch {start_epoch} 开始训练。当前最佳 Val Loss: {best_val_loss:.4f}。当前 LR: {current_lr:.6e}")

        except Exception as e:
            logger.error(f"❌  Checkpoint 加载失败: {e}. 将从 Epoch 1 重新开始。")
            start_epoch = 1

    # [NEW] 初始化当前训练步数，用于 Warmup/Cosine 调度
    current_step = (start_epoch - 1) * len(train_loader)

    # [MODIFIED] 循环从 start_epoch 开始
    for epoch in range(start_epoch, args.num_epochs + 1):
        model.train()
        running_nll = 0.0
        running_cov = 0.0
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}/{args.num_epochs}")

        for batch in pbar:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_oov_map = batch['src_oov_map'].to(device)

            optimizer.zero_grad()

            outputs, _, _, coverage_loss = model(
                src,
                tgt,
                src_oov_map=src_oov_map,
                teacher_forcing_ratio=args.teacher_forcing_ratio
            )

            nll_loss = calculate_nll_loss(outputs, tgt[:, 1:], pad_idx)
            total_loss = nll_loss + coverage_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            # [MODIFIED] 每步更新优化器
            optimizer.step()

            # [NEW] 每步更新学习率调度器 (Warmup/Cosine)
            scheduler.step()
            current_step += 1  # 更新全局步数

            running_nll += nll_loss.item()
            running_cov += coverage_loss.item()

            # [MODIFIED] 使用调度器获取的最新 LR
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix(
                {"NLL": f"{nll_loss.item():.4f}", "Cov": f"{coverage_loss.item():.4f}", "LR": f"{current_lr:.6e}"})

        avg_train_nll = running_nll / len(train_loader)
        avg_train_cov = running_cov / len(train_loader)
        avg_train_total = avg_train_nll + avg_train_cov
        logger.info(
            f"Epoch {epoch} Train Loss: Total={avg_train_total:.4f} (NLL={avg_train_nll:.4f}, Cov={avg_train_cov:.4f})")
        tb_writer.add_scalar("Train/TotalLoss", avg_train_total, epoch)

        # 验证
        model.eval()
        val_total_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Val Epoch {epoch}/{args.num_epochs}")
            for batch in val_pbar:
                src = batch['src'].to(device)
                tgt = batch['tgt'].to(device)
                src_oov_map = batch['src_oov_map'].to(device)

                outputs, _, _, coverage_loss = model(
                    src,
                    tgt,
                    src_oov_map=src_oov_map,
                    teacher_forcing_ratio=1.0
                )

                nll_loss = calculate_nll_loss(outputs, tgt[:, 1:], pad_idx)
                val_total_loss += (nll_loss + coverage_loss).item()

        avg_val_loss = val_total_loss / len(val_loader)
        logger.info(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f}")
        tb_writer.add_scalar("Val/TotalLoss", avg_val_loss, epoch)

        # [MODIFIED] 移除基于指标的调度器步进 (Warmup/Cosine 是基于 step 的，已在训练循环中完成)
        # scheduler.step(avg_val_loss)

        # ROUGE 指标计算
        if HAS_ROUGE and epoch % 2 == 0:
            logger.info("开始计算验证集 ROUGE 指标...")
            generated, references = generate_val_summaries(
                model=model,
                val_loader=val_loader,
                vocab=vocab,
                device=device,
                max_tgt_len=args.max_tgt_len
            )
            # 计算 ROUGE 分数
            rouge_scores = compute_rouge(generated, references)
            rouge1 = rouge_scores.get('rouge1_f', 0.0) * 100
            rouge2 = rouge_scores.get('rouge2_f', 0.0) * 100
            val_rouge_l = rouge_scores.get('rougeL_f', 0.0) * 100
            logger.info(
                "Epoch %d Val ROUGE-1: %.2f | ROUGE-2: %.2f | ROUGE-L: %.2f",
                epoch,
                rouge1,
                rouge2,
                val_rouge_l
            )
            tb_writer.add_scalar("Val/ROUGE-L", val_rouge_l, epoch)

            # 保存基于 ROUGE 的最佳模型
            if val_rouge_l > best_rouge_l:
                best_rouge_l = val_rouge_l
                torch.save(model.state_dict(), Path(args.save_dir) / "best_rouge_model.pt")
                logger.info(f"✨  新最佳 ROUGE 模型保存: {args.save_dir}/best_rouge_model.pt")
        elif not HAS_ROUGE:
            logger.warning("跳过 ROUGE 计算，因为 'rouge' 库未导入或 utils/metrics 缺失。")

        # 保存基于损失的最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), Path(args.save_dir) / "best_model.pt")
            logger.info(f"✨  新最佳损失模型保存: {args.save_dir}/best_model.pt")

        # 定期保存 checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = Path(args.save_dir) / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                # [MODIFIED] 保存最佳指标，方便恢复
                "best_val_loss": best_val_loss,
                "best_rouge_l": best_rouge_l,
                # "scheduler_state_dict": scheduler.state_dict(), # LambdaLR 可选，为简化不保存
                "config": {
                    "model": model_config,
                    "train": train_config,
                    "data": data_config
                }
            }, ckpt_path)
            logger.info(f"💾  定期保存模型 checkpoint: {ckpt_path}")

    tb_writer.close()
    logger.info("✅  正式训练完成！")
    logger.info(f"最佳验证损失: {best_val_loss:.4f}")
    if HAS_ROUGE:
        logger.info(f"最佳验证 ROUGE-L: {best_rouge_l:.2f}")


if __name__ == "__main__":
    # 示例运行命令: 
    # 从头开始: python train_pgct_layer.py --config ../configs/pgct_layer.yaml
    # 恢复训练: python train_pgct_layer.py --resume_ckpt_path ../checkpoints_pgct_layer/checkpoint_epoch_4.pt
    main()