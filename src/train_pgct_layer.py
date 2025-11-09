"""
Transformer+Pointer-Generator+Coverage (PGCT) 模型正式训练脚本
支持定期保存 checkpoint，新增配置文件参数加载（优先级：命令行>配置文件>默认值）
"""
from __future__ import annotations
import sys
from pathlib import Path
import logging
from typing import Optional, List
import argparse
import yaml  # 新增：导入yaml库用于读取配置文件

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.optim as optim
# -------------------------------------------------------------------------
# 引入学习率调度器
from torch.optim.lr_scheduler import ReduceLROnPlateau 
# -------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent.parent))

from datamodules.cnndm import prepare_datasets, get_dataloader
from utils.vocab import Vocab
# -------------------------------------------------------------------------
# 关键改动 1: 引入 PGCT_layer_Model 和解码函数
from models.pgct_layer.pgct_layer_model import PGCT_layer_Model 
from models.pgct_layer.pgct_decoding import pgct_greedy_decode  # 用于生成验证集摘要
# -------------------------------------------------------------------------

try:
    from utils.metrics import compute_rouge
    HAS_ROUGE = True
except ImportError:
    # 如果 utils.metrics 不存在或 ROUGE 库缺失，则设置 HAS_ROUGE 为 False
    HAS_ROUGE = False


def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


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
    log_probs = torch.log(preds_flat + 1e-12) # 避免 log(0)
    
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
                        token = vocab.idx2word.get(idx, vocab.UNK_TOKEN) #报错Vocab' object has no attribute 'unk_token'
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
    parser.add_argument("--config", type=str, help="YAML配置文件路径（例如 ../configs/pgct.yaml）")
    
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
        logger.info(f"✅ 成功加载配置文件: {args.config}")
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # 新增：参数优先级处理（命令行 > 配置文件 > 脚本默认值）
    # 数据相关参数（对应配置文件data字段）
    data_config = config.get("data", {})
    args.data_dir = args.data_dir or data_config.get("data_dir", "../data/raw")
    args.max_src_len = args.max_src_len or data_config.get("max_src_len", 400)
    args.max_tgt_len = args.max_tgt_len or data_config.get("max_tgt_len", 100)
    max_vocab_size = data_config.get("max_vocab_size", 50000)  # 词表参数单独提取
    min_freq = data_config.get("min_freq", 5)
    
    # 模型相关参数（对应配置文件model字段）
    model_config = config.get("model", {})
    args.embed_size = args.embed_size or model_config.get("embed_size", 512)
    args.hidden_size = args.hidden_size or model_config.get("hidden_size", 512)  # 优先读取配置文件的hidden_size
    args.num_encoder_layers = args.num_encoder_layers or model_config.get("num_encoder_layers", 3)
    args.num_decoder_layers = args.num_decoder_layers or model_config.get("num_decoder_layers", 3)
    args.nhead = args.nhead or model_config.get("nhead", 8)
    args.dropout = args.dropout or model_config.get("dropout", 0.1)
    args.cov_loss_weight = args.cov_loss_weight or model_config.get("cov_loss_weight", 1.0)
    use_layer_attention = model_config.get("use_layer_attention", True)
    
    # 训练相关参数（对应配置文件train字段）
    train_config = config.get("train", {})
    args.save_dir = args.save_dir or train_config.get("save_dir", "../checkpoints_pgct")
    args.num_epochs = args.num_epochs or train_config.get("num_epochs", 10)
    args.batch_size = args.batch_size or train_config.get("batch_size", 8)
    args.learning_rate = args.learning_rate or train_config.get("learning_rate", 1e-4)
    args.teacher_forcing_ratio = args.teacher_forcing_ratio or train_config.get("teacher_forcing_ratio", 0.5)
    args.grad_clip = args.grad_clip or train_config.get("grad_clip", 5.0)
    args.save_every = args.save_every or train_config.get("save_every", 2)
    args.num_samples = args.num_samples or train_config.get("num_samples", None)
    
    # 打印最终生效的核心参数（方便验证优先级）
    logger.info(f"🔧 最终生效的核心参数:")
    logger.info(f"  - 模型参数: hidden_size={args.hidden_size}, embed_size={args.embed_size}, nhead={args.nhead}")
    logger.info(f"  - 训练参数: batch_size={args.batch_size}, num_epochs={args.num_epochs}, lr={args.learning_rate}")
    logger.info(f"  - 数据参数: max_src_len={args.max_src_len}, max_tgt_len={args.max_tgt_len}, max_vocab_size={max_vocab_size}")
    # -------------------------------------------------------------------------

    # 数据预处理
    data_dir = Path(args.data_dir)
    processed_dir = data_dir.parent / "processed"
    # 优先从配置文件读取词表路径（如果有）
    vocab_path = data_config.get("vocab_path", processed_dir / "vocab.json")
    vocab_path = Path(vocab_path)
    
    # 始终运行 prepare_datasets，以确保 PG 兼容的原始 tokens 被保存
    logger.warning(f"确保 {processed_dir} 中的缓存文件包含 PG 所需的原始 tokens，若没有，将重新生成数据...")
    prepare_datasets(
        str(data_dir), 
        str(processed_dir), 
        str(vocab_path),
        max_vocab_size=max_vocab_size,  # 使用配置文件中的词表大小
        min_freq=min_freq,              # 使用配置文件中的最小词频
        max_src_len=args.max_src_len, 
        max_tgt_len=args.max_tgt_len
    )

    vocab = Vocab.load(str(vocab_path))
    pad_idx = vocab.pad_idx
    sos_idx = vocab.sos_idx
    eos_idx = vocab.eos_idx
    logger.info(f"词表已加载: {len(vocab)} 个词")

    # -------------------------------------------------------------------------
    # 为 PG 模型加载 DataLoader 时，必须传递 vocab 并设置 include_oov=True
    full_train_loader = get_dataloader(
        str(processed_dir), 
        batch_size=args.batch_size, 
        split="train", 
        shuffle=True, 
        vocab=vocab,
        include_oov=True # 启用 PG 机制
    )
    
    if args.num_samples is not None and args.num_samples < len(full_train_loader.dataset):
        # 创建一个数据集子集
        indices = list(range(args.num_samples))
        subset_dataset = Subset(full_train_loader.dataset, indices)
        # 直接使用 full_train_loader.collate_fn（PGCollateFn 实例）
        train_loader = DataLoader(
            subset_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=full_train_loader.collate_fn,
            num_workers=full_train_loader.num_workers
        )
        logger.info(f"🚧 限制训练集大小为 {args.num_samples} 个样本。")
    else:
        train_loader = full_train_loader
        
    val_loader = get_dataloader(
        str(processed_dir), 
        batch_size=args.batch_size, 
        split="val", 
        shuffle=False,
        vocab=vocab,
        include_oov=True # 验证集同样需要 PG 机制
    )
    # -------------------------------------------------------------------------


    # 模型（参数已通过优先级处理，直接使用 args 中的值）
    model = PGCT_layer_Model(
        vocab_size=len(vocab),
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,  # 此处已优先使用命令行/配置文件的参数
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        nhead=args.nhead,
        dropout=args.dropout,
        pad_idx=pad_idx,
        cov_loss_weight=args.cov_loss_weight,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        use_layer_attention=use_layer_attention
    ).to(device)
    logger.info("PGCT_layer_Model 初始化完成")
    
    # 创建保存目录
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    tb_writer = SummaryWriter(log_dir=Path(args.save_dir)/"runs")

    best_val_loss = float("inf")
    best_rouge_l = -float("inf")  # 用于跟踪最佳 ROUGE 分数

    for epoch in range(1, args.num_epochs+1):
        model.train()
        running_nll = 0.0
        running_cov = 0.0
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}/{args.num_epochs}")
        
        for batch in pbar:
            src = batch['src'].to(device)
            # tgt 现在是扩展词表索引 (tgt_ext)
            tgt = batch['tgt'].to(device) 
            
            # src_oov_map 用于 Pointer-Generator 机制 (源文本的扩展索引)
            src_oov_map = batch['src_oov_map'].to(device)

            optimizer.zero_grad()
            
            # PGCT_layer_Model.forward 计算输出和损失
            outputs, _, _, coverage_loss = model(
                src, 
                tgt, 
                src_oov_map=src_oov_map, 
                teacher_forcing_ratio=args.teacher_forcing_ratio
            ) # outputs: [B, T_out, V_ext]
            
            # 目标序列需要移位 (移除 SOS 令牌)
            nll_loss = calculate_nll_loss(outputs, tgt[:, 1:], pad_idx)
            
            # 总损失 = NLL Loss + Coverage Loss
            total_loss = nll_loss + coverage_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            running_nll += nll_loss.item()
            running_cov += coverage_loss.item()
            pbar.set_postfix({"NLL": f"{nll_loss.item():.4f}", "Cov": f"{coverage_loss.item():.4f}", "LR": optimizer.param_groups[0]['lr']})

        avg_train_nll = running_nll / len(train_loader)
        avg_train_cov = running_cov / len(train_loader)
        avg_train_total = avg_train_nll + avg_train_cov
        logger.info(f"Epoch {epoch} Train Loss: Total={avg_train_total:.4f} (NLL={avg_train_nll:.4f}, Cov={avg_train_cov:.4f})")
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

                # 验证时 teacher_forcing_ratio 设为 1.0
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
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        # -------------------------------------------------------------------------
        # ROUGE 指标计算（完整实现）
        if HAS_ROUGE and epoch % 2 == 0:  # 每2个epoch计算一次ROUGE（节省时间）
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
            # 可选：保存基于 ROUGE 的最佳模型
            if val_rouge_l > best_rouge_l:
                best_rouge_l = val_rouge_l
                torch.save(model.state_dict(), Path(args.save_dir)/"best_rouge_model.pt")
                logger.info(f"✨ 新最佳 ROUGE 模型保存: {args.save_dir}/best_rouge_model.pt")
        elif not HAS_ROUGE:
            logger.warning("跳过 ROUGE 计算，因为 'rouge' 库未导入或 utils/metrics 缺失。")
        # -------------------------------------------------------------------------


        # 保存基于损失的最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), Path(args.save_dir)/"best_model.pt")
            logger.info(f"✨ 新最佳损失模型保存: {args.save_dir}/best_model.pt")

        # 定期保存 checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = Path(args.save_dir)/f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_val_loss": avg_val_loss,
                "config": {  # 保存当前训练参数，方便推理时对齐
                    "model": model_config,
                    "train": train_config,
                    "data": data_config
                }
            }, ckpt_path)
            logger.info(f"💾 定期保存模型 checkpoint: {ckpt_path}")

    tb_writer.close()
    logger.info("✅ 正式训练完成！")
    logger.info(f"最佳验证损失: {best_val_loss:.4f}")
    if HAS_ROUGE:
        logger.info(f"最佳验证 ROUGE-L: {best_rouge_l:.2f}")


if __name__ == "__main__":
    # 示例运行命令 (在 src 目录下): 
    # python train_pgct_layer.py --config ../configs/pgct_layer.yaml --batch_size 10
    main()
