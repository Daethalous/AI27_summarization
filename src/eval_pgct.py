"""
模型评估和推理脚本本- 适配 Transformer+Pointer-Generator+Coverage (PGCT) 模型
"""
import os
import sys
import yaml
import argparse
import json
from pathlib import Path
import torch
from tqdm import tqdm

# 添加src到path
sys.path.insert(0, str(Path(__file__).parent))

from datamodules.cnndm import get_dataloader, prepare_datasets
from models.pointer_generator_coverage.pgct_model import PGCTModel  # 导入PGCT模型
from utils.decoding import greedy_decode, beam_search_decode
from utils.metrics import batch_compute_metrics, print_metrics
from utils.vocab import Vocab


@torch.no_grad()
def generate_summaries(
    model, 
    dataloader, 
    vocab, 
    device,
    max_len=512,
    decode_strategy='greedy',
    beam_size=5,
    output_file=None
):
    """生成摘要（适配PGCT模型，支持OOV处理）
    
    Args:
        model: PGCT模型实例
        dataloader: 数据加载器
        vocab: 词表
        device: 设备
        max_len: 最大解码长度
        decode_strategy: 'greedy' 或 'beam'
        beam_size: beam search的beam大小
        output_file: 输出文件路径（可选）
        
    Returns:
        predictions: 预测摘要token列表的列表
        references: 参考摘要token列表的列表
    """
    model.eval()
    
    all_predictions = []
    all_references = []
    all_sources = []
    
    results = []  # 用于保存到文件
    
    pbar = tqdm(dataloader, desc=f"Generating ({decode_strategy})")
    for batch in pbar:
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        src_len = batch.get('src_len', None)
        if src_len is not None:
            src_len = src_len.squeeze().to(device)
        src_oov_map = batch.get('src_oov_map', None)  # PGCT模型需要OOV映射表
        if src_oov_map is not None:
            src_oov_map = src_oov_map.to(device)
        oov_lists = batch.get('oov_list', [[] for _ in range(src.size(0))])  # OOV词列表
        
        batch_size = src.size(0)
        
        for i in range(batch_size):
            # 单个样本处理
            src_i = src[i:i+1]
            src_len_i = src_len[i:i+1] if src_len is not None else None
            src_oov_map_i = src_oov_map[i:i+1] if src_oov_map is not None else None
            oov_list_i = oov_lists[i]
            tgt_i = tgt[i]
            
            # 解码
            if decode_strategy == 'greedy':
                pred_ids, attn_weights = greedy_decode(
                    model, 
                    src=src_i, 
                    src_lens=src_len_i,
                    src_oov_map=src_oov_map_i,  # 传入OOV映射表
                    max_len=max_len, 
                    sos_idx=vocab.sos_idx, 
                    eos_idx=vocab.eos_idx, 
                    device=device
                )
            elif decode_strategy == 'beam':
                beams = beam_search_decode(
                    model, 
                    src=src_i, 
                    src_lens=src_len_i,
                    src_oov_map=src_oov_map_i,  # 传入OOV映射表
                    max_len=max_len,
                    sos_idx=vocab.sos_idx, 
                    eos_idx=vocab.eos_idx, 
                    beam_size=beam_size, 
                    device=device
                )
                pred_ids = beams[0][0] if beams else []
            else:
                raise ValueError(f"Unknown decode strategy: {decode_strategy}")
            
            # 转换为token（处理OOV）
            pred_tokens = []
            for idx in pred_ids:
                idx_val = idx if isinstance(idx, int) else idx.item()
                if idx_val < len(vocab):
                    # 常规词
                    token = vocab.idx2word.get(idx_val, Vocab.UNK_TOKEN)
                else:
                    # OOV词（索引 = 词表大小 + OOV在源文中的索引）
                    oov_idx = idx_val - len(vocab)
                    if oov_idx < len(oov_list_i):
                        token = oov_list_i[oov_idx]
                    else:
                        token = Vocab.UNK_TOKEN
                if token not in [Vocab.PAD_TOKEN, Vocab.SOS_TOKEN, Vocab.EOS_TOKEN]:
                    pred_tokens.append(token)
            
            # 参考摘要转换（无需处理OOV，因为参考摘要已在词表内）
            ref_tokens = vocab.decode(tgt_i.cpu().tolist(), skip_special=True)
            # 源文本转换（包含OOV）
            src_tokens = []
            src_ids = src_i.squeeze().cpu().tolist()
            for idx in src_ids:
                if idx < len(vocab):
                    token = vocab.idx2word.get(idx, Vocab.UNK_TOKEN)
                else:
                    oov_idx = idx - len(vocab)
                    if oov_idx < len(oov_list_i):
                        token = oov_list_i[oov_idx]
                    else:
                        token = Vocab.UNK_TOKEN
                if token not in [Vocab.PAD_TOKEN]:
                    src_tokens.append(token)
            
            all_predictions.append(pred_tokens)
            all_references.append(ref_tokens)
            all_sources.append(src_tokens)
            
            # 保存结果
            results.append({
                'source': ' '.join(src_tokens),
                'reference': ' '.join(ref_tokens),
                'prediction': ' '.join(pred_tokens)
            })
    
    # 保存到文件
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ 结果已保存到: {output_file}")
    
    return all_predictions, all_references


def main(args):
    """主评估函数（适配PGCT模型）"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载配置
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # 参数
    raw_data_dir = args.data_dir or config.get('data_dir', '../data/raw')
    vocab_path = args.vocab_path or config.get('vocab_path', '../data/processed/vocab.json')
    processed_dir = config.get('processed_dir', os.path.dirname(vocab_path))
    checkpoint_path = args.checkpoint or config.get('checkpoint_path', './checkpoints/pgct_best_model.pt')

    batch_size = args.batch_size or config.get('batch_size', 32)
    max_src_len = args.max_src_len or config.get('max_src_len', 400)
    max_tgt_len = args.max_tgt_len or config.get('max_tgt_len', 100)

    # 确保预处理数据与词表已就绪
    print(f"准备数据与词表: {vocab_path}")
    vocab = prepare_datasets(
        raw_dir=raw_data_dir,
        processed_dir=processed_dir,
        vocab_path=vocab_path,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        max_vocab_size=config.get('max_vocab_size', 50000),
        min_freq=config.get('min_freq', 5)
    )
    if not isinstance(vocab, Vocab):
        # 兼容prepare_datasets的返回值（可能返回词表路径，这里显式加载）
        vocab = Vocab.load(vocab_path)

    # 创建数据加载器（需包含OOV相关字段）
    print(f"加载{args.split}数据...")
    dataloader = get_dataloader(
        processed_dir,
        batch_size,
        split=args.split,
        num_workers=args.num_workers,
        shuffle=False,
        include_oov=True  # 关键：确保数据加载器返回OOV相关数据
    )
    
    # 加载PGCT模型
    print(f"加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model_config = checkpoint.get('config', {})
    model = PGCTModel(
        vocab_size=len(vocab),
        embed_size=model_config.get('embed_size', 256),
        hidden_size=model_config.get('hidden_size', 256),
        num_encoder_layers=model_config.get('num_encoder_layers', 3),
        num_decoder_layers=model_config.get('num_decoder_layers', 3),
        nhead=model_config.get('nhead', 8),  # Transformer多头注意力头数
        dropout=model_config.get('dropout', 0.1),
        pad_idx=vocab.pad_idx,
        cov_loss_weight=model_config.get('cov_loss_weight', 1.0),
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ 模型已加载 (训练至epoch {checkpoint.get('epoch', 'N/A')})")
    
    # 生成摘要
    print(f"\n开始生成摘要 (策略: {args.decode_strategy})...")
    predictions, references = generate_summaries(
        model, dataloader, vocab, device,
        max_len=max_tgt_len,
        decode_strategy=args.decode_strategy,
        beam_size=args.beam_size,
        output_file=args.output
    )
    
    # 计算指标
    print("\n计算评估指标...")
    vocab_set = set(vocab.word2idx.keys())
    metrics = batch_compute_metrics(predictions, references, vocab_set)
    
    # 打印结果
    print_metrics(metrics, prefix=f"{args.split.upper()} ({args.decode_strategy})")
    
    # 保存指标
    if args.metrics_output:
        os.makedirs(os.path.dirname(args.metrics_output), exist_ok=True)
        with open(args.metrics_output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ 指标已保存到: {args.metrics_output}")
    
    # 打印示例
    if args.show_examples > 0:
        print(f"\n{'=' * 80}")
        print("示例摘要:")
        print(f"{'=' * 80}")
        
        for i in range(min(args.show_examples, len(predictions))):
            print(f"\n--- 示例 {i+1} ---")
            print(f"源文本: {' '.join(all_sources[i][:100])}...")  # 显示部分源文本
            print(f"参考: {' '.join(references[i][:50])}...")
            print(f"预测: {' '.join(predictions[i][:50])}...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='评估 Transformer+Pointer-Generator+Coverage (PGCT) 模型')
    
    # 模型和数据
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, help='模型检查点路径')
    parser.add_argument('--data_dir', type=str, help='原始数据目录（包含train/validation/test子目录）')
    parser.add_argument('--vocab_path', type=str, help='词表路径')
    
    # 评估参数
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], 
                        help='评估的数据集划分')
    parser.add_argument('--batch_size', type=int, help='批次大小')
    parser.add_argument('--max_src_len', type=int, help='最大源文本长度')
    parser.add_argument('--max_tgt_len', type=int, help='最大目标文本长度')
    
    # 解码策略
    parser.add_argument('--decode_strategy', type=str, default='greedy', 
                        choices=['greedy', 'beam'], help='解码策略')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam search的beam大小')
    
    # 输出
    parser.add_argument('--output', type=str, help='生成摘要的输出文件路径')
    parser.add_argument('--metrics_output', type=str, help='评估指标的输出文件路径')
    parser.add_argument('--show_examples', type=int, default=3, help='打印示例数量')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')
    
    args = parser.parse_args()
    main(args)
