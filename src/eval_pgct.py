"""
PGCT 模型评估脚本
适配 Transformer + Pointer-Generator + Coverage 模型
"""
import os
import sys
import yaml
import argparse
import json
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

# 添加 src 到 path
sys.path.insert(0, str(Path(__file__).parent))

from datamodules.cnndm import get_dataloader
from models.pgct.pgct_model import PGCTModel
from models.pgct.pgct_decoding import pgct_greedy_decode, pgct_beam_search_decode
from utils.metrics import batch_compute_metrics, print_metrics
from utils.vocab import Vocab


@torch.no_grad()
def generate_summaries(
    model: PGCTModel,
    dataloader,
    vocab: Vocab,
    device: torch.device,
    max_len: int = 512,
    decode_strategy: str = 'greedy',
    beam_size: int = 5,
    output_file: str = None
):
    model.eval()
    all_predictions, all_references, all_sources = [], [], []
    results = []

    for batch in tqdm(dataloader, desc=f"Generating ({decode_strategy})"):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        src_len = batch.get('src_len', None)
        if src_len is not None:
            src_len = src_len.squeeze().to(device)
        src_oov_map = batch.get('src_oov_map', None)
        if src_oov_map is not None:
            src_oov_map = src_oov_map.to(device)
        src_tokens_batch = batch.get('src_tokens', [[] for _ in range(src.size(0))])

        oov_lists = []
        for i in range(src.size(0)):
            curr_src_tokens = src_tokens_batch[i]
            curr_oov_map = src_oov_map[i].cpu().tolist() if src_oov_map is not None else []
            oov_dict = {}
            oov_list = []
            if src_oov_map is not None and len(curr_oov_map) > 0:
                for token, oov_idx in zip(curr_src_tokens, curr_oov_map):
                    if token not in vocab.word2idx and oov_idx >= 0:
                        if token not in oov_dict:
                            oov_dict[token] = len(oov_list)
                            oov_list.append(token)
            oov_lists.append(oov_list)

        for i in range(src.size(0)):
            src_i = src[i:i+1]
            src_len_i = src_len[i:i+1] if (src_len is not None and len(src_len) > i) else None
            src_oov_map_i = src_oov_map[i:i+1] if src_oov_map is not None else None
            oov_list_i = oov_lists[i]
            tgt_i = tgt[i]
            curr_src_tokens = src_tokens_batch[i]

            if decode_strategy == 'greedy':
                pred_ids, _ = pgct_greedy_decode(
                    model=model,
                    src=src_i,
                    src_lens=src_len_i,
                    src_oov_map=src_oov_map_i,
                    max_length=max_len,
                    sos_idx=vocab.sos_idx,
                    eos_idx=vocab.eos_idx,
                    device=device
                )
            elif decode_strategy == 'beam':
                pred_ids, _ = pgct_beam_search_decode(
                    model=model,
                    src=src_i,
                    src_lens=src_len_i,
                    src_oov_map=src_oov_map_i,
                    beam_size=beam_size,
                    max_length=max_len,
                    sos_idx=vocab.sos_idx,
                    eos_idx=vocab.eos_idx,
                    device=device
                )
            else:
                raise ValueError(f"Unknown decode strategy: {decode_strategy}")

            pred_tokens = []
            for idx in pred_ids.squeeze().tolist():
                idx_val = idx if isinstance(idx, int) else idx.item()
                if idx_val < len(vocab):
                    token = vocab.idx2word.get(idx_val, Vocab.UNK_TOKEN)
                else:
                    oov_idx = idx_val - len(vocab)
                    token = oov_list_i[oov_idx] if (oov_idx >= 0 and oov_idx < len(oov_list_i)) else Vocab.UNK_TOKEN
                if token not in [Vocab.PAD_TOKEN, Vocab.SOS_TOKEN, Vocab.EOS_TOKEN]:
                    pred_tokens.append(token)

            ref_tokens = vocab.decode(tgt_i.cpu().tolist(), skip_special=True)

            src_tokens = []
            for idx in src_i.squeeze().cpu().tolist():
                if idx < len(vocab):
                    token = vocab.idx2word.get(idx, Vocab.UNK_TOKEN)
                else:
                    oov_idx = idx - len(vocab)
                    token = oov_list_i[oov_idx] if (oov_idx >= 0 and oov_idx < len(oov_list_i)) else Vocab.UNK_TOKEN
                if token != Vocab.PAD_TOKEN:
                    src_tokens.append(token)

            all_predictions.append(pred_tokens)
            all_references.append(ref_tokens)
            all_sources.append(src_tokens)
            results.append({
                'source': ' '.join(src_tokens),
                'reference': ' '.join(ref_tokens),
                'prediction': ' '.join(pred_tokens)
            })

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✓ 结果已保存到: {output_file}")

    return all_predictions, all_references, all_sources


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    data_cfg = config.get('data', {})
    model_cfg_raw = config.get('model', {})
    eval_cfg = config.get('eval', {})

    vocab_path = args.vocab_path or data_cfg.get('vocab_path', '../data/processed/vocab.json')
    checkpoint_path = args.checkpoint
    split = args.split
    batch_size = args.batch_size or eval_cfg.get('batch_size', 32)
    max_src_len = args.max_src_len or data_cfg.get('max_src_len', 400)
    max_tgt_len = args.max_tgt_len or data_cfg.get('max_tgt_len', 100)
    decode_strategy = args.decode_strategy or eval_cfg.get('decode_strategy', 'greedy')
    beam_size = args.beam_size or eval_cfg.get('beam_size', 5)
    num_workers = args.num_workers or eval_cfg.get('num_workers', 0)
    output_file = args.output or os.path.join(
        eval_cfg.get('output_dir', '../outputs_pgct'),
        eval_cfg.get('output_file', 'test_summaries.json')
    )
    show_examples = args.show_examples or eval_cfg.get('show_examples', 3)

    vocab = Vocab.load(vocab_path)
    print(f"✓ 词表已加载: {len(vocab)} 个词")

    dataloader = get_dataloader(
        processed_dir=Path(vocab_path).parent,
        batch_size=batch_size,
        split=split,
        num_workers=num_workers,
        shuffle=False,
        vocab=vocab,
        include_oov=True
    )
    print(f"✓ 数据集已加载: {split} 集，共 {len(dataloader.dataset)} 个样本")

    print(f"正在加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_cfg = {
        'embed_size': args.embed_size or model_cfg_raw.get('embed_size', 512),
        'hidden_size': args.hidden_size or model_cfg_raw.get('hidden_size', 512),
        'num_encoder_layers': args.num_encoder_layers or model_cfg_raw.get('num_encoder_layers', 3),
        'num_decoder_layers': args.num_decoder_layers or model_cfg_raw.get('num_decoder_layers', 3),
        'nhead': args.nhead or model_cfg_raw.get('nhead', 8),
        'dropout': args.dropout or model_cfg_raw.get('dropout', 0.1),
        'cov_loss_weight': args.cov_loss_weight or model_cfg_raw.get('cov_loss_weight', 1.0)
    }

    model = PGCTModel(
        vocab_size=len(vocab),
        embed_size=model_cfg['embed_size'],
        hidden_size=model_cfg['hidden_size'],
        num_encoder_layers=model_cfg['num_encoder_layers'],
        num_decoder_layers=model_cfg['num_decoder_layers'],
        nhead=model_cfg['nhead'],
        dropout=model_cfg['dropout'],
        pad_idx=vocab.pad_idx,
        cov_loss_weight=model_cfg['cov_loss_weight'],
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len
    ).to(device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ 模型参数加载完成（epoch: {checkpoint.get('epoch', '未知')}）")
    else:
        model.load_state_dict(checkpoint)
        print(f"✓ 纯模型参数加载完成")

    # 生成摘要
    print(f"\n开始生成摘要（策略：{decode_strategy}，最大长度：{max_tgt_len}）")
    predictions, references, sources = generate_summaries(
        model=model,
        dataloader=dataloader,
        vocab=vocab,
        device=device,
        max_len=max_tgt_len,
        decode_strategy=decode_strategy,
        beam_size=beam_size,
        output_file=output_file
    )

    # 计算并打印评估指标
    print(f"\n开始计算评估指标...")
    metrics = batch_compute_metrics(predictions, references, set(vocab.word2idx.keys()))
    print_metrics(metrics, prefix=f"{split.upper()} ({decode_strategy})")

    # 打印示例
    if show_examples > 0:
        print(f"\n=== 生成示例（共 {show_examples} 个）===")
        for i in range(min(show_examples, len(predictions))):
            print(f"\n--- 示例 {i+1} ---")
            print(f"源文本: {' '.join(sources[i][:100])}{'...' if len(sources[i]) > 100 else ''}")
            print(f"参考摘要: {' '.join(references[i][:50])}{'...' if len(references[i]) > 50 else ''}")
            print(f"预测摘要: {' '.join(predictions[i][:50])}{'...' if len(predictions[i]) > 50 else ''}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PGCT 模型评估脚本（支持 Pointer-Generator + Coverage）")
    parser.add_argument('--config', type=str, help='YAML 配置文件路径（可选）')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型 checkpoint 路径（必填）')
    parser.add_argument('--vocab_path', type=str, help='词表文件路径（默认：../data/processed/vocab.json）')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='评估数据集拆分')
    parser.add_argument('--batch_size', type=int, help='评估批次大小（默认：32）')
    parser.add_argument('--max_src_len', type=int, help='源文本最大长度（默认：400）')
    parser.add_argument('--max_tgt_len', type=int, help='摘要最大生成长度（默认：100）')
    parser.add_argument('--decode_strategy', type=str, default='greedy', choices=['greedy', 'beam'], help='解码策略')
    parser.add_argument('--beam_size', type=int, default=5, help='束搜索大小（仅 beam 策略生效）')
    parser.add_argument('--output', type=str, help='生成结果保存路径（JSON 文件，可选）')
    parser.add_argument('--show_examples', type=int, default=3, help='展示生成示例数量（默认：3）')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数（默认：0）')
    parser.add_argument('--embed_size', type=int, help='嵌入层维度（默认：512）')
    parser.add_argument('--hidden_size', type=int, help='模型隐藏层维度（默认：512）')
    parser.add_argument('--num_encoder_layers', type=int, help='Transformer 编码器层数（默认：3）')
    parser.add_argument('--num_decoder_layers', type=int, help='Transformer 解码器层数（默认：3）')
    parser.add_argument('--nhead', type=int, help='注意力头数（默认：8）')
    parser.add_argument('--dropout', type=float, help='Dropout 概率（默认：0.1）')
    parser.add_argument('--cov_loss_weight', type=float, help='覆盖损失权重（默认：1.0）')

    args = parser.parse_args()
    main(args)
