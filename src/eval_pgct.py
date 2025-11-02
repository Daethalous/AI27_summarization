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

from datamodules.cnndm import get_dataloader, prepare_datasets
from models.pgct.pgct_model import PGCTModel
from utils.decoding import greedy_decode, beam_search_decode
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
    """PGCT模型生成摘要（支持OOV）"""
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
        oov_lists = batch.get('oov_list', [[] for _ in range(src.size(0))])

        for i in range(src.size(0)):
            src_i = src[i:i+1]
            src_len_i = src_len[i:i+1] if src_len is not None else None
            src_oov_map_i = src_oov_map[i:i+1] if src_oov_map is not None else None
            oov_list_i = oov_lists[i]
            tgt_i = tgt[i]

            # 解码
            if decode_strategy == 'greedy':
                pred_ids, _ = greedy_decode(
                    model, src_i, src_len_i, src_oov_map_i,
                    max_len=max_len, sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx, device=device
                )
            elif decode_strategy == 'beam':
                beams = beam_search_decode(
                    model, src_i, src_len_i, src_oov_map_i,
                    max_len=max_len, sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx,
                    beam_size=beam_size, device=device
                )
                pred_ids = beams[0][0] if beams else []
            else:
                raise ValueError(f"Unknown decode strategy: {decode_strategy}")

            # 转 token
            pred_tokens = []
            for idx in pred_ids:
                idx_val = idx if isinstance(idx, int) else idx.item()
                if idx_val < len(vocab):
                    token = vocab.idx2word.get(idx_val, Vocab.UNK_TOKEN)
                else:
                    oov_idx = idx_val - len(vocab)
                    token = oov_list_i[oov_idx] if oov_idx < len(oov_list_i) else Vocab.UNK_TOKEN
                if token not in [Vocab.PAD_TOKEN, Vocab.SOS_TOKEN, Vocab.EOS_TOKEN]:
                    pred_tokens.append(token)

            ref_tokens = vocab.decode(tgt_i.cpu().tolist(), skip_special=True)

            # 源文本 token
            src_tokens = []
            for idx in src_i.squeeze().cpu().tolist():
                if idx < len(vocab):
                    token = vocab.idx2word.get(idx, Vocab.UNK_TOKEN)
                else:
                    oov_idx = idx - len(vocab)
                    token = oov_list_i[oov_idx] if oov_idx < len(oov_list_i) else Vocab.UNK_TOKEN
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

    # 加载配置
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    vocab_path = args.vocab_path or config.get('vocab_path', '../data/processed/vocab.json')
    checkpoint_path = args.checkpoint or config.get('checkpoint_path', './checkpoints/pgct_best_model.pt')
    split = args.split
    batch_size = args.batch_size or config.get('batch_size', 32)
    max_src_len = args.max_src_len or config.get('max_src_len', 400)
    max_tgt_len = args.max_tgt_len or config.get('max_tgt_len', 100)

    # 确保数据与词表
    vocab = Vocab.load(vocab_path)

    dataloader = get_dataloader(
        Path(vocab_path).parent, batch_size, split=split,
        num_workers=args.num_workers, shuffle=False, include_oov=True
    )

    # 加载模型
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = checkpoint.get('config', {})
    model = PGCTModel(
        vocab_size=len(vocab),
        embed_size=model_cfg.get('embed_size', 256),
        hidden_size=model_cfg.get('hidden_size', 256),
        num_encoder_layers=model_cfg.get('num_encoder_layers', 3),
        num_decoder_layers=model_cfg.get('num_decoder_layers', 3),
        nhead=model_cfg.get('nhead', 8),
        dropout=model_cfg.get('dropout', 0.1),
        pad_idx=vocab.pad_idx,
        cov_loss_weight=model_cfg.get('cov_loss_weight', 1.0),
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ 模型已加载: {checkpoint_path}")

    # 生成摘要
    predictions, references, sources = generate_summaries(
        model, dataloader, vocab, device,
        max_len=max_tgt_len,
        decode_strategy=args.decode_strategy,
        beam_size=args.beam_size,
        output_file=args.output
    )

    # 计算指标
    metrics = batch_compute_metrics(predictions, references, set(vocab.word2idx.keys()))
    print_metrics(metrics, prefix=f"{split.upper()} ({args.decode_strategy})")

    # 打印示例
    if args.show_examples > 0:
        for i in range(min(args.show_examples, len(predictions))):
            print(f"\n--- 示例 {i+1} ---")
            print(f"源文本: {' '.join(sources[i][:100])}...")
            print(f"参考: {' '.join(references[i][:50])}...")
            print(f"预测: {' '.join(predictions[i][:50])}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, help='模型检查点路径')
    parser.add_argument('--vocab_path', type=str, help='词表路径')
    parser.add_argument('--split', type=str, default='test', choices=['train','val','test'])
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--max_src_len', type=int)
    parser.add_argument('--max_tgt_len', type=int)
    parser.add_argument('--decode_strategy', type=str, default='greedy', choices=['greedy','beam'])
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--show_examples', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()
    main(args)
