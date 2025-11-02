"""
PGCT 模型推理脚本（单条或批量文本文件）
"""
import os
import sys
from pathlib import Path
import argparse
from typing import List, Dict

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from models.pgct.pgct_model import PGCTModel
from datamodules.cnndm import prepare_datasets
from utils.decoding import greedy_decode, beam_search_decode
from utils.vocab import Vocab
from nltk.tokenize import word_tokenize


def tokenize(text: str, lowercase: bool=True) -> List[str]:
    if lowercase:
        text = text.lower()
    return word_tokenize(text)


def process_oov(tokens: List[str], vocab: Vocab):
    src_indices, oov_dict, src_oov_map = [], {}, []
    for token in tokens:
        if token in vocab.word2idx:
            src_indices.append(vocab.word2idx[token])
            src_oov_map.append(0)
        else:
            if token not in oov_dict.values():
                oov_idx = len(oov_dict) + 1
                oov_dict[oov_idx] = token
            else:
                oov_idx = [k for k,v in oov_dict.items() if v==token][0]
            src_indices.append(vocab.unk_idx)
            src_oov_map.append(oov_idx)
    return src_indices, oov_dict, src_oov_map


def load_model(checkpoint_path: str, vocab_size: int, pad_idx: int, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get('config', {})
    model = PGCTModel(
        vocab_size=vocab_size,
        embed_size=cfg.get('embed_size',256),
        hidden_size=cfg.get('hidden_size',256),
        num_encoder_layers=cfg.get('num_encoder_layers',3),
        num_decoder_layers=cfg.get('num_decoder_layers',3),
        nhead=cfg.get('nhead',8),
        dropout=cfg.get('dropout',0.1),
        pad_idx=pad_idx,
        cov_loss_weight=cfg.get('cov_loss_weight',1.0),
        max_src_len=cfg.get('max_src_len',400),
        max_tgt_len=cfg.get('max_tgt_len',100)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def summarize_text(model, vocab, article, device, max_src_len, max_tgt_len, decode_strategy, beam_size):
    tokens = tokenize(article)[:max_src_len]
    src_len = len(tokens)
    src_indices, oov_dict, src_oov_map = process_oov(tokens, vocab)

    if len(src_indices) < max_src_len:
        pad_len = max_src_len - len(src_indices)
        src_indices += [vocab.pad_idx]*pad_len
        src_oov_map += [0]*pad_len

    src_tensor = torch.LongTensor([src_indices]).to(device)
    src_len_tensor = torch.LongTensor([src_len]).to(device)
    src_oov_tensor = torch.LongTensor([src_oov_map]).to(device)

    with torch.no_grad():
        if decode_strategy=='beam':
            beams = beam_search_decode(model, src_tensor, src_len_tensor, src_oov_tensor,
                                       max_len=max_tgt_len, sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx,
                                       beam_size=beam_size, device=device)
            pred_ids = beams[0][0] if beams else []
        else:
            pred_ids, _ = greedy_decode(model, src_tensor, src_len_tensor, src_oov_tensor,
                                        max_len=max_tgt_len, sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx,
                                        device=device)

    summary_tokens = []
    for idx in pred_ids:
        idx_val = idx if isinstance(idx,int) else idx.item()
        if idx_val < len(vocab):
            token = vocab.idx2word.get(idx_val,vocab.unk_token)
        else:
            oov_idx = idx_val - len(vocab)
            token = oov_dict.get(oov_idx,vocab.unk_token)
        if token not in [vocab.pad_token,vocab.sos_token,vocab.eos_token]:
            summary_tokens.append(token)
    return summary_tokens


def collect_inputs(input_path: str) -> List[Path]:
    path = Path(input_path)
    if path.is_dir():
        return sorted([p for p in path.glob('*.txt') if p.is_file()])
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"未找到输入路径: {input_path}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    config = {}
    if args.config:
        with open(args.config,'r') as f:
            config = yaml.safe_load(f)

    vocab_path = config.get('vocab_path','../data/processed/vocab.json')
    vocab = Vocab.load(vocab_path)

    checkpoint_path = args.checkpoint or config.get('checkpoint_path','./checkpoints/pgct_best_model.pt')
    model = load_model(checkpoint_path,len(vocab),vocab.pad_idx,device)
    print(f"✓ 模型已加载: {checkpoint_path}")

    input_files = collect_inputs(args.input)
    os.makedirs(os.path.dirname(args.output),exist_ok=True)
    results = []

    for idx,file in enumerate(input_files,1):
        with file.open('r',encoding='utf-8') as f:
            article = f.read().strip()
        print(f"\n--- 处理文件 {idx}/{len(input_files)} ---")
        summary_tokens = summarize_text(model,vocab,article,device,
                                        max_src_len=config.get('max_src_len',400),
                                        max_tgt_len=config.get('max_tgt_len',100),
                                        decode_strategy=args.decode_strategy,
                                        beam_size=args.beam_size)
        summary = ' '.join(summary_tokens)
        results.append({'id':idx,'file':str(file),'summary':summary})
        print(f"生成摘要: {summary[:200]}...")

    with open(args.output,'w',encoding='utf-8') as f:
        for item in results:
            f.write(f"=== 样本 {item['id']} ===\n文件: {item['file']}\n摘要: {item['summary']}\n\n")
    print(f"\n✓ 所有摘要已保存: {args.output}")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, help='模型检查点路径')
    parser.add_argument('--input', type=str, required=True, help='输入文本文件或目录')
    parser.add_argument('--output', type=str, default='../docs/pgct_samples.txt')
    parser.add_argument('--decode_strategy', type=str, default='greedy', choices=['greedy','beam'])
    parser.add_argument('--beam_size', type=int, default=5)
    main(parser.parse_args())
