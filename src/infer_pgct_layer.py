"""
PGCT_layer 模型推理脚本（单条或批量文本文件）
适配 Transformer + Pointer-Generator + Coverage 模型
支持 OOV 词处理与 Greedy/Beam 解码
"""
import os
import sys
import time
from pathlib import Path
import argparse
from typing import List, Dict

import torch
import yaml
import nltk
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

sys.path.insert(0, str(Path(__file__).parent))

from models.pgct_layer.pgct_layer_model import PGCT_layerModel
from models.pgct_layer.pgct_decoding import pgct_greedy_decode, pgct_beam_search_decode
from utils.vocab import Vocab


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    if lowercase:
        text = text.lower()
    return word_tokenize(text)


def process_oov(tokens: List[str], vocab: Vocab) -> tuple[List[int], Dict[int, str], List[int]]:
    src_indices, oov_dict, src_oov_map = [], {}, []
    for token in tokens:
        if token in vocab.word2idx:
            src_indices.append(vocab.word2idx[token])
            src_oov_map.append(-1)
        else:
            if token not in oov_dict.values():
                new_oov_idx = len(oov_dict)
                oov_dict[new_oov_idx] = token
            oov_idx = [k for k, v in oov_dict.items() if v == token][0]
            src_indices.append(vocab.unk_idx)
            src_oov_map.append(oov_idx)
    return src_indices, oov_dict, src_oov_map


def load_pgct_layer_model(checkpoint_path: str, vocab_size: int, pad_idx: int, device: torch.device, config: Dict = None) -> tuple[PGCT_layer_Model, Dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = config.get('model', {}) if config else {}
    checkpoint_cfg = checkpoint.get('config', {})

    model_cfg = {
        'embed_size': cfg.get('embed_size', checkpoint_cfg.get('embed_size', 256)),
        'hidden_size': cfg.get('hidden_size', checkpoint_cfg.get('hidden_size', 256)),
        'num_encoder_layers': cfg.get('num_encoder_layers', checkpoint_cfg.get('num_encoder_layers', 3)),
        'num_decoder_layers': cfg.get('num_decoder_layers', checkpoint_cfg.get('num_decoder_layers', 3)),
        'nhead': cfg.get('nhead', checkpoint_cfg.get('nhead', 8)),
        'dropout': cfg.get('dropout', checkpoint_cfg.get('dropout', 0.1)),
        'cov_loss_weight': cfg.get('cov_loss_weight', checkpoint_cfg.get('cov_loss_weight', 1.0)),
        'max_src_len': cfg.get('max_src_len', checkpoint_cfg.get('max_src_len', 400)),
        'max_tgt_len': cfg.get('max_tgt_len', checkpoint_cfg.get('max_tgt_len', 100))
    }

    model = PGCT_layer_Model(
        vocab_size=vocab_size,
        embed_size=model_cfg['embed_size'],
        hidden_size=model_cfg['hidden_size'],
        num_encoder_layers=model_cfg['num_encoder_layers'],
        num_decoder_layers=model_cfg['num_decoder_layers'],
        nhead=model_cfg['nhead'],
        dropout=model_cfg['dropout'],
        pad_idx=pad_idx,
        cov_loss_weight=model_cfg['cov_loss_weight'],
        max_src_len=model_cfg['max_src_len'],
        max_tgt_len=model_cfg['max_tgt_len']
    ).to(device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, model_cfg


def summarize_single_text(model: PGCT_layer_Model, vocab: Vocab, article: str, device: torch.device, max_src_len: int, max_tgt_len: int, decode_strategy: str, beam_size: int) -> str:
    tokens = tokenize(article)[:max_src_len]
    src_len = len(tokens)
    src_indices, oov_dict, src_oov_map = process_oov(tokens, vocab)

    pad_len = max_src_len - len(src_indices)
    src_indices += [vocab.pad_idx] * pad_len
    src_oov_map += [-1] * pad_len

    src_tensor = torch.LongTensor([src_indices]).to(device)
    src_len_tensor = torch.LongTensor([src_len]).to(device)
    src_oov_tensor = torch.LongTensor([src_oov_map]).to(device)

    with torch.no_grad():
        if decode_strategy == 'beam':
            pred_ids, _ = pgct_beam_search_decode(model, src_tensor, src_len_tensor, src_oov_tensor, max_tgt_len, vocab.sos_idx, vocab.eos_idx, beam_size, device)
        else:
            pred_ids, _ = pgct_greedy_decode(model, src_tensor, src_len_tensor, src_oov_tensor, max_tgt_len, vocab.sos_idx, vocab.eos_idx, device)

    summary_tokens = []
    for idx in pred_ids.squeeze().tolist():
        idx_val = idx if isinstance(idx, int) else idx.item()
        if idx_val < len(vocab):
            token = vocab.idx2word.get(idx_val, vocab.UNK_TOKEN)
        else:
            oov_rel_idx = idx_val - len(vocab)
            token = oov_dict.get(oov_rel_idx, vocab.UNK_TOKEN)
        if token not in [vocab.PAD_TOKEN, vocab.SOS_TOKEN, vocab.EOS_TOKEN]:
            summary_tokens.append(token)

    return ' '.join(summary_tokens)


def collect_input_files(input_path: str) -> List[Path]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"输入路径不存在：{input_path}")
    if path.is_file() and path.suffix == '.txt':
        return [path]
    elif path.is_dir():
        txt_files = sorted(list(path.rglob('*.txt')))
        if not txt_files:
            raise FileNotFoundError(f"目录下未找到任何 .txt 文件：{input_path}")
        return txt_files
    else:
        raise ValueError(f"输入路径必须是 .txt 文件或目录：{input_path}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ 使用设备：{device}")

    config = {}
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ 加载配置文件：{args.config}")

    data_cfg = config.get('data', {})
    model_cfg_raw = config.get('model', {})
    eval_cfg = config.get('eval', {})

    vocab_path = args.vocab_path or data_cfg.get('vocab_path', '../data/processed/vocab.json')
    checkpoint_path = args.checkpoint or config.get('train', {}).get('save_dir', '../checkpoints_pgct_layer/best_model.pt')
    input_path = args.input
    decode_strategy = args.decode_strategy or eval_cfg.get('decode_strategy', 'greedy')
    beam_size = args.beam_size or eval_cfg.get('beam_size', 5)
    output_file = args.output or os.path.join(eval_cfg.get('output_dir', '../outputs_pgct_layer'), eval_cfg.get('output_file', 'test_summaries.json'))

    vocab = Vocab.load(vocab_path)
    print(f"✅ 加载词表：{vocab_path}（大小：{len(vocab)}）")

    model, model_cfg = load_pgct_layer_model(checkpoint_path, len(vocab), vocab.pad_idx, device, config)
    print(f"✅ 加载模型：{checkpoint_path}")

    input_files = collect_input_files(input_path)
    print(f"✅ 收集到输入文件：{len(input_files)} 个")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    results = []
    print(f"\n🚀 开始批量生成摘要（策略：{decode_strategy}，束大小：{beam_size}）")
    for idx, file in enumerate(input_files, 1):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                article = f.read().strip()
            if not article:
                print(f"⚠️ 跳过空文件 [{idx}/{len(input_files)}]：{file.name}")
                continue
        except Exception as e:
            print(f"⚠️ 读取失败 [{idx}/{len(input_files)}]：{file.name}，错误：{str(e)[:30]}...，跳过")
            continue

        try:
            summary = summarize_single_text(
                model=model,
                vocab=vocab,
                article=article,
                device=device,
                max_src_len=model_cfg['max_src_len'],
                max_tgt_len=model_cfg['max_tgt_len'],
                decode_strategy=decode_strategy,
                beam_size=beam_size
            )
        except Exception as e:
            print(f"⚠️ 生成摘要失败 [{idx}/{len(input_files)}]：{file.name}，错误：{str(e)[:30]}...，跳过该文件")
            continue

        results.append({
            'id': idx,
            'file_name': file.name,
            'file_path': str(file),
            'article_char_count': len(article),
            'summary_token_count': len(summary.split()),
            'summary': summary
        })

        preview = summary[:150] + "..." if len(summary) > 150 else summary
        print(f"✅ 完成 [{idx}/{len(input_files)}] | 文件：{file.name} | 摘要词数：{len(summary.split())}")
        print(f"   摘要预览：{preview}")

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("PGCT_layer 模型推理结果汇总\n")
        f.write("=" * 60 + "\n")
        f.write(f"推理配置：\n")
        f.write(f"  - 模型路径：{checkpoint_path}\n")
        f.write(f"  - 词表路径：{vocab_path}\n")
        f.write(f"  - 解码策略：{decode_strategy}\n")
        f.write(f"  - 束搜索大小：{beam_size}\n")
        f.write(f"  - 最大源文本长度：{model_cfg['max_src_len']}\n")
        f.write(f"  - 最大摘要长度：{model_cfg['max_tgt_len']}\n")
        f.write(f"  - 输入文件数：{len(results)}\n")
        f.write("=" * 60 + "\n\n")

        for item in results:
            f.write(f"=== 样本 {item['id']} ===\n")
            f.write(f"文件名：{item['file_name']}\n")
            f.write(f"路径：{item['file_path']}\n")
            f.write(f"原文字符数：{item['article_char_count']}\n")
            f.write(f"摘要词数：{item['summary_token_count']}\n")
            f.write(f"生成摘要：{item['summary']}\n\n")

    elapsed = time.time() - start_time
    print(f"\n✅ 推理完成，共处理 {len(results)} 个样本，用时 {elapsed:.2f} 秒")
    print(f"✓ 结果已保存到：{output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PGCT_layer 模型推理脚本")
    parser.add_argument('--config', type=str, help='配置文件路径（如 ../configs/pgct_layer.yaml）')
    parser.add_argument('--checkpoint', type=str, help='模型 checkpoint 路径')
    parser.add_argument('--vocab_path', type=str, help='词表路径（默认从 config 读取）')
    parser.add_argument('--input', type=str, required=True, help='输入 .txt 文件或目录（如 ../data/raw/test）')
    parser.add_argument('--output', type=str, help='结果保存路径（默认从 config 读取）')
    parser.add_argument('--decode_strategy', type=str, choices=['greedy', 'beam'], help='解码策略（默认从 config 读取）')
    parser.add_argument('--beam_size', type=int, help='束搜索大小（仅 beam 策略生效）')
    args = parser.parse_args()
    main(args)
