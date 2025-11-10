"""
PGCT 模型推理脚本（单条或批量文本文件）
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

# 将 src 目录添加到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.pgct.pgct_model import PGCTModel
from models.pgct.pgct_decoding import pgct_greedy_decode, pgct_beam_search_decode
from utils.vocab import Vocab


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    """使用 nltk.word_tokenize 对文本进行分词和可选的小写化"""
    if lowercase:
        text = text.lower()
    return word_tokenize(text)


def process_oov(tokens: List[str], vocab: Vocab) -> tuple[List[int], Dict[int, str], List[int]]:
    """处理 OOV 词，返回词索引、OOV 词典和 OOV 映射"""
    src_indices, oov_dict, src_oov_map = [], {}, []
    for token in tokens:
        if token in vocab.word2idx:
            src_indices.append(vocab.word2idx[token])
            src_oov_map.append(-1) # -1 表示词在基础词表中
        else:
            # 这是一个 OOV 词
            if token not in oov_dict.values():
                # 分配一个新的 OOV 相对索引
                new_oov_idx = len(oov_dict)
                oov_dict[new_oov_idx] = token
            
            # 获取 OOV 相对索引
            oov_idx = [k for k, v in oov_dict.items() if v == token][0]
            
            src_indices.append(vocab.unk_idx) # 使用 <unk> 索引占位
            src_oov_map.append(oov_idx)       # 记录 OOV 词在扩展词表中的相对位置
    return src_indices, oov_dict, src_oov_map


# [MODIFIED] 修复了参数类型转换问题
def load_pgct_model(checkpoint_path: str, vocab_size: int, pad_idx: int, device: torch.device, config: Dict = None) -> tuple[PGCTModel, Dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = config.get('model', {}) if config else {}
    # 从 Checkpoint 中获取训练时的模型配置 (优先级最低)
    checkpoint_model_cfg = checkpoint.get('config', {}).get('model', {}) 
    
    # 辅助函数：安全地获取整数或浮点数参数
    def safe_get_param(key, default_val, is_int=True):
        # 优先级：配置文件 > Checkpoint配置 > 硬编码默认值
        val = cfg.get(key)
        if val is None:
             val = checkpoint_model_cfg.get(key)
             if val is None:
                 val = default_val
        
        try:
            # 强制转换为 int 或 float
            return int(val) if is_int else float(val)
        except (TypeError, ValueError):
            # 报告错误并返回默认值
            print(f"⚠️ 模型参数 {key} ('{val}') 读取或转换失败，使用默认值 {default_val}。")
            return default_val

    # 构造模型配置字典，并确保所有值都是 int 或 float
    model_cfg = {
        'embed_size': safe_get_param('embed_size', 512),
        'hidden_size': safe_get_param('hidden_size', 512),
        'num_encoder_layers': safe_get_param('num_encoder_layers', 3),
        'num_decoder_layers': safe_get_param('num_decoder_layers', 3),
        'nhead': safe_get_param('nhead', 8),
        'dropout': safe_get_param('dropout', 0.1, is_int=False), # 浮点数
        'cov_loss_weight': safe_get_param('cov_loss_weight', 1.0, is_int=False), # 浮点数
        'max_src_len': safe_get_param('max_src_len', 400),
        'max_tgt_len': safe_get_param('max_tgt_len', 100)
    }

    model = PGCTModel(
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
        # 兼容直接保存 state_dict 的情况 (例如 best_model.pt)
        model.load_state_dict(checkpoint)

    model.eval()
    return model, model_cfg


def summarize_single_text(model: PGCTModel, vocab: Vocab, article: str, device: torch.device, max_src_len: int, max_tgt_len: int, decode_strategy: str, beam_size: int) -> str:
    """对单篇文本进行摘要生成"""
    tokens = tokenize(article)[:max_src_len]
    src_len = len(tokens)
    src_indices, oov_dict, src_oov_map = process_oov(tokens, vocab)

    # 填充到最大源长度
    pad_len = max_src_len - len(src_indices)
    src_indices += [vocab.pad_idx] * pad_len
    src_oov_map += [-1] * pad_len

    # 转换为 Tensor
    src_tensor = torch.LongTensor([src_indices]).to(device)
    src_len_tensor = torch.LongTensor([src_len]).to(device)
    src_oov_tensor = torch.LongTensor([src_oov_map]).to(device)

    with torch.no_grad():
        if decode_strategy == 'beam':
            pred_ids, _ = pgct_beam_search_decode(
                model, src_tensor, src_len_tensor, src_oov_tensor, 
                max_tgt_len, vocab.sos_idx, vocab.eos_idx, beam_size, device
            )
        else:
            pred_ids, _ = pgct_greedy_decode(
                model, src_tensor, src_len_tensor, src_oov_tensor, 
                max_tgt_len, vocab.sos_idx, vocab.eos_idx, device
            )

    summary_tokens = []
    # 解码结果转换为文本
    for idx in pred_ids.squeeze().tolist():
        idx_val = idx if isinstance(idx, int) else idx.item()
        if idx_val < len(vocab):
            # 基础词表中的词
            token = vocab.idx2word.get(idx_val, vocab.UNK_TOKEN)
        else:
            # 扩展词表中的 OOV 词
            oov_rel_idx = idx_val - len(vocab)
            token = oov_dict.get(oov_rel_idx, vocab.UNK_TOKEN)
        
        # 过滤特殊 token
        if token not in [vocab.PAD_TOKEN, vocab.SOS_TOKEN, vocab.EOS_TOKEN]:
            summary_tokens.append(token)

    return ' '.join(summary_tokens)


def collect_input_files(input_path: str) -> List[Path]:
    """收集输入路径下的所有 .txt 文件"""
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
    start_time = time.time() # 记录开始时间

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ 使用设备：{device}")

    config = {}
    if args.config:
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ 加载配置文件：{args.config}")
        except Exception as e:
            print(f"⚠️ 配置文件加载失败，使用默认参数。错误：{e}")

    data_cfg = config.get('data', {})
    eval_cfg = config.get('eval', {})

    # -------------------------------------------------------------------------
    # 关键修改1：修正 vocab_path 优先级，强制转为绝对路径
    # 优先级：命令行输入 > 配置文件 > 默认值
    vocab_path = args.vocab_path  # 优先使用用户命令行传入的路径
    if not vocab_path:  # 若用户未传入（None 或空字符串），从配置文件读取
        vocab_path = data_cfg.get('vocab_path')
    if not vocab_path:  # 最后使用默认相对路径（从 src 目录出发）
        vocab_path = '../data/processed/vocab.json'
    
    # 强制转为绝对路径（彻底解决相对路径歧义）
    vocab_path = Path(vocab_path).resolve()
    print(f"📌 最终词表路径（绝对路径）：{vocab_path}")
    print(f"📌 词表文件是否存在：{vocab_path.exists()}")
    # -------------------------------------------------------------------------

    # 处理 checkpoint 路径（保持原有逻辑，增加绝对路径转换）
    checkpoint_path = args.checkpoint or config.get('train', {}).get('save_dir', '../checkpoints_pgct')
    if Path(checkpoint_path).is_dir():
         checkpoint_path = str(Path(checkpoint_path) / 'best_model.pt')
    checkpoint_path = Path(checkpoint_path).resolve()  # 也转为绝对路径，避免模型加载失败

    input_path = args.input
    decode_strategy = args.decode_strategy or eval_cfg.get('decode_strategy', 'greedy')
    beam_size = args.beam_size or eval_cfg.get('beam_size', 5)
    
    # 组合输出路径（转为绝对路径）
    output_dir = eval_cfg.get('output_dir', '../outputs_pgct')
    output_name = eval_cfg.get('output_file', 'test_summaries.txt')
    output_file = args.output or os.path.join(output_dir, output_name)
    output_file = Path(output_file).resolve()

    # -------------------------------------------------------------------------
    # 关键修改2：加载词表前验证路径，明确报错信息
    try:
        if not vocab_path.exists():
            raise FileNotFoundError(f"词表文件不存在（绝对路径：{vocab_path}）")
        # 确保传入字符串路径（兼容 Vocab.load 可能的格式要求）
        vocab = Vocab.load(str(vocab_path))
        print(f"✅ 加载词表：{vocab_path}（大小：{len(vocab)}）")
    except Exception as e:
        print(f"❌ 词表加载失败：{e}。请检查：1. 路径是否正确；2. 文件是否存在；3. 文件格式是否为 valid JSON。")
        return
    # -------------------------------------------------------------------------

    # 加载模型（使用绝对路径）
    try:
        model, model_cfg = load_pgct_model(str(checkpoint_path), len(vocab), vocab.pad_idx, device, config)
        print(f"✅ 加载模型：{checkpoint_path}")
    except Exception as e:
        print(f"❌ 模型加载或初始化失败。错误：{e}")
        return

    # 收集输入文件
    try:
        input_files = collect_input_files(input_path)
        print(f"✅ 收集到输入文件：{len(input_files)} 个")
    except Exception as e:
        print(f"❌ 输入文件收集失败：{e}")
        return

    # 创建输出目录
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results = []
    print(f"\n🚀 开始批量生成摘要（策略：{decode_strategy}，束大小：{beam_size}）")
    for idx, file in enumerate(input_files, 1):
        # 进度指示器
        print(f"\r处理进度：[{idx}/{len(input_files)}]", end="", flush=True)

        try:
            with open(file, 'r', encoding='utf-8') as f:
                article = f.read().strip()
            if not article:
                continue
        except Exception as e:
            print(f"\n⚠️ 读取失败 [{idx}/{len(input_files)}]：{file.name}，错误：{str(e)[:50]}...，跳过")
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
            print(f"\n⚠️ 生成摘要失败 [{idx}/{len(input_files)}]：{file.name}，错误：{str(e)[:50]}...，跳过该文件")
            continue

        results.append({
            'id': idx,
            'file_name': file.name,
            'file_path': str(file),
            'article_char_count': len(article),
            'summary_token_count': len(summary.split()),
            'summary': summary
        })

    # 预览最后一个生成结果
    if results:
        last_item = results[-1]
        summary = last_item['summary']
        preview = summary[:150] + "..." if len(summary) > 150 else summary
        print(f"\n\n✅ 最后一个完成 | 文件：{last_item['file_name']} | 摘要词数：{last_item['summary_token_count']}")
        print(f"   摘要预览：{preview}")

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("PGCT 模型推理结果汇总\n")
        f.write("=" * 60 + "\n")
        f.write(f"推理配置：\n")
        f.write(f"  - 模型路径：{checkpoint_path}\n")
        f.write(f"  - 词表路径：{vocab_path}\n")
        f.write(f"  - 解码策略：{decode_strategy}\n")
        f.write(f"  - 束搜索大小：{beam_size}\n")
        f.write(f"  - 最大源文本长度：{model_cfg['max_src_len']}\n")
        f.write(f"  - 最大摘要长度：{model_cfg['max_tgt_len']}\n")
        f.write(f"  - 成功处理文件数：{len(results)}\n")
        f.write("=" * 60 + "\n\n")

        for item in results:
            f.write(f"=== 样本 {item['id']} ===\n")
            f.write(f"文件名：{item['file_name']}\n")
            f.write(f"路径：{item['file_path']}\n")
            f.write(f"原文字符数：{item['article_char_count']}\n")
            f.write(f"摘要词数：{item['summary_token_count']}\n")
            f.write(f"生成摘要：{item['summary']}\n\n")

    elapsed = time.time() - start_time
    print(f"\n✅ 推理完成，共处理 {len(results)} 个样本，总用时 {elapsed:.2f} 秒")
    print(f"✓ 结果已保存到：{output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PGCT 模型推理脚本")
    parser.add_argument('--config', type=str, help='配置文件路径（如 ../configs/pgct.yaml）')
    parser.add_argument('--checkpoint', type=str, help='模型 checkpoint 路径')
    parser.add_argument('--vocab_path', type=str, help='词表绝对路径（优先于配置文件，必填）')
    parser.add_argument('--input', type=str, required=True, help='输入 .txt 文件或目录（如 ../data/raw/test）')
    parser.add_argument('--output', type=str, help='结果保存绝对路径（默认从 config 读取）')
    parser.add_argument('--decode_strategy', type=str, choices=['greedy', 'beam'], help='解码策略（默认从 config 读取）')
    parser.add_argument('--beam_size', type=int, help='束搜索大小（仅 beam 策略生效）')
    args = parser.parse_args()
    main(args)
