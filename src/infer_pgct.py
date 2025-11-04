"""
PGCT 模型推理脚本（单条或批量文本文件）
适配 Transformer + Pointer-Generator + Coverage 模型
"""
import os
import sys
import nltk
from pathlib import Path
import argparse
from typing import List, Dict

import torch
import yaml
from nltk.tokenize import word_tokenize

# 确保 NLTK 分词资源存在
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 添加项目根路径到环境变量
sys.path.insert(0, str(Path(__file__).parent))

# 关键修复1：导入 PGCT 专用解码函数（替换通用解码）
from models.pgct.pgct_model import PGCTModel
from models.pgct.pgct_decoding import pgct_greedy_decode, pgct_beam_search_decode
from utils.vocab import Vocab


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    """文本分词（与训练预处理逻辑一致）"""
    if lowercase:
        text = text.lower()
    return word_tokenize(text)


def process_oov(tokens: List[str], vocab: Vocab) -> tuple[List[int], Dict[int, str], List[int]]:
    """
    关键修复2：修正 OOV 处理逻辑
    返回：
    - src_indices：源文本索引（OOV 用 UNK 索引）
    - oov_dict：OOV 词映射（相对索引→词）
    - src_oov_map：OOV 相对索引（非 OOV 填 -1）
    """
    src_indices = []
    oov_dict = {}  # 相对索引从 0 开始，匹配 PG 模型逻辑
    src_oov_map = []

    for token in tokens:
        if token in vocab.word2idx:
            # 普通词：用基础词表索引，OOV 映射填 -1
            src_indices.append(vocab.word2idx[token])
            src_oov_map.append(-1)
        else:
            # OOV 词：分配相对索引，基础词表用 UNK
            if token not in oov_dict.values():
                new_oov_idx = len(oov_dict)  # 修复：去掉 +1，避免索引偏移
                oov_dict[new_oov_idx] = token
            oov_idx = [k for k, v in oov_dict.items() if v == token][0]
            src_indices.append(vocab.unk_idx)
            src_oov_map.append(oov_idx)

    return src_indices, oov_dict, src_oov_map


def load_pgct_model(
    checkpoint_path: str,
    vocab_size: int,
    pad_idx: int,
    device: torch.device,
    config: Dict = None
) -> PGCTModel:
    """
    关键修复3：兼容两种 checkpoint 格式（带 model_state_dict / 纯参数）
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 优先从 config 获取参数，其次从 checkpoint，最后用默认值
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

    # 初始化 PGCT 模型
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

    # 兼容两种 checkpoint 格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)  # 纯参数文件（如 best_model.pt）

    model.eval()
    return model, model_cfg


def summarize_text(
    model: PGCTModel,
    vocab: Vocab,
    article: str,
    device: torch.device,
    max_src_len: int,
    max_tgt_len: int,
    decode_strategy: str = 'greedy',
    beam_size: int = 5
) -> str:
    """单条文本生成摘要（调用 PGCT 专用解码）"""
    # 1. 预处理：分词 + 截断
    tokens = tokenize(article)[:max_src_len]
    src_len = len(tokens)
    src_indices, oov_dict, src_oov_map = process_oov(tokens, vocab)

    # 2. Padding 到 max_src_len
    if len(src_indices) < max_src_len:
        pad_len = max_src_len - len(src_indices)
        src_indices += [vocab.pad_idx] * pad_len
        src_oov_map += [-1] * pad_len  # 非 OOV 位置补 -1

    # 3. 转换为张量（batch_size=1）
    src_tensor = torch.LongTensor([src_indices]).to(device)
    src_len_tensor = torch.LongTensor([src_len]).to(device)
    src_oov_tensor = torch.LongTensor([src_oov_map]).to(device)

    # 4. 模型推理（调用 PGCT 专用解码）
    with torch.no_grad():
        if decode_strategy == 'beam':
            pred_ids, _ = pgct_beam_search_decode(
                model=model,
                src=src_tensor,
                src_lens=src_len_tensor,
                src_oov_map=src_oov_tensor,
                max_length=max_tgt_len,
                sos_idx=vocab.sos_idx,
                eos_idx=vocab.eos_idx,
                beam_size=beam_size,
                device=device
            )
        else:
            pred_ids, _ = pgct_greedy_decode(
                model=model,
                src=src_tensor,
                src_lens=src_len_tensor,
                src_oov_map=src_oov_tensor,
                max_length=max_tgt_len,
                sos_idx=vocab.sos_idx,
                eos_idx=vocab.eos_idx,
                device=device
            )

    # 5. 索引转文本（处理 OOV）
    summary_tokens = []
    for idx in pred_ids.squeeze().tolist():
        idx_val = idx if isinstance(idx, int) else idx.item()
        if idx_val < len(vocab):
            token = vocab.idx2word.get(idx_val, vocab.unk_token)
        else:
            # OOV 词：相对索引 = 预测索引 - 基础词表大小
            oov_rel_idx = idx_val - len(vocab)
            token = oov_dict.get(oov_rel_idx, vocab.unk_token)
        # 跳过特殊符号
        if token not in [vocab.pad_token, vocab.sos_token, vocab.eos_token]:
            summary_tokens.append(token)

    return ' '.join(summary_tokens)


def collect_inputs(input_path: str) -> List[Path]:
    """收集输入路径下的所有 .txt 文件（支持单文件/目录）"""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到输入路径: {input_path}")

    if path.is_dir():
        # 递归查找所有 .txt 文件并排序
        txt_files = sorted([p for p in path.glob('*.txt') if p.is_file()])
        if not txt_files:
            raise FileNotFoundError(f"目录下无 .txt 文件: {input_path}")
        return txt_files
    elif path.is_file() and path.suffix == '.txt':
        return [path]
    else:
        raise ValueError(f"输入必须是 .txt 文件或目录: {input_path}")


def main(args):
    # 设备初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载配置文件
    config = {}
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✓ 加载配置文件: {args.config}")

    # 加载词表
    vocab_path = args.vocab_path or config.get('data', {}).get('vocab_path', '../data/processed/vocab.json')
    if not Path(vocab_path).exists():
        raise FileNotFoundError(f"词表文件不存在: {vocab_path}")
    vocab = Vocab.load(vocab_path)
    print(f"✓ 加载词表: {vocab_path}（大小: {len(vocab)}）")

    # 加载模型
    checkpoint_path = args.checkpoint or config.get('train', {}).get('save_dir', '../checkpoints_pgct/best_model.pt')
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"模型 checkpoint 不存在: {checkpoint_path}")
    model, model_cfg = load_pgct_model(
        checkpoint_path=checkpoint_path,
        vocab_size=len(vocab),
        pad_idx=vocab.pad_idx,
        device=device,
        config=config
    )
    print(f"✓ 加载模型: {checkpoint_path}")

    # 收集输入文件
    try:
        input_files = collect_inputs(args.input)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ 输入处理失败: {e}")
        sys.exit(1)
    print(f"✓ 收集到 {len(input_files)} 个输入文件")

    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 批量生成摘要
    results = []
    print(f"\n--- 开始生成摘要（策略: {args.decode_strategy}）---")
    for idx, file in enumerate(input_files, 1):
        # 读取文章（提取 === ARTICLE === 后的正文，忽略参考摘要）
        with file.open('r', encoding='utf-8') as f:
            article = f.read().strip()
        # 适配 raw 目录下的文件格式（分离 ARTICLE 和 SUMMARY）
        if '=== ARTICLE ===' in article:
            article = article.split('=== ARTICLE ===')[1].split('=== SUMMARY ===')[0].strip()

        # 生成摘要
        summary = summarize_text(
            model=model,
            vocab=vocab,
            article=article,
            device=device,
            max_src_len=model_cfg['max_src_len'],
            max_tgt_len=model_cfg['max_tgt_len'],
            decode_strategy=args.decode_strategy,
            beam_size=args.beam_size
        )

        results.append({'id': idx, 'file': str(file), 'summary': summary})
        print(f"[{idx}/{len(input_files)}] 完成: {file.name}")
        print(f"  摘要预览: {summary[:150]}...\n")

    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(f"=== 样本 {item['id']} ===\n")
            f.write(f"文件路径: {item['file']}\n")
            f.write(f"生成摘要: {item['summary']}\n\n")
    print(f"✓ 所有结果已保存到: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PGCT 模型推理脚本")
    # 配置与模型
    parser.add_argument('--config', type=str, help='配置文件路径（如 ../configs/pgct.yaml）')
    parser.add_argument('--checkpoint', type=str, help='模型 checkpoint 路径')
    parser.add_argument('--vocab_path', type=str, help='词表路径（默认从 config 读取）')
    # 输入输出
    parser.add_argument('--input', type=str, required=True, help='输入 .txt 文件或目录（如 ../data/raw/test）')
    parser.add_argument('--output', type=str, default='../outputs_pgct/infer_results.txt', help='结果保存路径')
    # 解码参数
    parser.add_argument('--decode_strategy', type=str, default='greedy', choices=['greedy', 'beam'], help='解码策略')
    parser.add_argument('--beam_size', type=int, default=5, help='束搜索大小（仅 beam 策略生效）')
    args = parser.parse_args()
    main(args)
