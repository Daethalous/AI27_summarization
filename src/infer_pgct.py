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

# 确保 NLTK 分词资源存在
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 添加项目根路径到环境变量，确保能导入内部模块
sys.path.insert(0, str(Path(__file__).parent))

# 关键修改1：导入 PGCT 专用解码函数（替换通用解码）
from models.pgct.pgct_model import PGCTModel
from models.pgct.pgct_decoding import pgct_greedy_decode, pgct_beam_search_decode
from utils.vocab import Vocab


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    """文本分词（与数据预处理逻辑一致）"""
    if lowercase:
        text = text.lower()
    return word_tokenize(text)


def process_oov(tokens: List[str], vocab: Vocab) -> tuple[List[int], Dict[int, str], List[int]]:
    """
    处理 OOV 词，生成：
    - src_indices：源文本在基础词表中的索引（OOV 用 UNK 索引）
    - oov_dict：OOV 词映射（key：相对索引，value：OOV 词）
    - src_oov_map：OOV 词的相对索引（与 src_indices 长度一致，非 OOV 填 -1）
    """
    src_indices = []  # 基础词表索引（含 UNK）
    oov_dict = {}     # 存储 OOV 词：{相对索引: OOV词}（相对索引从 0 开始）
    src_oov_map = []  # 对应 src_indices，OOV 词填相对索引，普通词填 -1

    for token in tokens:
        if token in vocab.word2idx:
            # 普通词：用基础词表索引，OOV 映射填 -1
            src_indices.append(vocab.word2idx[token])
            src_oov_map.append(-1)
        else:
            # OOV 词：分配相对索引，基础词表索引用 UNK
            if token not in oov_dict.values():
                new_oov_idx = len(oov_dict)  # 相对索引从 0 开始
                oov_dict[new_oov_idx] = token
            oov_idx = [k for k, v in oov_dict.items() if v == token][0]
            src_indices.append(vocab.unk_idx)  # 基础词表用 UNK
            src_oov_map.append(oov_idx)        # OOV 映射填相对索引

    return src_indices, oov_dict, src_oov_map


def load_pgct_model(
    checkpoint_path: str, 
    vocab_size: int, 
    pad_idx: int, 
    device: torch.device,
    config: Dict = None
) -> tuple[PGCTModel, Dict]:
    """加载 PGCT 模型（兼容带 config 和纯参数的 checkpoint）"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 优先从外部 config 获取参数，其次从 checkpoint 内的 config，最后用默认值
    cfg = config.get('model', {}) if config else {}
    checkpoint_cfg = checkpoint.get('config', {})

    # 模型核心参数（与 PGCTModel 初始化参数严格对齐）
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

    # 加载模型参数（兼容两种 checkpoint 格式）
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)  # 纯参数文件（如 best_model.pt）

    model.eval()  # 推理模式：关闭 Dropout 和 BatchNorm
    return model, model_cfg


def summarize_single_text(
    model: PGCTModel,
    vocab: Vocab,
    article: str,
    device: torch.device,
    max_src_len: int = 400,
    max_tgt_len: int = 100,
    decode_strategy: str = 'greedy',
    beam_size: int = 5
) -> str:
    """单条文本生成摘要（核心推理逻辑）"""
    # 1. 文本预处理：分词 + 截断（与训练时一致）
    tokens = tokenize(article)[:max_src_len]  # 截断到最大源文本长度
    src_len = len(tokens)  # 实际有效长度（不含 padding）

    # 2. OOV 处理：生成基础索引、OOV 映射（适配 Pointer-Generator）
    src_indices, oov_dict, src_oov_map = process_oov(tokens, vocab)

    # 3. Padding：补到 max_src_len 长度（保证输入维度一致）
    if len(src_indices) < max_src_len:
        pad_len = max_src_len - len(src_indices)
        src_indices += [vocab.pad_idx] * pad_len  # 基础索引补 PAD
        src_oov_map += [-1] * pad_len             # OOV 映射补 -1（标记非 OOV）

    # 4. 转换为张量（batch_size=1，适配模型输入格式）
    src_tensor = torch.LongTensor([src_indices]).to(device)  # [1, max_src_len]
    src_len_tensor = torch.LongTensor([src_len]).to(device)  # [1]（有效长度，用于掩码）
    src_oov_tensor = torch.LongTensor([src_oov_map]).to(device)  # [1, max_src_len]

    # 5. 模型推理：调用 PGCT 专用解码函数
    with torch.no_grad():  # 关闭梯度计算，节省显存并加速
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

    # 6. 结果转换：索引 → 文本（处理 OOV 词，还原原始词汇）
    summary_tokens = []
    for idx in pred_ids.squeeze().tolist():  # 压缩 batch 维度，转为列表
        idx_val = idx if isinstance(idx, int) else idx.item()
        if idx_val < len(vocab):
            # 普通词：从基础词表获取
            token = vocab.idx2word.get(idx_val, vocab.unk_token)
        else:
            # OOV 词：计算相对索引（idx_val = 基础词表大小 + 相对索引）
            oov_rel_idx = idx_val - len(vocab)
            token = oov_dict.get(oov_rel_idx, vocab.unk_token)
        # 跳过特殊符号（PAD/SOS/EOS，不加入最终摘要）
        if token not in [vocab.pad_token, vocab.sos_token, vocab.eos_token]:
            summary_tokens.append(token)

    # 7. 拼接为完整摘要字符串
    return ' '.join(summary_tokens)


def collect_input_files(input_path: str) -> List[Path]:
    """收集输入路径下的所有 .txt 文件（支持单文件或目录）"""
    input_path_obj = Path(input_path)
    if not input_path_obj.exists():
        raise FileNotFoundError(f"输入路径不存在：{input_path}")
    
    if input_path_obj.is_file():
        if input_path_obj.suffix == '.txt':
            return [input_path_obj]
        else:
            raise ValueError(f"输入文件必须是 .txt 格式（当前：{input_path_obj.suffix}）")
    
    if input_path_obj.is_dir():
        # 递归查找目录下所有 .txt 文件，按路径排序确保结果稳定
        txt_files = sorted(list(input_path_obj.rglob('*.txt')))
        if not txt_files:
            raise FileNotFoundError(f"目录下未找到任何 .txt 文件：{input_path}")
        return txt_files

    raise TypeError(f"输入路径必须是文件或目录（当前：{input_path_obj.stat().st_mode}）")


def main(args):
    # 记录推理开始时间
    start_time = time.time()

    # 1. 设备初始化（自动检测 GPU/CPU，优先用 GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ 使用设备：{device}")

    # 2. 加载配置文件（若指定）
    config = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在：{args.config}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ 加载配置文件：{args.config}")

    # 3. 加载词表（必须与训练时使用的词表一致）
    vocab_path = args.vocab_path or config.get('data', {}).get('vocab_path', '../data/processed/vocab.json')
    vocab_path_obj = Path(vocab_path)
    if not vocab_path_obj.exists():
        raise FileNotFoundError(f"词表文件不存在：{vocab_path}")
    vocab = Vocab.load(str(vocab_path_obj))
    print(f"✅ 加载词表：{vocab_path}（词表大小：{len(vocab)}）")

    # 4. 加载 PGCT 模型（核心步骤）
    checkpoint_path = args.checkpoint or config.get('train', {}).get('save_dir', '../checkpoints_pgct/best_model.pt')
    checkpoint_path_obj = Path(checkpoint_path)
    if not checkpoint_path_obj.exists():
        raise FileNotFoundError(f"模型 checkpoint 不存在：{checkpoint_path}")
    model, model_cfg = load_pgct_model(
        checkpoint_path=str(checkpoint_path_obj),
        vocab_size=len(vocab),
        pad_idx=vocab.pad_idx,
        device=device,
        config=config
    )
    print(f"✅ 加载模型：{checkpoint_path}")
    print(f"  - 模型参数：hidden_size={model_cfg['hidden_size']}, encoder_layers={model_cfg['num_encoder_layers']}")
    print(f"  - 推理配置：max_src_len={model_cfg['max_src_len']}, max_tgt_len={model_cfg['max_tgt_len']}")

    # 5. 收集输入文件（单文件或目录下所有 .txt）
    try:
        input_files = collect_input_files(args.input)
    except (FileNotFoundError, ValueError, TypeError) as e:
        print(f"❌ 输入文件处理失败：{e}")
        sys.exit(1)
    print(f"✅ 收集到输入文件：{len(input_files)} 个")
    for i, file in enumerate(input_files[:3], 1):  # 打印前 3 个文件示例（避免输出过长）
        print(f"  {i}. {file.name}（路径：{str(file.parent)[:50]}...）")
    if len(input_files) > 3:
        print(f"  ... 还有 {len(input_files)-3} 个文件未显示")

    # 6. 创建输出目录（确保输出路径存在，避免保存失败）
    output_path_obj = Path(args.output)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)  # 递归创建父目录

    # 7. 批量生成摘要（核心推理循环）
    results = []  # 存储所有推理结果（用于后续保存）
    print(f"\n🚀 开始批量生成摘要（解码策略：{args.decode_strategy}，束大小：{args.beam_size}）")
    for idx, file in enumerate(input_files, 1):
        # 读取输入文本（假设 .txt 文件直接存储纯文章内容，无特殊格式）
        try:
            with open(file, 'r', encoding='utf-8') as f:
                article = f.read().strip()
            if not article:
                print(f"⚠️ 跳过空文件 [{idx}/{len(input_files)}]：{file.name}")
                continue
        except Exception as e:
            print(f"⚠️ 读取文件失败 [{idx}/{len(input_files)}]：{file.name}，错误：{str(e)[:30]}...，跳过该文件")
            continue

        # 调用函数生成摘要
        try:
            summary = summarize_single_text(
                model=model,
                vocab=vocab,
                article=article,
                device=device,
                max_src_len=model_cfg['max_src_len'],
                max_tgt_len=model_cfg['max_tgt_len'],
                decode_strategy=args.decode_strategy,
                beam_size=args.beam_size
            )
        except Exception as e:
            print(f"⚠️ 生成摘要失败 [{idx}/{len(input_files)}]：{file.name}，错误：{str(e)[:30]}...，跳过该文件")
            continue

        # 记录结果（包含关键信息，便于后续分析）
        results.append({
            'id': idx,
            'file_name': file.name,
            'file_path': str(file),
            'article_char_count': len(article),  # 文章字符数
            'summary_token_count': len(summary.split()),  # 摘要词数
            'summary': summary
        })

        # 打印实时进度（摘要预览限制 150 字符，避免输出过长）
        summary_preview = summary[:150] + "..." if len(summary) > 150 else summary
        print(f"✅ 完成 [{idx}/{len(input_files)}] | 文件：{file.name} | 摘要词数：{len(summary.split())}")
        print(f"   摘要：{summary_preview}")

    # 8. 保存推理结果到文件（文本格式，便于阅读和后续分析）
    with open(output_path_obj, 'w', encoding='utf-8') as f:
        # 写入头部信息（配置+统计）
        f.write("="*60 + "\n")
        f.write("PGCT 模型推理结果汇总\n")
        f.write("="*60 + "\n")
        f.write(f"推理配置：\n")
        f.write(f"  - 模型路径：{checkpoint_path}\n")
        f.write(f"  - 词表路径：{vocab_path}\n")
        f.write(f"  - 解码策略：{args.decode_strategy}\n")
        f.write(f"  - 束搜索大小：{args.beam_size}\n")
        f.write(f"  - 最大源文本长度：{model_cfg['max_src_len']}\n
