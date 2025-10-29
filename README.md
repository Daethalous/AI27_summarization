# Text Summarization with Seq2Seq + Attention

基于 Seq2Seq + Luong Attention 的文本摘要模型，使用 CNN/DailyMail 数据集训练。

## 项目结构

```
summarization/
├── data/
│   ├── raw/                    # 原始数据（可选，由 datasets 拉取）
│   └── processed/              # 处理后的词表等
│       ├── vocab.json
│       ├── train.pkl
│       ├── val.pkl
│       └── test.pkl
├── src/
│   ├── datamodules/
│   │   └── cnndm.py            # 数据加载与预处理
│   ├── models/
│   │   ├── baseline/           # Baseline 模型
│   │   │   ├── __init__.py
|   |   |   ├──encoder.py
│   │   │   ├── decoder.py
│   │   │   └── model.py
│   │   └── pointer_generator/  # 新增 Pointer-Generator 模型
│   │       ├── pg_decoder.py
|   |       ├── pg_model.py
│   │       └── __init__.py
│   ├── utils/
│   │   ├── vocab.py            # 词表工具
│   │   └── metrics.py          # 评估指标
│   ├── train.py                # Baseline 训练脚本
│   ├── train_pg.py             # PG 训练脚本
│   ├── eval.py                 # 评估脚本
│   └── quick_test_train.py     # 快速测试脚本
├── configs/
│   └── seq2seq_attn.yaml       # 配置文件示例
├── checkpoints_baseline/       # Baseline 模型保存目录
├── checkpoints_pg/             # PG 模型保存目录
├── outputs/                    # 评估结果输出
├── requirements.txt
└── README.md
```

### 1. 环境依赖

```bash
pip install -r requirements.txt
```

> 首次运行会自动检查并下载 NLTK `punkt` / `punkt_tab` 资源；若机器无法联网，可手动执行 `python -m nltk.downloader punkt punkt_tab`。

### 2. 数据准备

本项目默认从 `data/raw` 目录读取 CNN/DailyMail 原始文本（`.txt`，包含 `=== ARTICLE ===` 与 `=== SUMMARY ===` 两段）。若目录缺失，`train.py` 会在安装了 `datasets` 库的情况下自动下载 Hugging Face 上的 CNN/DailyMail 数据（`dataset_version` 默认 `3.0.0`，可在配置或命令行调整；将 `auto_download` 设为 `false` 或传入 `--no_auto_download` 可关闭自动下载）。首次运行 `train.py` 或单独调用预处理逻辑时会：

1. 使用 **NLTK** `word_tokenize` 对文本分词，统一转换为小写并截断至 512；
2. 基于训练集构建词表（最大 50k、最小词频 5）并保存到 `data/processed/vocab.json`；
3. 将编码后的样本写入 `train.pkl` / `val.pkl` / `test.pkl`（包含 `<SOS>/<EOS>` 和 padding=0）。

目录示例：

```
summarization/data/
├── raw/
│   ├── train/train_000000.txt
│   ├── validation/validation_000000.txt
│   └── test/test_000000.txt
└── processed/
    ├── vocab.json
    ├── train.pkl
    ├── val.pkl
    └── test.pkl
```

### 3.1 训练baseline模型

```bash
cd src
python train.py --config ../configs/seq2seq_attn.yaml
```

训练默认参数：`batch size=32`、`learning rate=1e-4`、`epoch=10`、`max_src_len=max_tgt_len=512`，每轮指标会自动写入 `logs/baseline.log` 以检查 ROUGE-L 波动是否低于 1%。

### 3.2.1 快速上手： 测试baseline/pointer generator模型(仅使用100个样本)

```bash
cd src
python quick_test_train.py --model baseline --num_samples 100 --num_epochs 2
python quick_test_train.py --model pg --num_samples 100 --num_epochs 2
```
### 3.2.2 训练pointer generator模型(数据集子集 1000个样本)

```bash
cd src
python train_pg.py --data_dir ../data/raw --num_epochs 10 --num_samples 1000
```

### 3.2.3 训练pointer generator模型(完整数据集)

```bash
cd src
python train_pg.py --data_dir ../data/raw --num_epochs 10
```

训练默认参数：`batch size=32`、`learning rate=1e-4`、`epoch=10`、`max_src_len=max_tgt_len=512`，每轮指标会自动写入 `logs/baseline.log` 以检查 ROUGE-L 波动是否低于 1%。

**使用脚本：**
```bash
cd scripts
run_train.bat
```

**自定义参数：**
```bash
cd src
python train.py \
    --data_dir ../data/raw \
    --vocab_path ../data/processed/vocab.json \
    --save_dir ../checkpoints \
    --batch_size 32 \
    --epochs 10 \
    --lr 0.0001 \
    --max_src_len 512 \
    --max_tgt_len 512
```

### 4. 评估模型

**Greedy Decoding：**
```bash
cd src
python eval.py \
    --checkpoint ../checkpoints/best_model.pt \
    --split test \
    --decode_strategy greedy \
    --output ../outputs/test_summaries.json
```

**Beam Search：**
```bash
cd src
python eval.py \
    --checkpoint ../checkpoints/best_model.pt \
    --split test \
    --decode_strategy beam \
    --beam_size 5 \
    --output ../outputs/test_summaries_beam.json
```

**使用脚本：**
```bash
cd scripts
run_eval.bat
```

**使用默认配置：**
```bash
cd src
python eval.py --config ../configs/seq2seq_attn.yaml
```

**自定义参数：**
```bash
cd src
python eval.py \
  --data_dir ../data/raw \
  --model_path ../checkpoints/best_model.pt \
  --vocab_path ../data/processed/vocab.json \
  --batch_size 32 \
  --max_src_len 512 \
  --max_tgt_len 128
```

## 模型架构

### Encoder
- **BiLSTM** (双向LSTM)
- 输入: 词嵌入 (embed_size=256)
- 隐藏层: hidden_size//2 per direction
- 输出: hidden_size=512

### Decoder
- **LSTM** (单向)
- **Luong Attention** (General形式)
- 输入: embedding + context vector
- 输出: vocab distribution

### 训练细节
- **损失函数**: CrossEntropyLoss (忽略padding)
- **优化器**: Adam (lr=1e-4)
- **学习率调度**: ReduceLROnPlateau
- **梯度裁剪**: max_norm=5.0
- **Teacher Forcing**: 0.5概率

## 配置说明

编辑 `configs/seq2seq_attn.yaml`:

```yaml
# 数据
max_vocab_size: 50000      # 最大词表大小
min_freq: 5                # 最小词频
max_src_len: 512           # 最大源文本长度
max_tgt_len: 512           # 最大摘要长度

# 模型
embed_size: 256            # 词嵌入维度
hidden_size: 512           # LSTM隐藏层维度
num_layers: 1              # LSTM层数
dropout: 0.1               # Dropout率

# 训练
batch_size: 32             # 批次大小
num_epochs: 10             # 训练轮数
learning_rate: 0.0001      # 学习率
```

## 评估指标

模型会自动计算以下指标：

- **ROUGE-1/2/L**: F1, Precision, Recall
- **Repetition Rate**: n-gram 重复率
- **OOV Rate**: Out-of-Vocabulary 率
- **Average Length**: 平均生成长度

## 输出文件

### 训练输出

```
checkpoints/
├── best_model.pt              # 最佳模型
├── checkpoint_epoch_5.pt      # 定期检查点
├── checkpoint_epoch_10.pt
└── runs/                      # TensorBoard日志
```

可视化：
```
checkpoints/
└── rouge_l_trend.png          # ROUGE-L 折线图（按 epoch 绘制）
```

### 评估输出

```
outputs/
├── test_summaries.json        # 生成的摘要
└── test_metrics.json          # 评估指标
```

**摘要文件格式：**
```json
[
  {
    "source": "article text...",
    "reference": "reference summary...",
    "prediction": "predicted summary..."
  },
  ...
]
```

## TensorBoard 可视化

```bash
tensorboard --logdir checkpoints/runs
```

查看：
- 训练/验证损失曲线
- ROUGE分数变化
- 学习率调整

## 常见问题

### Q1: 内存不足？
- 减小 `batch_size` (如 16 或 8)
- 减小 `max_src_len` 和 `max_tgt_len`
- 减小 `hidden_size`

### Q2: 训练太慢？
- 使用GPU（自动检测）
- 增加 `num_workers` (多线程加载数据)
- 减少数据量用于快速实验

### Q3: 如何只使用部分数据？
修改 `src/datamodules/cnndm.py` 中的数据集划分比例。

### Q4: 如何从checkpoint继续训练？
修改 `train.py`，添加加载checkpoint的逻辑：
```python
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

## 进阶使用

### 自定义数据集

修改 `src/datamodules/cnndm.py` 中的 `parse_story_file()` 函数以适配您的数据格式。

### 添加Coverage机制

参考 Pointer-Generator 论文，在 Attention 中添加 coverage vector。

## 参考文献

- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) (Luong et al., 2015)
- [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368) (See et al., 2017)

## 许可证

MIT License
