"""
独立的数据下载与原始数据格式转换脚本。

用于在运行 quick_test_train.py 或 train_*.py 之前，
确保 raw data/raw/cnn_dailymail 存在，从而避免 prepare_datasets 报错。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List
from tqdm import tqdm

# 确保能导入项目路径中的依赖
sys.path.insert(0, str(Path(__file__).parent.parent))

def ensure_raw_dataset(raw_dir: str, dataset_version: str = '3.0.0') -> None:
    """保证原始 CNN/DailyMail 文本存在，缺失时尝试自动下载。"""
    raw_dir_path = Path(raw_dir)
    raw_dir_path.mkdir(parents=True, exist_ok=True)

    expected_splits = ['train', 'validation', 'test']
    missing_splits = []

    # 检查原始数据目录是否为空或缺失
    for split in expected_splits:
        split_path = raw_dir_path / split

        # 只有在目录下没有 .txt 文件时才算缺失
        if not split_path.is_dir() or not any(split_path.glob('*.txt')):
            missing_splits.append(split)

    if not missing_splits:
        print("✓ 原始数据目录已存在且包含文件，跳过下载。")
        return

    print(
        "🚨 检测到原始数据缺失: "
        f"{missing_splits} 划分，开始从 Hugging Face 下载 CNN/DailyMail ({dataset_version})..."
    )

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "🛑 缺少 CNN/DailyMail 原始数据，且未安装 `datasets` 库，无法自动下载。\n"
            "请执行 `pip install datasets` 或手动将数据放置在 data/raw 目录下。"
        ) from exc

    # 统一使用 'cnn_dailymail'
    dataset = load_dataset('cnn_dailymail', dataset_version)
    print("下载完成，开始导出为项目所需格式...")

    for split in missing_splits:
        # HuggingFace split name: 'validation' for 'val', others match
        hf_split = 'validation' if split == 'validation' else split
        subset = dataset[hf_split]
        split_path = raw_dir_path / split
        split_path.mkdir(parents=True, exist_ok=True)
        print(f"导出 {split} 划分，共 {len(subset)} 个样本...")

        # tqdm 包装迭代器以显示进度条
        for idx, example in enumerate(tqdm(subset, desc=f"Writing {split}", unit='sample')):
            filename = f"{split}_{idx:06d}.txt"
            filepath = split_path / filename
            article = example['article'].strip()
            # CNN/DailyMail 使用 'highlights' 字段作为摘要
            summary = example['highlights'].strip()

            with filepath.open('w', encoding='utf-8') as f:
                f.write("=== ARTICLE ===\n")
                f.write(article)
                f.write("\n\n=== SUMMARY ===\n")
                f.write(summary)

        print(f"✓ {split} 划分导出完成: {split_path}")

    print("🎉 已完成 CNN/DailyMail 数据集下载与导出。")

def main():
    # 假设项目结构是 AI27_summarization/src/utils/download_raw_data.py
    project_root = Path(__file__).parent.parent.parent.resolve()
    raw_data_dir = str(project_root / 'data' / 'raw')

    # 确保 datasets 库已安装
    try:
        import datasets
    except ImportError:
        print("🚨 正在安装 datasets 库...")
        os.system(f"{sys.executable} -m pip install datasets")
        import datasets

    # 检查是否在正确的目录，并切换
    if Path.cwd() != project_root:
        print(f"切换到项目根目录: {project_root}")
        os.chdir(project_root)

    ensure_raw_dataset(raw_data_dir, dataset_version='3.0.0')

if __name__ == '__main__':
    main()
