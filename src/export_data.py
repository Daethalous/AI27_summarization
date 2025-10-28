"""
从 Hugging Face datasets 导出 CNN/DailyMail 原始数据
（可选工具，如果已有tokenized数据则不需要）
"""
import os
import argparse
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def export_to_text_files(dataset, output_dir, split_name):
    """将数据集导出为文本文件
    
    Args:
        dataset: HuggingFace dataset对象
        output_dir: 输出目录
        split_name: 数据集划分名称 (train/validation/test)
    """
    output_path = Path(output_dir) / split_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"导出 {split_name} 数据到 {output_path}...")
    
    for i, example in enumerate(tqdm(dataset)):
        article = example['article']
        summary = example['highlights']
        
        # 创建文件
        filename = f"{split_name}_{i:06d}.txt"
        filepath = output_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=== ARTICLE ===\n")
            f.write(article)
            f.write("\n\n=== SUMMARY ===\n")
            f.write(summary)
    
    print(f"✓ 导出完成: {len(dataset)} 个样本")


def main(args):
    """主函数"""
    
    print(f"加载 CNN/DailyMail 数据集 (版本: {args.version})...")
    dataset = load_dataset('cnn_dailymail', args.version)
    
    print(f"数据集统计:")
    print(f"  - Train: {len(dataset['train'])} 样本")
    print(f"  - Validation: {len(dataset['validation'])} 样本")
    print(f"  - Test: {len(dataset['test'])} 样本")
    
    # 导出各个划分
    if args.export_train:
        export_to_text_files(dataset['train'], args.output_dir, 'train')
    
    if args.export_val:
        export_to_text_files(dataset['validation'], args.output_dir, 'validation')
    
    if args.export_test:
        export_to_text_files(dataset['test'], args.output_dir, 'test')
    
    print(f"\n✓ 所有数据已导出到: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='导出 CNN/DailyMail 原始数据')
    
    parser.add_argument('--output_dir', type=str, default='./data/raw',
                        help='输出目录')
    parser.add_argument('--version', type=str, default='3.0.0',
                        help='数据集版本')
    parser.add_argument('--export_train', action='store_true', default=True,
                        help='导出训练集')
    parser.add_argument('--export_val', action='store_true', default=True,
                        help='导出验证集')
    parser.add_argument('--export_test', action='store_true', default=True,
                        help='导出测试集')
    
    args = parser.parse_args()
    main(args)
