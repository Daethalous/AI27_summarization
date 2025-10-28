"""
评估指标：ROUGE、重复率、OOV统计
"""
from typing import List, Dict
from collections import Counter
from rouge_score import rouge_scorer


def compute_rouge(
    predictions: List[str],
    references: List[str],
    rouge_types: List[str] = None
) -> Dict[str, float]:
    """计算ROUGE分数
    
    Args:
        predictions: 预测摘要列表（每个是字符串）
        references: 参考摘要列表
        rouge_types: ROUGE类型，默认 ['rouge1', 'rouge2', 'rougeL']
        
    Returns:
        平均ROUGE分数字典
    """
    if rouge_types is None:
        rouge_types = ['rouge1', 'rouge2', 'rougeL']
    
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    
    scores = {f'{rt}_f': [] for rt in rouge_types}
    scores.update({f'{rt}_p': [] for rt in rouge_types})
    scores.update({f'{rt}_r': [] for rt in rouge_types})
    
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for rt in rouge_types:
            scores[f'{rt}_f'].append(result[rt].fmeasure)
            scores[f'{rt}_p'].append(result[rt].precision)
            scores[f'{rt}_r'].append(result[rt].recall)
    
    # 计算平均值
    avg_scores = {k: sum(v) / len(v) if len(v) > 0 else 0.0 for k, v in scores.items()}
    
    return avg_scores


def compute_repetition_rate(tokens: List[str], n: int = 3) -> float:
    """计算n-gram重复率
    
    Args:
        tokens: token列表
        n: n-gram大小
        
    Returns:
        重复率（0-1之间）
    """
    if len(tokens) < n:
        return 0.0
    
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    if len(ngrams) == 0:
        return 0.0
    
    unique_count = len(set(ngrams))
    total_count = len(ngrams)
    
    repetition_rate = 1.0 - (unique_count / total_count)
    
    return repetition_rate


def compute_oov_stats(
    tokens: List[str],
    vocab: set
) -> Dict[str, float]:
    """计算OOV（Out-of-Vocabulary）统计
    
    Args:
        tokens: token列表
        vocab: 词表集合
        
    Returns:
        {
            'oov_count': OOV词数量,
            'oov_rate': OOV率,
            'total_tokens': 总token数
        }
    """
    if len(tokens) == 0:
        return {'oov_count': 0, 'oov_rate': 0.0, 'total_tokens': 0}
    
    oov_count = sum(1 for token in tokens if token not in vocab)
    oov_rate = oov_count / len(tokens)
    
    return {
        'oov_count': oov_count,
        'oov_rate': oov_rate,
        'total_tokens': len(tokens)
    }


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """打印评估指标
    
    Args:
        metrics: 指标字典
        prefix: 前缀（如 "Train", "Val"）
    """
    print(f"\n{prefix} Metrics:")
    print("=" * 60)
    
    # ROUGE分数
    if any('rouge' in k for k in metrics.keys()):
        print("ROUGE Scores:")
        for key in sorted(metrics.keys()):
            if 'rouge' in key and key.endswith('_f'):
                rouge_type = key.replace('_f', '')
                print(f"  {rouge_type.upper():10s}: {metrics[key]:.4f}")
    
    # 其他指标
    other_metrics = {k: v for k, v in metrics.items() if 'rouge' not in k}
    if other_metrics:
        print("\nOther Metrics:")
        for key, value in other_metrics.items():
            print(f"  {key:20s}: {value:.4f}")
    
    print("=" * 60)


def batch_compute_metrics(
    predictions: List[List[str]],
    references: List[List[str]],
    vocab: set = None
) -> Dict[str, float]:
    """批量计算所有指标
    
    Args:
        predictions: 预测token列表的列表
        references: 参考token列表的列表
        vocab: 词表（可选，用于OOV统计）
        
    Returns:
        汇总的指标字典
    """
    # 转换为字符串用于ROUGE计算
    pred_strings = [' '.join(tokens) for tokens in predictions]
    ref_strings = [' '.join(tokens) for tokens in references]
    
    # ROUGE
    metrics = compute_rouge(pred_strings, ref_strings)
    
    # 重复率
    repetition_rates = [compute_repetition_rate(tokens) for tokens in predictions]
    metrics['avg_repetition_rate'] = sum(repetition_rates) / len(repetition_rates) if repetition_rates else 0.0
    
    # OOV（如果提供了vocab）
    if vocab is not None:
        all_oov = [compute_oov_stats(tokens, vocab) for tokens in predictions]
        metrics['avg_oov_rate'] = sum(s['oov_rate'] for s in all_oov) / len(all_oov) if all_oov else 0.0
    
    # 平均长度
    metrics['avg_pred_length'] = sum(len(tokens) for tokens in predictions) / len(predictions) if predictions else 0.0
    metrics['avg_ref_length'] = sum(len(tokens) for tokens in references) / len(references) if references else 0.0
    
    return metrics
