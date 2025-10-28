@echo off
REM 评估 Seq2Seq + Attention 模型

cd /d %~dp0..\src

python eval.py ^
    --config ../configs/seq2seq_attn.yaml ^
    --checkpoint ../checkpoints/best_model.pt ^
    --data_dir ../../data/raw ^
    --vocab_path ../data/processed/vocab.json ^
    --split test ^
    --batch_size 32 ^
    --decode_strategy greedy ^
    --output ../outputs/test_summaries.json ^
    --metrics_output ../outputs/test_metrics.json ^
    --show_examples 5

pause
