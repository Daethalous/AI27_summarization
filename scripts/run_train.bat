@echo off
REM 训练 Seq2Seq + Attention 模型

cd /d %~dp0..\src

python train.py ^
    --config ../configs/seq2seq_attn.yaml ^
    --data_dir ../../data/raw ^
    --vocab_path ../data/processed/vocab.json ^
    --save_dir ../checkpoints ^
    --batch_size 32 ^
    --epochs 10 ^
    --lr 0.0001 ^
    --num_workers 0

pause
