#!/bin/bash

python entry.py\
  --mode finetune_classification \
  --dataset data/finetune_data/bace \
  --task_name bace \
  --warmup_updates 10 \
  --tot_updates 5000000000 \
  --peak_lr 2e-4 \
  --end_lr 5e-5 \
  --seed 4545 \
  --batch_size 256 \
  --epochs 500 \
  --cuda 3 \
  --dropout 0.35 \
  --log_dir tb_lgs \
  --not_fast_read \
  --patience 50 \
  --val_every_n_epoch 1 \
  --prompt_mol "C[C@H](N)C(=O)O" \
  --model_path /mnt/sdb/home/wy/IMBioLG/tb_lgs/uspto_1/version_13/checkpoints/epoch=4-step=27515.ckpt
