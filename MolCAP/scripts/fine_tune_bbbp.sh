#!/bin/bash

python entry.py\
  --mode prompt_gin_classification \
  --dataset data/finetune_data/bbbp \
  --task_name bbbp \
  --warmup_updates 1000 \
  --tot_updates 5000000000 \
  --peak_lr 2e-4 \
  --seed 145 \
  --end_lr 1e-9 \
  --batch_size 256 \
  --epochs 500 \
  --cuda 1 \
  --dropout 0.3 \
  --max_single_hop 4 \
  --log_dir tb_lgs \
  --not_fast_read \
  --patience 50 \
  --acc_batches 1 \
  --d_model 256 \
  --dim_feedforward 512 \
  --num_encoder_layer 12 \
  --warmup_updates 2e3 \
  --norm_first \
  --nhead 16 \
  --max_single_hop 4 \
  --model_path /mnt/sdb/home/wy/IMBioLG/tb_lgs/uspto_1/version_14/checkpoints/epoch=9-step=55030.ckpt
#  --split_type scaffold
