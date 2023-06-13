#!/bin/bash

python entry.py\
  --mode finetune_DDI \
  --dataset data/finetune_data/biosnap \
  --task_name biosnap \
  --warmup_updates 1000 \
  --tot_updates 5000000000 \
  --peak_lr 2e-4 \
  --end_lr 1e-5 \
  --seed 3706 \
  --batch_size 128 \
  --epochs 500 \
  --cuda 3 \
  --dropout 0.3 \
  --log_dir tb_lgs \
  --not_fast_read \
  --patience 50 \
  --split_type random \
  --model_path /mnt/sdb/home/wy/IMBioLG/tb_lgs/uspto_1/version_13/checkpoints/epoch=4-step=27515.ckpt
#  --d_model 256 \
#  --dim_feedforward 512 \
#  --num_encoder_layer 12 \
#  --norm_first \
#  --nhead 16 \
#  --max_single_hop 4 \
