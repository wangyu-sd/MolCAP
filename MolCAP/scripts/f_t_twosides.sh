#!/bin/bash

python entry.py\
  --mode finetune_DDI \
  --dataset data/finetune_data/twosides \
  --task_name twosides \
  --warmup_updates 1000 \
  --tot_updates 5000000000 \
  --peak_lr 2e-4 \
  --end_lr 1e-9 \
  --seed 3706 \
  --batch_size 64 \
  --epochs 500 \
  --cuda 3 \
  --dropout 0.4 \
  --log_dir tb_lgs \
  --not_fast_read \
  --patience 100 \
  --model_path /mnt/sdb/home/wy/IMBioLG/tb_lgs/uspto_1/version_14/checkpoints/epoch=9-step=55030.ckpt
#  --acc_batches 1 \
#  --d_model 256 \
#  --dim_feedforward 512 \
#  --num_encoder_layer 12 \
#  --warmup_updates 2e3 \
#  --norm_first \
#  --nhead 16 \
#  --max_single_hop 4 \
