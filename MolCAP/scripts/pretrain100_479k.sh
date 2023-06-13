#!/bin/bash

python entry.py\
  --batch_size 256 \
  --acc_batches 1 \
  --d_model 256 \
  --dim_feedforward 512 \
  --num_encoder_layer 12 \
  --epochs 20000 \
  --dropout 0.3 \
  --warmup_updates 2e3 \
  --tot_updates 1e10 \
  --peak_lr 2e-4 \
  --end_lr 1e-9 \
  --dataset data/uspto_479k \
  --norm_first \
  --nhead 16 \
  --num_encoder_layer 12 \
  --seed 3706 \
  --cuda 2 \
  --max_single_hop 4 \
  --log_dir tb_lgs \
  --not_fast_read \
  --pretraining_fraction 1 \
  --name uspto_479k_1 \
