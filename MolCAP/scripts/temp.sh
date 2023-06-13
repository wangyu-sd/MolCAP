#!/bin/bash

python entry.py\
  --batch_size 256 \
  --acc_batches 1 \
  --d_model 128 \
  --dim_feedforward 256 \
  --epochs 2000 \
  --dropout 0.2 \
  --warmup_updates 2e3 \
  --tot_updates 1e10 \
  --end_lr 1e-10 \
  --dataset data/uspto \
  --known_rxn_cnt \
  --norm_first \
  --nhead 16 \
  --p_layer 3 \
  --r_layer 3 \
  --fusion_layer 3 \
  --seed 123 \
  --cuda 0 \
  --max_single_hop 4 \
  --log_dir tb_lgs \
  --not_fast_read
