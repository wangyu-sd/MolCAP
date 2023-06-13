#!/bin/bash

python entry.py\
  --batch_size 256 \
  --acc_batches 1 \
  --d_model 256 \
  --dim_feedforward 512 \
  --epochs 2000 \
  --dropout 0.4 \
  --warmup_updates 2e3 \
  --tot_updates 1e10 \
  --peak_lr 1e-6 \
  --end_lr 1e-10 \
  --dataset data/uspto_100k \
  --known_rxn_cnt \
  --norm_first \
  --nhead 16 \
  --p_layer 5 \
  --r_layer 5 \
  --fusion_layer 1 \
  --seed 123 \
  --cuda 1 \
  --max_single_hop 4 \
  --log_dir tb_lgs \
  --not_fast_read \
  --pretraining_fraction 0.3 \
  --name uspto_03 \
