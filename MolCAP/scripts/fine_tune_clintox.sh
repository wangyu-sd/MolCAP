#!/bin/bash

python entry.py\
  --mode prompt_gat_v1_classification \
  --dataset data/finetune_data/clintox \
  --task_name clintox \
  --tot_updates 5000000000 \
  --peak_lr 2e-4 \
  --end_lr 1e-9 \
  --seed 123 \
  --batch_size 128 \
  --epochs 500 \
  --cuda 2 \
  --dropout 0.3 \
  --max_single_hop 4 \
  --log_dir tb_lgs \
  --not_fast_read \
  --patience 100 \
  --acc_batches 1 \
  --d_model 256 \
  --dim_feedforward 512 \
  --num_encoder_layer 12 \
  --warmup_updates 2e3 \
  --norm_first \
  --nhead 16 \
  --model_path /mnt/sdb/home/wy/IMBioLG/tb_lgs/uspto_1/version_13/checkpoints/epoch=4-step=27515.ckpt \
#  --split_type scaffold \
#  --model_path tb_lgs/pretrain/version_57/checkpoints/epoch=69-step=202300.ckpt
  #  --model_path /mnt/sdb/home/wy/IMBioLG/tb_lgs/pretrain/version_52/checkpoints/epoch=14-step=43350.ckpt
