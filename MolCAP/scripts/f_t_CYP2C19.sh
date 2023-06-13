#!/bin/bash

python entry.py\
  --mode finetune_classification \
  --dataset data/finetune_data/CYP2C19_Veith \
  --task_name cyp2c19_veith \
  --warmup_updates 1000 \
  --tot_updates 5000000000 \
  --peak_lr 2e-4 \
  --end_lr 1e-9 \
  --seed 66 \
  --batch_size 128 \
  --epochs 500 \
  --cuda 2 \
  --dropout 0.3 \
  --max_single_hop 4 \
  --log_dir tb_lgs \
  --not_fast_read \
  --patience 20 \
  --acc_batches 1 \
  --d_model 256 \
  --dim_feedforward 512 \
  --num_encoder_layer 12 \
  --warmup_updates 2e3 \
  --norm_first \
  --nhead 16 \
  --max_single_hop 4 \
  --model_path /mnt/sdb/home/wy/IMBioLG/tb_lgs/uspto_1/version_14/checkpoints/epoch=9-step=55030.ckpt \
#  --test
#  --split_type balanced_random_scaffold \
  #  --model_path /mnt/sdb/home/wy/IMBioLG/tb_lgs/pretrain/version_52/checkpoints/epoch=14-step=43350.ckpt
