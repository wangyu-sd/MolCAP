#!/bin/bash

python entry.py\
  --mode finetune_DDI \
  --dataset data/finetune_data/drugbank \
  --task_name drugbank \
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
  --patience 20 \
  --split_type random \
  --model_path /mnt/sdb/home/wy/IMBioLG/tb_lgs/uspto_1/version_13/checkpoints/epoch=4-step=27515.ckpt
#  --acc_batches 1 \
#  --d_model 256 \
#  --dim_feedforward 512 \
#  --num_encoder_layer 12 \
#  --norm_first \
#  --nhead 16 \
#  --max_single_hop 4 \
#  --model_path /mnt/sdb/home/wy/IMBioLG/tb_lgs/uspto_1/version_5/checkpoints/epoch=9-step=27520.ckpt
#  --model_path /mnt/sdb/home/wy/IMBioLG/tb_lgs/drugbank/version_8/checkpoints/last.ckpt
#  --model_path /mnt/sdb/home/wy/IMBioLG/tb_lgs/uspto_1/version_1/checkpoints/last.ckpt
