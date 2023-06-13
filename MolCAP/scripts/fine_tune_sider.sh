#!/bin/bash

python entry.py\
  --mode finetune_classification \
  --dataset data/finetune_data/SIDER \
  --task_name sider\
  --warmup_updates 10 \
  --tot_updates 5000000000 \
  --peak_lr 2e-4 \
  --end_lr 5e-5 \
  --seed 45 \
  --batch_size 32 \
  --epochs 500 \
  --cuda 3 \
  --dropout 0.3 \
  --max_single_hop 4 \
  --log_dir tb_lgs \
  --not_fast_read \
  --patience 50 \
  --acc_batches 1 \
  --d_model 256 \
  --norm_first \
  --nhead 16 \
  --val_every_n_epoch 1 \
  --max_single_hop 4 \
#  --model_path /mnt/sdb/home/wy/IMBioLG/tb_lgs/uspto_1/version_13/checkpoints/epoch=4-step=27515.ckpt
  #  --model_path /mnt/sdb/home/wy/IMBioLG/tb_lgs/pretrain/version_52/checkpoints/epoch=14-step=43350.ckpt
