#!/bin/bash

python entry.py\
  --mode finetune_classification \
  --dataset data/finetune_data/HIA_Hou \
  --task_name HIA_Hou \
  --warmup_updates 1000 \
  --tot_updates 5000000000 \
  --peak_lr 2e-4 \
  --end_lr 1e-9 \
  --seed 45 \
  --batch_size 128 \
  --epochs 500 \
  --cuda 0 \
  --dropout 0.4 \
  --max_single_hop 4 \
  --log_dir tb_lgs \
  --not_fast_read \
  --patience 20 \
  --model_path tb_lgs/pretrain/version_57/checkpoints/epoch=69-step=202300.ckpt
  #  --model_path /mnt/sdb/home/wy/IMBioLG/tb_lgs/pretrain/version_52/checkpoints/epoch=14-step=43350.ckpt
