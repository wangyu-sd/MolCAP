#!/bin/bash

python entry.py\
  --mode finetune_classification \
  --dataset data/finetune_data/TOX21 \
  --task_name tox21 \
  --warmup_updates 1000 \
  --tot_updates 5000000000 \
  --peak_lr 2e-4 \
  --end_lr 1e-9 \
  --seed 66 \
  --batch_size 256 \
  --epochs 500 \
  --cuda 2 \
  --d_model 256 \
  --dropout 0.4 \
  --max_single_hop 4 \
  --log_dir tb_lgs \
  --not_fast_read \
  --patience 50 \
  --model_path /mnt/sdb/home/wy/IMBioLG/tb_lgs/uspto_1/version_20/checkpoints/epoch=14-step=41280.ckpt
#  --model_path /mnt/sdb/home/wy/IMBioLG/tb_lgs/uspto_1/version_14/checkpoints/epoch=9-step=55030.ckpt
#  --model_path /mnt/sdb/home/wy/IMBioLG/tb_lgs/uspto_1/version_13/checkpoints/epoch=4-step=27515.ckpt
#  --split_type scaffold \
  #  --model_path /mnt/sdb/home/wy/IMBioLG/tb_lgs/pretrain/version_52/checkpoints/epoch=14-step=43350.ckpt
