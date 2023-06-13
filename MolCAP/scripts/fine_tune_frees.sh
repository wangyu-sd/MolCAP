#!/bin/bash

python entry.py\
  --mode finetune_regression_not_freeze \
  --dataset data/finetune_data/freesolv \
  --task_name freesolv \
  --warmup_updates 10 \
  --tot_updates 5000000000 \
  --peak_lr 2e-4 \
  --end_lr 5e-5 \
  --seed 123 \
  --batch_size 2048 \
  --epochs 500 \
  --cuda 3 \
  --dropout 0.3 \
  --max_single_hop 4 \
  --log_dir tb_lgs \
  --patience 50 \
  --val_every_n_epoch 1 \
#  --model_path /mnt/sdb/home/wy/IMBioLG/tb_lgs/uspto_1/version_20/checkpoints/epoch=14-step=41280.ckpt
#  --prompt_mol O
#  --split_type scaffold \
  #  --model_path /mnt/sdb/home/wy/IMBioLG/tb_lgs/pretrain/version_52/checkpoints/epoch=14-step=43350.ckpt
#  --mode prompt_gat_v2_regression \
