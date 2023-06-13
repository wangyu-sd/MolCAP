#!/bin/bash

python entry.py\
  --mode finetune_classification \
  --dataset data/finetune_data/bace \
  --task_name bace \
  --seed 1246 \
  --log_dir tb_lgs \
  --not_fast_read \
  --model_path /mnt/sdb/home/wy/IMBioLG/tb_lgs/bace/version_272/checkpoints/epoch=126-step=635.ckpt\
  --test \
  --predict \
