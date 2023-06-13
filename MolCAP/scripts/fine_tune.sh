#!/bin/bash

python entry.py\
  --mode finetune_classification \
  --dataset data/finetune_data/bace \
  --task_name bace \
  --warmup_updates 10 \
  --tot_updates 5000000000 \
  --peak_lr 0.00023347012818074672 \
  --end_lr 6.909532661604662e-7 \
  --seed 1246 \
  --weight_decay 0.6684378608424649 \
  --batch_size 128 \
  --epochs 500 \
  --cuda 3 \
  --dropout 0.3 \
  --log_dir tb_lgs \
  --not_fast_read \
  --patience 50 \
  --val_every_n_epoch 1 \
  --n_downstream_layer 8 \
  --model_path /mnt/sdb/home/wy/IMBioLG/tb_lgs/uspto_1/version_22/checkpoints/last.ckpt \
  --prompt_mol O \
#  --d_model 256 \
#  --dim_feedforward 512 \
#  --num_encoder_layer 12 \
#  --warmup_updates 2e3 \
#  --norm_first \
#  --nhead 16 \
#  --max_single_hop 4 \

#  --test
#  --split_type scaffold \
#  --model_path /mnt/sdb/home/wy/IMBioLG/tb_lgs/bace/version_57/checkpoints/last.ckpt \

#  --model_path tb_lgs/pretrain/version_57/checkpoints/epoch=69-step=202300.ckpt
  #  --model_path /mnt/sdb/home/wy/IMBioLG/tb_lgs/pretrain/version_52/checkpoints/epoch=14-step=43350.ckpt
