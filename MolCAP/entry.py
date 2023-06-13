# !/usr/bin/python3
# @File: train.py
# --coding:utf-8--
# @Author:yuwang
# @Email:as1003208735@foxmail.com
# @Time: 2022.03.27.19
import os
import argparse
import pytorch_lightning as pl
import torch
import numpy as np
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.utilities import seed
from model.IMLearner import IMLearner
from data.datamodule import MolDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Timer, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def main():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = IMLearner.add_model_specific_args(parser)

    # trainer configuration
    parser.add_argument('--mode', type=str, default='pretrain')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--acc_batches', type=int, default=8)
    parser.add_argument('--log_dir', type=str, default='tb_logs')
    parser.add_argument('--name', type=str, default='pretrain')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--predict', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--val_every_n_epoch', type=int, default=5)

    # dataset configuration
    parser.add_argument('--dataset', type=str, default="data/uspto")
    parser.add_argument('--num_tasks', type=int, default=1)
    parser.add_argument('--not_fast_read', default=False, action='store_true')
    parser.add_argument('--use_3d_info', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--task_name', type=str, default='uspto')
    parser.add_argument('--pretraining_fraction', type=float, default=1)
    parser.add_argument('--num_shots', type=int, default=None)
    parser.add_argument('--split_type', type=str, default='random_scaffold')

    args = parser.parse_args()
    print(args)

    seed.seed_everything(args.seed)

    print("Building DataModule...")
    dm = build_datamodule(args)
    print("Finished DataModule.")

    if args.mode != 'pretrain':
        args.num_tasks = dm.num_tasks
        args.loss_type = dm.loss_type
        args.name = args.task_name
    else:
        args.loss_type = None


    print("Building Model...")
    model = build_model(args)
    print("Finished Model...")

    print("Building Trainer...")
    trainer = build_trainer(args)
    print("Finished Trainer...")

    if not args.test and not args.predict:
        trainer.fit(model, dm)  # Runs the full optimization routine.
        print('Finished training..')
        print(args)

        print('Testing...')
        trainer.test(model, dm, ckpt_path='best')
        # 对测试集执行一个评估周期。
    elif args.test:
        print('Testing...')
        performance = trainer.test(model, dm, ckpt_path=args.model_path)
        print(performance[0])

    elif args.test and args.predict:
        res = trainer.predict(model, dm, ckpt_path=args.model_path)
        ori_smi = dm.dataset_dict['test'].smiles
        pred = []




def build_trainer(args):
    logger = TensorBoardLogger(args.log_dir, name=args.name)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_cb = ModelCheckpoint(monitor="best_point", save_last=True, mode='max')
    early_stop = EarlyStopping(monitor="best_point", patience=args.patience, mode='max')

    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        ))

    trainer = Trainer(
        accelerator='gpu',
        # strategy='ddp',
        logger=logger,
        devices=list(map(int, args.cuda.split(','))),
        max_epochs=args.epochs,
        # accumulate_grad_batches=args.acc_batches,
        callbacks=[lr_monitor, checkpoint_cb, progress_bar, early_stop],
        check_val_every_n_epoch=args.val_every_n_epoch,
        log_every_n_steps=5,
        detect_anomaly=True,
        # precision=16 if not args.predict else 32,
        # auto_lr_find='peak_lr',
        gradient_clip_algorithm='norm'
    )
    return trainer


def build_datamodule(args):
    dm = MolDataModule(
        root=args.dataset,
        batch_size=args.batch_size,
        fast_read=not args.not_fast_read,
        num_workers=args.num_workers,
        predict=args.predict or args.test,
        task_name=args.task_name,
        seed=args.seed,
        split_type=args.split_type,
        pretraining_fraction=args.pretraining_fraction,
        num_shots=args.num_shots,
        prompt='prompt' in args.mode,
    )
    return dm


def build_model(args):
    if args.model_path == '':  # 没经过预训练
        model = IMLearner(
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layer=args.num_encoder_layer,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            norm_first=args.norm_first,
            activation=args.activation,
            weight_decay=args.weight_decay,
            use_3d_info=args.use_3d_info,
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            peak_lr=args.peak_lr,
            end_lr=args.end_lr,
            max_single_hop=args.max_single_hop,
            use_dist_adj=not args.not_use_dist_adj,
            mode=args.mode,
            num_tasks=args.num_tasks,
            loss_type=args.loss_type,
        )
    elif args.mode == 'pretrain':  # 预训练的模型, 预训练断点恢复
        model = IMLearner.load_from_checkpoint(
            args.model_path,
            strict=True,
        )
    else:
        model = IMLearner.load_from_checkpoint(
            args.model_path,
            strict=False,
            mode=args.mode,
            loss_type=args.loss_type,
            num_tasks=args.num_tasks,
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            peak_lr=args.peak_lr,
            end_lr=args.end_lr,
            dropout=args.dropout,
            prompt_mol=args.prompt_mol,

            # d_molde=256,
            # dim_feedforward=512,
        )

    return model


if __name__ == '__main__':
    main()
