# !/usr/bin/python3
# @File: datamodual.py
# --coding:utf-8--
# @Author:yuwang
# @Email:as1003208735@foxmail.com
# @Time: 2022.03.19.12
import pytorch_lightning as pl
import torch
from torch_geometric import loader
from torch.utils.data import DataLoader

from data.data_utils import get_downstream_task_info, RandomScaffoldSplitter, BalancedRandomScaffoldSplitter, \
    ScaffoldSplitter
from data.datasets import RXNDataSet, MolPropDataSet, TDCSingleDatset, TDCMultiDatset, MolPropDataSetWithPrompt, \
    BiosnapDataset
import os
import os.path as osp
import pandas as pd
from tqdm import tqdm
from tdc.single_pred import ADME
from tdc.multi_pred import DDI
from tdc.metadata import dataset_names
from tdc.utils import get_label_map
from torch.utils.data import random_split


class MolDataModule(pl.LightningDataModule):
    def __init__(
            self,
            root,
            batch_size,
            fast_read=False,
            split_names=None,
            num_workers=None,
            pin_memory=True,
            shuffle=True,
            predict=True,
            seed=114,
            task_name='uspto',
            split_type='random_scaffold',
            pretraining_fraction=1,
            num_shots=None,
            prompt=False,
    ):
        super().__init__()
        if num_workers is None:
            num_workers = len(os.sched_getaffinity(0))
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.dataset_dict = {}
        self.prompt = prompt
        # 处理数据集
        if split_names is None:
            split_names = ['train', 'valid', 'test']

        self.split_names = split_names
        self.data_loader_fn = loader.DataLoader if prompt else DataLoader
        # uspto
        if 'uspto' in task_name:
            max_node = 100
            for data_split in split_names:
                self.dataset_dict[data_split] = RXNDataSet(root=root,
                                                           data_split=data_split,
                                                           fast_read=fast_read,
                                                           max_node=max_node)
                if pretraining_fraction < 1:
                    raw_length = len(self.dataset_dict[data_split])
                    cur_length = int(raw_length * pretraining_fraction)
                    self.dataset_dict[data_split] = random_split(self.dataset_dict[data_split],
                                                                 lengths=[cur_length, raw_length - cur_length],
                                                                 generator=torch.Generator().manual_seed(seed))[0]
        # 下游任务
        else:
            if task_name in dataset_names['ADME']:
                os.makedirs(osp.join(root, 'raw'), exist_ok=True)
                # data = HTS(name=task_name, path=osp.join(root, 'raw'))
                data = ADME(name=task_name, path=osp.join(root, 'raw'))
                mol_dataset = TDCSingleDatset(root=root, task_name=task_name, fast_read=fast_read)
                # self.num_tasks = len(get_label_map(name=task_name, task='ADME', path=osp.join(root, 'raw')))
                self.num_tasks = 1
                self.loss_type = 'bce'

            elif task_name in dataset_names['DDI']:
                os.makedirs(osp.join(root, 'raw'), exist_ok=True)
                # data = HTS(name=task_name, path=osp.join(root, 'raw'))
                data = DDI(name=task_name, path=osp.join(root, 'raw'))
                if task_name == 'drugbank':
                    target_key = ['X1', 'X2', 'Y']
                    self.num_tasks = len(get_label_map(name=task_name, task='DDI', path=osp.join(root, 'raw')))
                    self.loss_type = 'cross_entropy'
                elif task_name == 'twosides':
                    target_key = ['X1', 'X2', 'Y']
                    self.num_tasks = len(get_label_map(name=task_name, task='DDI',
                                                       name_column='Side Effect Name', path=osp.join(root, 'raw')))
                    self.loss_type = 'bce'
                else:
                    raise ValueError(f"Task name {task_name} is not matched.")
                mol_dataset = TDCMultiDatset(root=root, task_name=task_name,
                                             fast_read=fast_read, target_key=target_key)
            elif task_name == 'biosnap':
                mol_dataset = {}
                for split in ['train', 'test']:
                    mol_dataset[split] = BiosnapDataset(root=root,
                                                        task_name=task_name,
                                                        split=split,
                                                        fast_read=fast_read,
                                                        sep=',', )

            else:
                config_dict = {'task_name': task_name, 'root': root}
                print(task_name)
                print(root)
                try:
                    config_dict = get_downstream_task_info(config_dict)
                except ValueError as e:
                    raise ValueError(f"Task name {task_name} is not matched.")

                if not prompt:
                    mol_dataset = MolPropDataSet(
                        root=config_dict['root'],
                        task_name=config_dict['task_name'],
                        target=config_dict['target'],
                        fast_read=fast_read,
                    )
                else:
                    mol_dataset = MolPropDataSetWithPrompt(
                        root=config_dict['root'],
                        task_name=config_dict['task_name'],
                        target=config_dict['target'],
                        fast_read=fast_read,
                        prompt=True,
                    )

                self.num_tasks = len(config_dict['target'])
                self.loss_type = config_dict['loss_type']

            if split_type == 'random_scaffold':
                splitter = RandomScaffoldSplitter()
            elif split_type == 'balanced_random_scaffold':
                splitter = BalancedRandomScaffoldSplitter()
            elif split_type == 'scaffold':
                splitter = ScaffoldSplitter()
            elif split_type == 'random':
                splitter = None
            else:
                raise ValueError(
                    "split_type must be one of ['random_scaffold', 'balanced_random_scaffold, scaffold, "
                    "random']")

            if splitter is not None:
                datasets = splitter.split(mol_dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=seed)
                for i, data_split in enumerate(split_names):
                    self.dataset_dict[data_split] = datasets[i]
            elif isinstance(mol_dataset, dict):
                valid_size = int(0.125 * len(mol_dataset['train']))
                train_size = len(mol_dataset['train']) - valid_size
                train_dataset, valid_dataset = torch.utils.data.random_split(mol_dataset['train'],
                                                                             lengths=[train_size, valid_size],
                                                                             generator=torch.Generator().manual_seed(
                                                                                 seed))
                self.dataset_dict['train'] = train_dataset
                self.dataset_dict['valid'] = valid_dataset
                self.dataset_dict['test'] = mol_dataset['test']

            else:
                train_size = int(0.8 * len(mol_dataset))
                valid_size = int(0.1 * len(mol_dataset))
                test_size = len(mol_dataset) - train_size - valid_size
                train_dataset, valid_dataset, test_dataset = \
                    torch.utils.data.random_split(mol_dataset,
                                                  lengths=[train_size, valid_size, test_size],
                                                  generator=torch.Generator().manual_seed(seed))
                self.dataset_dict['train'] = train_dataset
                self.dataset_dict['valid'] = valid_dataset
                self.dataset_dict['test'] = test_dataset

            if num_shots is not None:
                others = len(self.dataset_dict['train']) - num_shots
                self.dataset_dict['train'] = random_split(self.dataset_dict['train'],
                                                          lengths=[num_shots, others],
                                                          generator=torch.Generator().manual_seed(seed))[0]

    def train_dataloader(self):
        return self.data_loader_fn(
            self.dataset_dict[self.split_names[0]],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return self.data_loader_fn(
            self.dataset_dict[self.split_names[1]],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return self.data_loader_fn(
            self.dataset_dict[self.split_names[2]],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self):
        return self.test_dataloader()

# class RxnCollector:
#     def __init__(self, max_node):
#         self.max_node = max_node
#
#     def __call__(self, data_list):
#         batch_size = len(data_list)
#         max_atoms = max([data[3] for data in data_list])
#
#         y_list = []
#         center_cnt = []
#         p_dict = {
#             'atom_fea': torch.zeros(batch_size, 11, max_atoms, dtype=torch.half),
#             'bond_adj': torch.zeros(batch_size, max_atoms, max_atoms, dtype=torch.uint8),
#             'dist_adj': torch.zeros(batch_size, max_atoms, max_atoms, dtype=torch.float),
#             'n_atom': []
#         }
#         r_dict = {
#             'atom_fea': torch.zeros(batch_size, 11, max_atoms, dtype=torch.half),
#             'bond_adj': torch.zeros(batch_size, max_atoms, max_atoms, dtype=torch.uint8),
#             'dist_adj': torch.zeros(batch_size, max_atoms, max_atoms, dtype=torch.float),
#             'n_atom': []
#         }
#
#         for idx, data in enumerate(data_list):
#             y_list.append(data[0])
#             n_pro, n_rct = data[1]['n_atom'], data[2]['n_atom']
#             p_dict['n_atom'].append(n_pro)
#             r_dict['n_atom'].append(n_rct)
#             center_cnt.append(data[4])
#
#             p_dict['atom_fea'][idx, :, :n_pro] = data[1]['atom_fea']
#             p_dict['bond_adj'][idx, :n_pro, :n_pro] = data[1]['bond_adj'] + 1
#             p_dict['dist_adj'][idx, :n_pro, :n_pro] = data[1]['dist_adj']
#
#             # print(data[2]['atom_fea'].size(), n_rct, data[5], data[2]['atom_fea'][0])
#
#             r_dict['atom_fea'][idx, :, :n_rct] = data[2]['atom_fea']
#             r_dict['bond_adj'][idx, :n_rct, :n_rct] = data[2]['bond_adj'] + 1
#             r_dict['dist_adj'][idx, :n_rct, :n_rct] = data[2]['dist_adj']
#
#             p_dict['atom_fea'][idx, 7, :n_pro] += 8
#             r_dict['atom_fea'][idx, 7, :n_rct] += 8
#
#         p_dict['bond_adj'][idx, :n_pro, :n_pro] = data[1]['bond_adj'] + 1
#         r_dict['bond_adj'][idx, :n_rct, :n_rct] = data[2]['bond_adj'] + 1
#
#         p_dict['n_atom'] = torch.tensor(p_dict['n_atom'])
#         r_dict['n_atom'] = torch.tensor(r_ict['n_atom'])
#         y_list = torch.tensor(y_list)
#         center_cnt = torch.tensor(center_cnt)
#
#         return p_dict, r_dict, center_cnt, y_list
