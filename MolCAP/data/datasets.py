# !/usr/bin/python3
# @File: dataset.py
# --coding:utf-8--
# @Author:yuwang
# @Email:as1003208735@foxmail.com
# @Time: 2022.03.18.21
import os
import os.path as osp

import numpy
import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from torch_geometric.data import Dataset
from torch.utils.data import Dataset as BaseDataset
from tqdm import tqdm
from rdkit import Chem
from data.data_utils import smile_to_mol_info, get_tgt_adj_order, \
    get_bond_order_adj, padding_mol_info, load_finetune_dataset, NUM_ATOM_FEAT

RDLogger.DisableLog('rdApp.*')

FILE_NAME = '.pt'


# class ProtDrugSet(BaseDataset):
#     def __init__(self, prot_list, drug_list):
#         super(ProtMolSet, self).__init__()
#         self.mol_dict_list = []
#         for prot, drug in zip(prot_list, drug_list):
#             mol_dict_list.append(merge_prot_drug(prot, drug))
#
#     @staticmethod
#     def merge_prot_drug(prot, drug):
#         prot_mol = smile_to_mol_info(Chem.MolToSmiles(Chem.MolFromSequence(prot)))
#         drug_mol = smile_to_mol_info(drug)
#         return concat_mol_dict(drug_mol, prot_mol), drug['n_atom']
#
#     def __getitem__(self, index):
#         return self.prot_list[index], self.mol_list[index]
#
#     def __len__(self):
#         return len(self.prot_list)

class TDCBaseDataSet(Dataset):
    @property
    def raw_file_names(self):
        return [file_dir for file_dir in os.listdir(osp.join(self.root, "raw"))
                if "train" not in file_dir or 'test' not in file_dir or 'valid' not in file_dir]

    @property
    def processed_file_names(self):
        return [f"mol_prop_data_{idx}" + FILE_NAME for idx in range(self.size)]

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "processed", 'mol_prop_data')

    def __init__(self, root, task_name, fast_read=True, save_cache=True):
        self.root = root
        self.task_name = task_name
        self.fast_read = fast_read
        self.save_cache = save_cache
        self.max_node = -1
        self.size_path = osp.join(osp.join(self.root, "processed"), "num_files" + FILE_NAME)
        self.size_path = osp.join(osp.join(self.root, "processed"), "num_files" + FILE_NAME)
        if osp.exists(self.size_path):
            self.size = int(torch.load(self.size_path))
            self.smiles_list = torch.load(osp.join(self.root, "processed", "smiles_list" + FILE_NAME))
        else:
            self.size = 0
            self.smiles_list = []
        super(TDCBaseDataSet, self).__init__(root)

        if fast_read:
            data_cache_path = osp.join(self.root, osp.join('processed', f'cache_{task_name}' + FILE_NAME))
            if osp.isfile(data_cache_path) and save_cache:
                print(f"read cache from {data_cache_path}...")
                # self.data = torch.load(data_cache_path)
                self.data = torch.load(data_cache_path)
            else:
                self.fast_read = False
                self.data = [rxn_data for rxn_data in tqdm(self)]
                if save_cache:
                    # torch.save(self.data, data_cache_path)
                    torch.save(self.data, data_cache_path)

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)

        for raw_file_name in self.raw_file_names:
            print(f"Processing the {raw_file_name} dataset to torch geometric format...\n")
            mol_csv = pd.read_csv(osp.join(self.raw_dir, raw_file_name),
                                  sep='\t' if 'csv' not in raw_file_name else ',')
            cur_idx = 0
            smiles_list = []
            for idx, (smiles, label, drug_id) in enumerate(tqdm(zip(mol_csv.Drug, mol_csv.Y, mol_csv.Drug_ID))):
                # if smiles is None:
                #     continue
                mol_info = smile_to_mol_info(smiles)
                # ---------
                if mol_info is None:
                    continue
                # ---------
                if mol_info['n_atom'] > 100:
                    continue
                self.max_node = max(self.max_node, mol_info['n_atom'])
                self.max_node = min(100, self.max_node)
                mol_prop_data = (mol_info, label, drug_id)
                # print(len(mol_prop_data))
                torch.save(mol_prop_data, osp.join(self.processed_dir, f"mol_prop_data_{cur_idx}" + FILE_NAME))
                cur_idx += 1
                smiles_list.append(smiles)

            print(f"max node num: {self.max_node}, padding mol_features...")
            for idx in tqdm(range(cur_idx)):
                # -------------
                # if osp.exists(osp.join(self.processed_dir, f"mol_prop_data_{idx}" + FILE_NAME)):
                mol_prop_data = torch.load(osp.join(self.processed_dir, f"mol_prop_data_{idx}" + FILE_NAME))
                padding_mol_info(mol_prop_data[0], self.max_node)
                # print(len(mol_prop_data))
                torch.save(mol_prop_data, osp.join(self.processed_dir, f"mol_prop_data_{idx}" + FILE_NAME))
                # --------------
                self.smiles_list.append(smiles_list[idx])

            self.size = cur_idx
            torch.save(self.size, self.size_path)
            torch.save(self.smiles_list, osp.join(self.root, 'processed', 'smiles_list' + FILE_NAME))

    def len(self):
        return self.size

    def get(self, idx):
        if self.fast_read:
            mol_prop_data = self.data[idx]
        else:
            mol_prop_data = torch.load(osp.join(self.processed_dir, f"mol_prop_data_{idx}" + FILE_NAME))
        return mol_prop_data[:2]


class TDCSingleDatset(TDCBaseDataSet):
    def __init__(self, root, task_name, fast_read=True, save_cache=True):
        super(TDCSingleDatset, self).__init__(root, task_name, fast_read, save_cache)


class TDCMultiDatset(TDCBaseDataSet):
    def __init__(self, root, task_name, fast_read=True, save_cache=True, num_tasks=None, target_key=None, sep='\t'):
        self.num_tasks = num_tasks
        self.target_key = target_key
        self.sep = sep
        super(TDCMultiDatset, self).__init__(root, task_name, fast_read, save_cache)

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        for raw_file_name in self.raw_file_names:
            print(f"Processing the {raw_file_name} dataset to torch geometric format...\n")
            mol_csv = pd.read_csv(osp.join(self.raw_dir, raw_file_name), sep=self.sep)
            cur_idx = 0

            smiles_list = []
            for idx, (smi1, smi2, label) in enumerate(tqdm(zip(*[mol_csv[key]
                                                               for key in self.target_key]), total=len(mol_csv))):
                # if smiles is None:
                #     continue
                mol_info_1 = smile_to_mol_info(smi1)
                mol_info_2 = smile_to_mol_info(smi2)
                # ---------
                if mol_info_1 is None or mol_info_2 is None:
                    continue
                # ---------
                self.max_node = max(self.max_node, mol_info_1['n_atom'], mol_info_2['n_atom'])
                self.max_node = min(100, self.max_node)
                if mol_info_1['n_atom'] > 100 or mol_info_2['n_atom'] > 100:
                    continue
                if self.task_name == 'drugbank':
                    label -= 1
                mol_prop_data = ((mol_info_1, mol_info_2), label, None)
                # print(len(mol_prop_data))
                torch.save(mol_prop_data, osp.join(self.processed_dir, f"mol_prop_data_{cur_idx}" + FILE_NAME))
                cur_idx += 1
                smiles_list.append(smi1 + '.' + smi2)

            print(f"max node num: {self.max_node}, padding mol_features...")
            for idx in tqdm(range(cur_idx)):
                # -------------
                # if osp.exists(osp.join(self.processed_dir, f"mol_prop_data_{idx}" + FILE_NAME)):
                mol_prop_data = torch.load(osp.join(self.processed_dir, f"mol_prop_data_{idx}" + FILE_NAME))
                padding_mol_info(mol_prop_data[0], self.max_node)
                # print(len(mol_prop_data))
                torch.save(mol_prop_data, osp.join(self.processed_dir, f"mol_prop_data_{idx}" + FILE_NAME))
                # --------------
                self.smiles_list.append(smiles_list[idx])

            self.size = cur_idx
            torch.save(self.size, self.size_path)
            torch.save(self.smiles_list, osp.join(self.root, 'processed', 'smiles_list' + FILE_NAME))

class BiosnapDataset(TDCMultiDatset):
    @property
    def raw_file_names(self):
        return [osp.join(self.split + ".csv")]

    def __init__(self, root, task_name, fast_read=True, save_cache=True, num_tasks=None, split='train', sep=','):
        self.split = split
        self.num_tasks = num_tasks
        super().__init__(root, task_name, fast_read, save_cache, sep=sep,
                         target_key=['Drug1_SMILES', 'Drug2_SMILES', 'label'])



class MolPropDataSet(Dataset):
    @property
    def raw_file_names(self):
        return [file_dir for file_dir in os.listdir(osp.join(self.root, "raw")) if ".csv" in file_dir]

    @property
    def processed_file_names(self):
        return [f"mol_prop_data_{idx}" + FILE_NAME for idx in range(self.size)]

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "processed", 'mol_prop_data')

    def __init__(self, root, task_name, target, fast_read=True, save_cache=True, prompt=False, base_path=None):
        self.root = root
        self.task_name = task_name
        self.target = target
        self.fast_read = fast_read
        self.save_cache = save_cache
        self.prompt = prompt
        self.max_node = -1
        if base_path is None:
            base_path = osp.join(self.root, "processed")
        self.base_path = base_path
        self.size_path = osp.join(base_path, "num_files" + FILE_NAME)
        if osp.exists(self.size_path):
            self.size = int(torch.load(self.size_path))
            self.smiles_list = torch.load(osp.join(base_path, "smiles_list" + FILE_NAME))
        else:
            self.size = 0
        super(MolPropDataSet, self).__init__(root)

        if fast_read:
            data_cache_path = osp.join(base_path, f'cache_{task_name}' + FILE_NAME)
            if osp.isfile(data_cache_path) and save_cache:
                print(f"read cache from {data_cache_path}...")
                # self.data = torch.load(data_cache_path)
                self.data = torch.load(data_cache_path)
            else:
                self.fast_read = False
                self.data = [rxn_data for rxn_data in tqdm(self)]
                if save_cache:
                    # torch.save(self.data, data_cache_path)
                    torch.save(self.data, data_cache_path)

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)

        for raw_file_name in self.raw_file_names:
            print(f"Processing the {raw_file_name} dataset to torch geometric format...\n")
            smiles_list, mol_info_list, labeles = load_finetune_dataset(osp.join(self.root, 'raw', raw_file_name)
                                                                        , target=self.target, prompt=self.prompt)

            for mol_info in mol_info_list:
                if mol_info is not None:
                    self.max_node = max(self.max_node, mol_info['n_atom'])
            print('max node:', self.max_node)
            # self.max_node = min(100, self.max_node)
            # print('max node:', self.max_node)

            cur_idx = 0
            self.smiles_list = []
            for idx, mol_info in enumerate(tqdm(mol_info_list)):
                if mol_info is None or mol_info['n_atom'] == 0 or mol_info['n_atom'] > self.max_node:
                    continue
                self.smiles_list.append(smiles_list[idx])
                padding_mol_info(mol_info, self.max_node)
                mol_prop_data = (mol_info, labeles[idx])
                torch.save(mol_prop_data, osp.join(self.processed_dir, f"mol_prop_data_{cur_idx}" + FILE_NAME))
                cur_idx += 1
            self.size = cur_idx
        torch.save(self.size, self.size_path)
        torch.save(self.smiles_list, osp.join(self.base_path, 'smiles_list' + FILE_NAME))

    def len(self):
        return self.size

    def get(self, idx):
        if self.fast_read:
            mol_prop_data = self.data[idx]
        else:
            # rxn_data = torch.load(osp.join(self.processed_dir, f"rxn_data_{idx}.pt"))[:-1]
            mol_prop_data = torch.load(osp.join(self.processed_dir, f"mol_prop_data_{idx}" + FILE_NAME))
        return mol_prop_data


class MolPropDataSetWithPrompt(MolPropDataSet):
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "processed_with_prompt", 'mol_prop_data')

    def __init__(self, root, task_name, target, fast_read=True, save_cache=True, prompt=True):
        super(MolPropDataSetWithPrompt, self).__init__(root, task_name, target, fast_read, save_cache, prompt,
                                                       base_path=osp.join(root, "processed_with_prompt"))


FILE_NAME_2 = ".pt"


class RXNDataSet(Dataset):
    @property
    def raw_file_names(self):
        return [self.data_split + ".csv"]

    @property
    def processed_file_names(self):
        return [f"rxn_data_{idx}" + FILE_NAME_2 for idx in range(self.size)]

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, osp.join("processed", self.data_split))

    def __init__(self, root, data_split, fast_read=True, max_node=100, min_node=5, save_cache=True, use_3d_info=False):
        self.root = root
        # self.rxn_center_path = osp.join(root, osp.join('processed', 'rxn_center.pt'))
        self.min_node = min_node
        self.max_node = max_node
        self.data_split = data_split
        self.use_3d_info = use_3d_info
        self.size_path = osp.join(osp.join(self.root, osp.join("processed", self.data_split)), "num_files" + FILE_NAME_2)
        if osp.exists(self.size_path):
            # self.size = torch.load(self.size_path)
            self.size = torch.load(self.size_path)
        else:
            self.size = 0
        super().__init__(root)
        if fast_read:
            data_cache_path = osp.join(self.root, osp.join('processed', f'cache_{data_split}' + FILE_NAME_2))
            if osp.isfile(data_cache_path) and save_cache:
                print(f"read cache from {data_cache_path}...")
                # self.data = torch.load(data_cache_path)
                self.data = torch.load(data_cache_path)
            else:
                self.fast_read = False
                self.data = [rxn_data for rxn_data in tqdm(self)]
                if save_cache:
                    # torch.save(self.data, data_cache_path)
                    torch.save(self.data, data_cache_path)

        self.fast_read = fast_read

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)

        for raw_file_name in self.raw_file_names:
            print(f"Processing the {raw_file_name} dataset to torch geometric format...\n")

            csv = pd.read_csv(osp.join(self.raw_dir, raw_file_name))
            reaction_list = csv['rxn_smiles']
            reactant_list = list(
                map(lambda x: x.split('>>')[0], reaction_list))
            product_list = list(
                map(lambda x: x.split('>>')[1], reaction_list))
            total, cur_id = len(csv), 0
            for idx, (product, reactant) in tqdm(enumerate(zip(product_list, reactant_list)), total=total):
                try:
                    product = smile_to_mol_info(product, use_3d_info=self.use_3d_info)
                    reactant = smile_to_mol_info(reactant, use_3d_info=self.use_3d_info)
                except AttributeError:
                    continue

                order = get_tgt_adj_order(product['mol'], reactant['mol'])  # 传入rdkit原子对象
                if reactant['n_atom'] != order.size(0):
                    continue
                all_atom = max(reactant['n_atom'], product['n_atom'])
                if all_atom > self.max_node or all_atom < self.min_node:
                    continue

                reactant['atom_fea'] = reactant['atom_fea'][:, order]
                reactant['bond_adj'] = reactant['bond_adj'][order, :][:, order]
                reactant['dist_adj'] = reactant['dist_adj'][order, :][:, order]

                n_pro, n_rea = product['n_atom'], reactant['n_atom']

                pro_bond_adj = get_bond_order_adj(product['mol'])
                rea_bond_adj = get_bond_order_adj(reactant['mol'])[order][:, order]
                target_bond = -(rea_bond_adj.clone())
                target_bond[:n_pro, :n_pro] += pro_bond_adj.float()

                center = target_bond.nonzero()
                center_cnt = center.size(0) // 2
                if center_cnt >= 15:
                    # indicate a wrong reaction mapping
                    continue
                new_target = torch.zeros(self.max_node, self.max_node, dtype=target_bond.dtype)
                new_target[:n_rea, :n_rea] = target_bond
                target_bond = new_target

                target_atom = -(reactant['atom_fea'][2:, :n_pro].clone().float())
                target_atom[:, :n_pro] += product['atom_fea'][2:]

                target_atom_new = torch.zeros(NUM_ATOM_FEAT-2, self.max_node, dtype=torch.float) - 1000
                target_atom_new[:, :n_pro] = target_atom

                padding_mol_info(reactant, self.max_node)
                rxn_data = (
                    reactant,
                    {
                        'atom': target_atom_new,
                        'bond': target_bond,
                    },
                )
                # torch.save(rxn_data, osp.join(self.processed_dir, f"rxn_data_{cur_id}.pt"))
                torch.save(rxn_data, osp.join(self.processed_dir, f"rxn_data_{cur_id}" + FILE_NAME_2))
                cur_id += 1

            print(f"Completed the {raw_file_name} dataset to torch geometric format...")
            print(f"|total={cur_id}|\t|passed={idx - cur_id}|", '*' * 90)
            self.size = cur_id
            # torch.save(self.size, self.size_path)
            torch.save(self.size, self.size_path)

    def len(self):
        return self.size

    def get(self, idx):
        if self.fast_read:
            rxn_data = self.data[idx]
        else:
            # rxn_data = torch.load(osp.join(self.processed_dir, f"rxn_data_{idx}.pt"))[:-1]
            rxn_data = torch.load(osp.join(self.processed_dir, f"rxn_data_{idx}" + FILE_NAME_2))
            # print(rxn_data[:-1])
        return rxn_data
