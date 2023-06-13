# uncompyle6 version 3.8.0
# Python bytecode 3.7.0 (3394)
# Decompiled from: Python 3.8.8 (default, Apr 13 2021, 19:58:26) 
# [GCC 7.3.0]
# Embedded file name: /mnt/sdb/home/wy/IMBioLG/data/data_utils.py
# Compiled at: 2022-11-09 21:34:02
# Size of source mod 2**32: 10025 bytes
import os

import numpy as np, torch
import pandas as pd
from torch import Tensor
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from functools import cmp_to_key
from rdkit.Chem import rdDistGeom as molDG
from tqdm import tqdm
from torch_geometric.data import Data

# import rdkit.Chem as molDG
BOND_ORDER_MAP = {0: 0,
                  1: 1, 1.5: 2, 2: 3, 3: 4}

NUM_ATOM_FEAT = 5


def concat_prompt_mol_dict(mol_dic1, prompt_mol):
    mol_dic2 = smile_to_mol_info(prompt_mol)
    return concat_padded_mol_dict(mol_dic1, mol_dic2)


def concat_mol_dict(mol_dic1, mol_dic2):
    mol_dic = {'n_atom': mol_dic1['n_atom'] + mol_dic2['n_atom'],
               'atom_fea': torch.cat([mol_dic1['atom_fea'], mol_dic2['atom_fea']], dim=1)}
    for key in ('bond_adj', 'dist_adj', 'dist_adj_3d'):
        if mol_dic1[key] is not None:
            mol_dic[key] = cat_adj(mol_dic1[key], mol_dic2[key])

    return mol_dic


def concat_padded_mol_dict(mol_dic1, mol_dic2):
    size_1, size_2 = mol_dic1['atom_fea'].size(-1), mol_dic2['atom_fea'].size(-1)
    bsz, n_fea, _ = mol_dic1['atom_fea'].size()
    size_2 = mol_dic2['atom_fea'].size(-1) if mol_dic2['atom_fea'].dim() == 3 else mol_dic2['n_atom']

    new_mol_dict = {
        'atom_fea': torch.zeros(bsz, n_fea, size_1 + size_2,
                                dtype=mol_dic1['n_atom'].dtype, device=mol_dic1['n_atom'].device),
        'n_atom': mol_dic1['n_atom'] + mol_dic2['n_atom']
    }
    for i in range(bsz):
        n_atom_1 = mol_dic1['n_atom'][i]
        n_atom_2 = mol_dic2['n_atom'][i] if mol_dic2['atom_fea'].dim() == 3 else mol_dic2['n_atom']
        new_mol_dict['atom_fea'][i, :, :n_atom_1] = mol_dic1['atom_fea'][i, :, :n_atom_1]
        if mol_dic2['atom_fea'].dim() == 3:
            new_mol_dict['atom_fea'][i, :, n_atom_1:n_atom_1 + n_atom_2] = mol_dic2['atom_fea'][i, :, :n_atom_2]
        else:
            new_mol_dict['atom_fea'][i, :, n_atom_1:n_atom_1 + n_atom_2] = mol_dic2['atom_fea'][:, :n_atom_2]

        for key in ('bond_adj', 'dist_adj', 'dist_adj_3d'):
            if mol_dic1[key] is not None:
                new_mol_dict[key] = torch.zeros(bsz, size_1 + size_2, size_1 + size_2,
                                                dtype=mol_dic1[key].dtype, device=mol_dic1[key].device)
                new_mol_dict[key][i, :size_1, :size_1] = mol_dic1[key][i, :size_1, :size_1]
                if mol_dic2['atom_fea'].dim() == 3:
                    new_mol_dict[key][i, size_1:size_1 + size_2, size_1:size_1 + size_2] = mol_dic2[key][i, :size_2,
                                                                                           :size_2]
                else:
                    new_mol_dict[key][i, size_1:size_1 + size_2, size_1:size_1 + size_2] = mol_dic2[key][:size_2,
                                                                                           :size_2]

    return new_mol_dict


def padding_mol_info(mol_dic, max_nodes, padding_idx=0):
    """
    :param mol_dic={
            atom_fea: [n_atom_fea_type, n_atom],
            bond_adj: [n_hop, n_adj_type, n_atom, n_atom],
            ...,
    }
    :param max_nodes:
    :param padding_idx: value of padding token
    :return: {
            atom_fea: [n_atom_fea_type, max_nodes],
            bond_adj: [n_hop, n_adj_type, max_nodes, max_nodes],
            ...,
    }
    """
    if isinstance(mol_dic, dict):
        mol_dic = (mol_dic,)

    for mol_dic_i in mol_dic:
        n_atom_fea, n_atom = mol_dic_i['atom_fea'].size()
        new_atom_fea = torch.zeros(n_atom_fea, max_nodes, dtype=mol_dic_i['atom_fea'].dtype)
        new_atom_fea[:, :n_atom] = mol_dic_i['atom_fea']
        mol_dic_i['atom_fea'] = new_atom_fea
        new_bond_adj = torch.zeros(max_nodes, max_nodes, dtype=torch.uint8)
        new_bond_adj[:n_atom, :n_atom] = mol_dic_i['bond_adj'] + 1
        mol_dic_i['bond_adj'] = new_bond_adj
        mol_dic_i['dist_adj'] = pad_adj(mol_dic_i['dist_adj'] + 1e-05, max_nodes)
        if 'dist_adj_3d' in mol_dic_i.keys() and mol_dic_i['dist_adj_3d'] is None:
            del mol_dic_i['dist_adj_3d']
        else:
            if 'dist_adj_3d' in mol_dic_i.keys():
                mol_dic_i['dist_adj_3d'] = pad_adj(mol_dic_i['dist_adj_3d'] + 1e-05, max_nodes)

        if 'mol' in mol_dic_i.keys():
            del mol_dic_i['mol']


def cat_adj(m1, m2):
    n1, n2 = m1.size(0), m2.size(0)
    m_new = torch.zeros((n1 + n2), (n1 + n2), dtype=(m1.dtype))
    m_new[:n1, :n1] = m1
    m_new[n1:, n1:] = m2
    return m_new


def smile_to_mol_info(smile, calc_dist=True, use_3d_info=False, obj=None, prompt=False):
    mol = Chem.MolFromSmiles(smile)
    # --------
    if mol is None:
        return None
    # --------
    atom_fea, n_atom = get_atoms_info(mol)

    # for atom_idx in range(n_atom):`
    #     for fea_idx in range(len(DISCRETE_MAX)):
    #         if atom_fea[fea_idx, atom_idx] > DISCRETE_MAX[fea_idx]:
    #             return None

    bond_adj = get_bond_adj(mol)
    dist_adj = get_dist_adj(mol) if calc_dist else None
    dist_adj_3d = get_dist_adj(mol, use_3d_info) if calc_dist else None

    mol_infor = {'mol': mol,
                 'bond_adj': bond_adj,
                 'dist_adj': dist_adj,
                 'dist_adj_3d': dist_adj_3d,
                 'atom_fea': atom_fea,
                 'n_atom': n_atom}

    if prompt:
        mol_infor['graph_data'] = mol_infor_2_graph(mol_infor)

    return mol_infor


def mol_infor_2_graph(mol_info):
    """
    :param mol_info: {
            atom_fea: [n_atom_fea_type, n_atom],
            bond_adj: [n_hop, n_adj_type, n_atom, n_atom],
            ...,
    }
    :return: pyg.data.Data
    """
    atom_fea = mol_info['atom_fea'].t()
    edge_index = torch.where(mol_info['bond_adj'] > 0, 1, 0).nonzero().t()
    edge_attr = mol_info['bond_adj'][edge_index[0], edge_index[1]]

    return Data(
        x=atom_fea,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )


def mol2graph(mol):
    """
    :param mol: RDKit mol object
    :return: pyg.data.Data
    """

    edge_index = []
    edge_attr = []
    mol_fea, n_atom = get_atoms_info(mol)

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append(torch.LongTensor([i, j]))
        edge_index.append(torch.LongTensor([i, j]))
        edge_attr.append(torch.LongTensor([BOND_ORDER_MAP[bond.GetBondTypeAsDouble()]]))
        edge_attr.append(torch.LongTensor([BOND_ORDER_MAP[bond.GetBondTypeAsDouble()]]))

    edge_index = torch.stack(edge_index, dim=0).t().contiguous()
    edge_attr = torch.cat(edge_attr, dim=0)

    return Data(
        x=mol_fea,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )


# 得到原子信息
def get_atoms_info(mol):
    atoms = mol.GetAtoms()
    n_atom = len(atoms)
    atom_fea = torch.zeros(NUM_ATOM_FEAT, n_atom, dtype=torch.float)  # 分子特征矩阵 两行n列，一列是一个原子;第一行是原子数，第二行是
    # AllChem.ComputeGasteigerCharges(mol)  # 将mol中每个原子的Gasteiger Charges计算出来，并保存在每一个原子的_GasteigerCharge中;
    for idx, atom in enumerate(atoms):
        atom_fea[0, idx] = atom.GetAtomicNum()  # 原子的原子序数
        atom_fea[1, idx] = atom.GetTotalDegree() + 1  # 原子的度数
        atom_fea[2, idx] = atom.GetFormalCharge() + 10  # 原子的电荷数
        atom_fea[3, idx] = atom.GetTotalNumHs() + 1  # 原子的氢原子数
        atom_fea[4, idx] = int(atom.GetChiralTag()) + 1  # 原子的手性标记
    return atom_fea, n_atom


def get_bond_order_adj(mol):
    n_atom = len(mol.GetAtoms())
    bond_adj = torch.zeros(n_atom, n_atom, dtype=torch.float)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # bond_adj[(i, j)] = bond_adj[(j, i)] = BOND_ORDER_MAP[bond.GetBondTypeAsDouble()]
        bond_adj[(i, j)] = bond_adj[(j, i)] = bond.GetBondTypeAsDouble()

    return bond_adj


# 化学键邻接矩阵。节点是原子
def get_bond_adj(mol):
    """
    :param mol: rdkit mol
    :return: multi graph for {
                sigmoid_bond_graph,
                pi_bond_graph,
                2pi_bond_graph,
                aromic_graph,
                conjugate_graph,
                ring_graph,
    }
    """
    n_atom = len(mol.GetAtoms())
    bond_adj = torch.zeros(n_atom, n_atom, dtype=torch.uint8)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()
        bond_adj[(i, j)] = bond_adj[(j, i)] = 1
        if bond_type in (2, 3):
            bond_adj[(i, j)] = bond_adj[(j, i)] = bond_adj[(i, j)] + 2
        if bond_type == 3:
            bond_adj[(i, j)] = bond_adj[(j, i)] = bond_adj[(i, j)] + 4
        if bond_type == 1.5:
            bond_adj[(i, j)] = bond_adj[(j, i)] = bond_adj[(i, j)] + 8

    return bond_adj


def get_tgt_adj_order(product, reactants):
    p_idx2map_idx = {}
    for atom in product.GetAtoms():
        p_idx2map_idx[atom.GetIdx()] = atom.GetAtomMapNum()
        # ‘GetAtomMapNum’: map id 原子smarts形式冒号后面的数字，如[N:4], map id 就是4
        # 反应过程中，原子如果做了标记，就会与对应的被标记的原子匹配

    map_idx2r_idx = {0: []}
    for atom in reactants.GetAtoms():
        if atom.GetAtomMapNum() == 0 or atom.GetAtomMapNum() not in p_idx2map_idx.values():
            map_idx2r_idx[0].append(atom.GetIdx())
        else:
            map_idx2r_idx[atom.GetAtomMapNum()] = atom.GetIdx()

    order = []
    for atom in product.GetAtoms():
        order.append(map_idx2r_idx[p_idx2map_idx[atom.GetIdx()]])

    order.extend(map_idx2r_idx[0])
    return torch.tensor(order, dtype=torch.long)


def atom_cmp(a1, a2):
    an1, an2 = a1.GetAtomicNum(), a2.GetAtomicNum()
    if an1 != an2:
        return an1 - an2
    hy1, hy2 = a1.GetHybridization(), a2.GetHybridization()
    return hy1 - hy2


def get_dist_adj(mol, use_3d_info=False):
    if use_3d_info:
        m2 = Chem.AddHs(mol)
        # 加氢
        # 在rdkit中，分子在默认情况下是不显示氢的，所以在计算3D构象前，需要使用Chem.AddHs()方法加上氢原子。
        is_success = AllChem.EmbedMolecule(m2, enforceChirality=False)  # 生成3D构象
        if is_success == -1:
            dist_adj = None
        else:
            AllChem.MMFFOptimizeMolecule(m2)
            m2 = Chem.RemoveHs(m2)  # 减氢
            dist_adj = torch.tensor((AllChem.Get3DDistanceMatrix(m2)), dtype=(torch.float))
    else:
        dist_adj = torch.tensor((molDG.GetMoleculeBoundsMatrix(mol)), dtype=(torch.float))
        # Returns the distance bounds matrix（距离边界矩阵）for a molecule
    return dist_adj


def pad_1d(x, n_max_nodes):
    if not isinstance(x, Tensor):
        raise TypeError(type(x), 'is not a torch.Tensor.')
    n = x.size(0)
    new_x = torch.zeros(n_max_nodes).to(x)
    new_x[:n] = x
    return new_x


def pad_adj(x, n_max_nodes):
    if x is None:
        return
    if not isinstance(x, Tensor):
        raise TypeError(type(x), 'is not a torch.Tensor.')
    n = x.size(0)
    assert x.size(0) == x.size(1)
    new_x = torch.zeros([n_max_nodes, n_max_nodes], dtype=(x.dtype))
    new_x[:n, :n] = x
    return new_x


def get_downstream_task_info(config):
    """
    Get task names of downstream dataset
    """
    if config['task_name'] == 'bace':
        target = ["Class"]
        task = 'classification'
        loss_type = 'bce'
    elif config['task_name'] == 'bbbp':
        target = ["p_np"]
        task = 'classification'
        loss_type = 'bce'
    elif config['task_name'] == 'clintox':
        target = ['CT_TOX', 'FDA_APPROVED']
        task = 'classification'
        loss_type = 'bce'
    elif config['task_name'] == 'hiv':
        target = ["HIV_active"]
        task = 'classification'
        loss_type = 'bce'
    elif config['task_name'] == 'muv':
        target = [
            'MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644', 'MUV-548', 'MUV-852',
            'MUV-600', 'MUV-810', 'MUV-712', 'MUV-737', 'MUV-858', 'MUV-713', 'MUV-733',
            'MUV-652', 'MUV-466', 'MUV-832'
        ]
        task = 'classification'
        loss_type = 'bce'
    elif config['task_name'] == 'sider':
        target = [
            "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues",
            "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders",
            "Gastrointestinal disorders", "Social circumstances", "Immune system disorders",
            "Reproductive system and breast disorders",
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
            "General disorders and administration site conditions", "Endocrine disorders",
            "Surgical and medical procedures", "Vascular disorders",
            "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders",
            "Congenital, familial and genetic disorders", "Infections and infestations",
            "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders",
            "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions",
            "Ear and labyrinth disorders", "Cardiac disorders",
            "Nervous system disorders", "Injury, poisoning and procedural complications"
        ]
        task = 'classification'
        loss_type = 'bce'
    elif config['task_name'] == 'tox21':
        target = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]
        task = 'classification'
        loss_type = 'bce'
    elif config['task_name'] == 'toxcast':
        # raw_path = os.path.join(config['root'], 'toxcast', 'raw')
        raw_path = os.path.join(config['root'], 'raw')
        csv_file = os.listdir(raw_path)[0]
        input_df = pd.read_csv(os.path.join(raw_path, csv_file), sep=',')
        target = list(input_df.columns)[1:]
        task = 'classification'
        loss_type = 'bce'
        # 回归数据集
    elif config['task_name'] == 'esol':
        target = ["measured log solubility in mols per litre"]
        task = 'regression'
        loss_type = 'mse'
    elif config['task_name'] == 'freesolv':
        target = ["expt"]
        task = 'regression'
        loss_type = 'mse'
    elif config['task_name'] == 'lipophilicity':
        target = ['exp']
        task = 'regression'
        loss_type = 'mse'
    elif config['task_name'] == 'qm7':
        target = ['u0_atom']
        task = 'regression'
        loss_type = 'l1'
    elif config['task_name'] == 'qm8':
        target = ['E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2',
                  'E1-PBE0', 'E2-PBE0', 'f1-PBE0', 'f2-PBE0',
                  'E1-CAM', 'E2-CAM', 'f1-CAM', 'f2-CAM']
        task = 'regression'
        loss_type = 'l1'
    elif config['task_name'] == 'qm9':
        target = ['homo', 'lumo', 'gap']
        task = 'regression'
        loss_type = 'l1'
    elif config['task_name'] == 'physprop_perturb':
        target = ['LogP']
        task = 'regression'
        loss_type = 'mse'
    else:
        return None
    config['target'] = target
    config['task'] = task
    config['loss_type'] = loss_type
    config['num_tasks'] = len(target)
    return config


def load_finetune_dataset(input_path, target, prompt=False):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    labels = input_df[target]
    labels = labels.fillna(-1).values.tolist()
    real_smiles_list = []
    real_labels = []
    for idx, smile in enumerate(smiles_list):
        if Chem.MolFromSmiles(smile) is not None:
            real_smiles_list.append(smile)
            real_labels.append(labels[idx])
    smiles_list = real_smiles_list
    mol_info = [smile_to_mol_info(s, prompt=prompt) for s in tqdm(smiles_list)]

    # convert 0 to -1
    # labels = labels.replace(0, -1)
    # there are no nans

    labels = real_labels
    assert len(smiles_list) == len(mol_info)
    assert len(smiles_list) == len(labels)
    return smiles_list, mol_info, torch.tensor(labels)


# 从paddlepaddle中的pahelix中复制
# 有很多种数据划分方式


import random
import numpy as np
from itertools import compress
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

__all__ = [
    'RandomSplitter',
    'IndexSplitter',
    'ScaffoldSplitter',
    'RandomScaffoldSplitter',
]


def create_splitter(split_type):
    """Return a splitter according to the ``split_type``"""
    if split_type == 'random':
        splitter = RandomSplitter()
    elif split_type == 'index':
        splitter = IndexSplitter()
    elif split_type == 'scaffold':
        splitter = ScaffoldSplitter()
    elif split_type == 'random_scaffold':
        splitter = RandomScaffoldSplitter()
    else:
        raise ValueError('%s not supported' % split_type)
    return splitter


def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles

    Args:
        smiles: smiles sequence
        include_chirality: Default=False

    Return:
        the scaffold of the given smiles.
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold


class Splitter(object):
    """
    The abstract class of splitters which split up dataset into train/valid/test
    subsets.
    """

    def __init__(self):
        super(Splitter, self).__init__()


class RandomSplitter(Splitter):
    """
    Random splitter.
    """

    def __init__(self):
        super(RandomSplitter, self).__init__()

    def split(self,
              dataset,
              frac_train=None,
              frac_valid=None,
              frac_test=None,
              seed=9):
        """
        Args:
            dataset(InMemoryDataset): the dataset to split.
            frac_train(float): the fraction of pretrain_data to be used for the train split.
            frac_valid(float): the fraction of pretrain_data to be used for the valid split.
            frac_test(float): the fraction of pretrain_data to be used for the test split.
            seed(int|None): the random seed.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)

        indices = list(range(N))
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
        train_cutoff = int(frac_train * N)
        valid_cutoff = int((frac_train + frac_valid) * N)
        if isinstance(dataset, list):
            train_dataset = np.array(dataset)[indices[:train_cutoff]].tolist()
            valid_dataset = np.array(dataset)[indices[train_cutoff:valid_cutoff]].tolist()
            test_dataset = np.array(dataset)[indices[valid_cutoff:]].tolist()
        else:
            train_dataset = dataset[indices[:train_cutoff]]
            valid_dataset = dataset[indices[train_cutoff:valid_cutoff]]
            test_dataset = dataset[indices[valid_cutoff:]]

        return train_dataset, valid_dataset, test_dataset


class IndexSplitter(Splitter):
    """
    Split daatasets that has already been orderd. The first `frac_train` proportion
    is used for train set, the next `frac_valid` for valid set and the final `frac_test`
    for test set.
    """

    def __init__(self):
        super(IndexSplitter, self).__init__()

    def split(self,
              dataset,
              frac_train=None,
              frac_valid=None,
              frac_test=None):
        """
        Args:
            dataset(InMemoryDataset): the dataset to split.
            frac_train(float): the fraction of pretrain_data to be used for the train split.
            frac_valid(float): the fraction of pretrain_data to be used for the valid split.
            frac_test(float): the fraction of pretrain_data to be used for the test split.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)

        indices = list(range(N))
        train_cutoff = int(frac_train * N)
        valid_cutoff = int((frac_train + frac_valid) * N)

        train_dataset = dataset[indices[:train_cutoff]]
        valid_dataset = dataset[indices[train_cutoff:valid_cutoff]]
        test_dataset = dataset[indices[valid_cutoff:]]
        return train_dataset, valid_dataset, test_dataset


class ScaffoldSplitter(Splitter):
    """
    Adapted from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py

    Split dataset by Bemis-Murcko scaffolds
    """

    def __init__(self):
        super(ScaffoldSplitter, self).__init__()

    def split(self,
              dataset,
              frac_train=None,
              frac_valid=None,
              frac_test=None,
              seed=None):
        """
        Args:
            dataset(InMemoryDataset): the dataset to split. Make sure each element in
                the dataset has key "smiles" which will be used to calculate the
                scaffold.
            frac_train(float): the fraction of pretrain_data to be used for the train split.
            frac_valid(float): the fraction of pretrain_data to be used for the valid split.
            frac_test(float): the fraction of pretrain_data to be used for the test split.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)

        # create dict of the form {scaffold_i: [idx1, idx....]}
        all_scaffolds = {}

        for i, smiles in enumerate(dataset.smiles_list):
            scaffold = generate_scaffold(smiles, include_chirality=False)
            if scaffold not in all_scaffolds:
                all_scaffolds[scaffold] = [i]
            else:
                all_scaffolds[scaffold].append(i)

        # sort from largest to smallest sets
        all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
        all_scaffold_sets = [
            scaffold_set for (scaffold, scaffold_set) in sorted(
                all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
        ]

        # get train, valid test indices
        train_cutoff = frac_train * N
        valid_cutoff = (frac_train + frac_valid) * N
        train_idx, valid_idx, test_idx = [], [], []
        for scaffold_set in all_scaffold_sets:
            if len(train_idx) + len(scaffold_set) > train_cutoff:
                if len(train_idx) + len(test_idx) + len(scaffold_set) > valid_cutoff:
                    valid_idx.extend(scaffold_set)
                else:
                    test_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        assert len(set(train_idx).intersection(set(valid_idx))) == 0
        assert len(set(test_idx).intersection(set(valid_idx))) == 0

        train_dataset = dataset[train_idx]
        valid_dataset = dataset[valid_idx]
        test_dataset = dataset[test_idx]
        return train_dataset, valid_dataset, test_dataset


class RandomScaffoldSplitter(Splitter):
    """
    Adapted from https://github.com/pfnet-research/chainer-chemistry/blob/master/chainer_chemistry/dataset/splitters/scaffold_splitter.py

    Split dataset by Bemis-Murcko scaffolds
    """

    def __init__(self):
        super(RandomScaffoldSplitter, self).__init__()

    def split(self,
              dataset,
              frac_train=None,
              frac_valid=None,
              frac_test=None,
              seed=None):
        """
        Args:
            dataset(InMemoryDataset): the dataset to split. Make sure each element in
                the dataset has key "smiles" which will be used to calculate the
                scaffold.
            frac_train(float): the fraction of pretrain_data to be used for the train split.
            frac_valid(float): the fraction of pretrain_data to be used for the valid split.
            frac_test(float): the fraction of pretrain_data to be used for the test split.
            seed(int|None): the random seed.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)

        rng = np.random.RandomState(seed)

        scaffolds = defaultdict(list)
        for ind in range(N):
            scaffold = generate_scaffold(dataset.smiles_list[ind], include_chirality=True)
            scaffolds[scaffold].append(ind)

        scaffold_sets = rng.permutation(np.array(list(scaffolds.values()), dtype=object))

        n_total_valid = int(np.floor(frac_valid * len(dataset)))
        n_total_test = int(np.floor(frac_test * len(dataset)))

        train_idx = []
        valid_idx = []
        test_idx = []

        for scaffold_set in scaffold_sets:
            if len(valid_idx) + len(scaffold_set) <= n_total_valid:
                valid_idx.extend(scaffold_set)
            elif len(test_idx) + len(scaffold_set) <= n_total_test:
                test_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        train_dataset = dataset[train_idx]
        valid_dataset = dataset[valid_idx]
        test_dataset = dataset[test_idx]
        return train_dataset, valid_dataset, test_dataset


class BalancedRandomScaffoldSplitter(Splitter):
    """
    Adapted from https://github.com/pfnet-research/chainer-chemistry/blob/master/chainer_chemistry/dataset/splitters/scaffold_splitter.py

    Split dataset by Bemis-Murcko scaffolds
    """

    def __init__(self):
        super(BalancedRandomScaffoldSplitter, self).__init__()

    def split(self,
              dataset,
              frac_train=None,
              frac_valid=None,
              frac_test=None,
              seed=None):
        """
        Args:
            dataset(InMemoryDataset): the dataset to split. Make sure each element in
                the dataset has key "smiles" which will be used to calculate the
                scaffold.
            frac_train(float): the fraction of pretrain_data to be used for the train split.
            frac_valid(float): the fraction of pretrain_data to be used for the valid split.
            frac_test(float): the fraction of pretrain_data to be used for the test split.
            seed(int|None): the random seed.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)

        rng = np.random.RandomState(seed)

        scaffolds = defaultdict(list)
        for ind in range(N):
            scaffold = generate_scaffold(dataset.smiles_list[ind], include_chirality=True)
            scaffolds[scaffold].append(ind)

        scaffold_sets = rng.permutation(np.array(list(scaffolds.values()), dtype=object))

        n_total_train = int(np.floor(frac_train * len(dataset)))
        n_total_valid = int(np.floor(frac_valid * len(dataset)))
        n_total_test = int(np.floor(frac_test * len(dataset)))

        train_idx = []
        valid_idx = []
        test_idx = []

        for scaffold_set in scaffold_sets:
            big_index_sets = []
            small_index_sets = []
            for scaffold_set in scaffold_sets:
                if len(scaffold_set) > n_total_valid / 2 or len(scaffold_set) > n_total_test / 2:
                    big_index_sets.append(scaffold_set)
                else:
                    small_index_sets.append(scaffold_set)
            scaffold_sets = big_index_sets + small_index_sets

        for scaffold_set in scaffold_sets:
            if len(train_idx) + len(scaffold_set) <= n_total_valid:
                train_idx.extend(scaffold_set)
            elif len(valid_idx) + len(scaffold_set) <= n_total_valid:
                valid_idx.extend(scaffold_set)
            else:
                test_idx.extend(scaffold_set)

        train_dataset = dataset[train_idx]
        valid_dataset = dataset[valid_idx]
        test_dataset = dataset[test_idx]
        return train_dataset, valid_dataset, test_dataset
