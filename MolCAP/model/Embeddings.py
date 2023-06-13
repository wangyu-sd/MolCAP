# !/usr/bin/python3
# @File: Embeddings.py
# --coding:utf-8--
# @Author:yuwang
# @Email:as1003208735@foxmail.com
# @Time: 2022.04.09.14
import torch
from torch import nn
from data.data_utils import NUM_ATOM_FEAT
from torch.nn import functional as F
PI = 3.14159
A = (2 * PI) ** 0.5
import math
import numpy as np


class EmbeddingLayer(nn.Module):
    def __init__(self, d_model, n_head, max_single_hop, n_layers, need_graph_token, use_3d_info=False, dropout=0.,
                 known_rxn_cnt=True, use_dist_adj=True):
        super(EmbeddingLayer, self).__init__()

        self.atom_encoder = AtomFeaEmbedding(d_model, n_layers, need_graph_token, known_rxn_cnt=known_rxn_cnt)
        self.edge_encoder = EdgeEmbedding(d_model, n_head, max_single_hop, n_layers, need_graph_token,
                                          use_3d_info=use_3d_info, known_rxn_cnt=known_rxn_cnt,
                                          use_dist_adj=use_dist_adj)

    def forward(self, atom_fea, bond_adj, dist_adj, center_cnt=None, dist3d_adj=None, contrast=False):
        return self.atom_encoder(atom_fea, center_cnt, contrast), \
               self.edge_encoder(bond_adj, dist_adj, center_cnt, dist3d_adj, contrast)


class AtomFeaEmbedding(nn.Module):
    def __init__(self, d_model, n_layers=1, need_graph_token=True, known_rxn_cnt=False):
        super(AtomFeaEmbedding, self).__init__()
        self.known_rxn_cnt = known_rxn_cnt
        self.atom_encoders = nn.ModuleList([
            nn.Embedding(100, d_model, padding_idx=0),  # 原子数
            nn.Embedding(40, d_model, padding_idx=0),  # 度数
            nn.Embedding(30, d_model, padding_idx=0),  # 电荷数
            nn.Embedding(20, d_model, padding_idx=0),  # 氢变化
            nn.Embedding(5, d_model, padding_idx=0)  # 手性
        ])
        self.need_graph_token = need_graph_token

        if need_graph_token:
            self.graph_token = nn.Embedding(1, d_model)
            if known_rxn_cnt:
                self.cnt_token = nn.Embedding(50, d_model)
            self.contrast_token = nn.Embedding(1, d_model)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, atom_fea, center_cnt=None, contrast=False):
        """
        :param atom_fea: [bsz, n_fea_type, n_atom]
        :return: [bsz, n_atom + I, d_model] I = 1 if need_graph_token else 0
        """
        if atom_fea.dim() == 3:
            bsz, n_fea_type, n_atom = atom_fea.size()
            out = 0

            for idx in range(n_fea_type):
                # print(idx, atom_fea[:, idx].int().max(), atom_fea[:, idx].int().min())
                out += self.atom_encoders[idx](atom_fea[:, idx].long())

            if self.need_graph_token:
                graph_token = self.graph_token.weight[0].clone()
                if contrast:
                    graph_token += self.contrast_token.weight[0]

                graph_token = graph_token.view(1, 1, -1).repeat(bsz, 1, 1)

                if self.known_rxn_cnt:
                    # print(center_cnt.max())
                    graph_token += self.cnt_token(center_cnt).view(bsz, 1, -1)

                out = torch.cat([graph_token, out], dim=1)

        elif atom_fea.dim() == 2:
            _, n_fea_type = atom_fea.size()
            out = 0
            for idx in range(n_fea_type):
                # print(idx, atom_fea[:, idx].int().max(), atom_fea[:, idx].int().min())
                out += self.atom_encoders[idx](atom_fea[:, idx].long())
        else:
            raise ValueError("atom_fea.dim() must be 2 or 3")

        return out


class EdgeEmbedding(nn.Module):
    def __init__(self, embed_dim, n_head=16, max_single_hop=4, n_layers=1, need_graph_token=True, use_3d_info=False,
                 known_rxn_cnt=True, use_dist_adj=True, max_paths=50, n_graph_type=4, ):
        """
        :param embed_dim:
        :param n_head:
        :param max_paths:
        :param n_graph_type:
        :param max_single_hop:
        :param n_layers:
        """
        super().__init__()
        self.known_rxn_cnt = known_rxn_cnt
        self.num_heads = n_head
        self.embed_dim = embed_dim
        # assert n_head & (n_head - 1) == 0, f"n_head mast has to be 2 to the nth power, but got{n_head}"
        self.n_graph_type = n_graph_type
        self.max_single_hop = max_single_hop
        self.use_3d_info = use_3d_info
        self.use_dist_adj = use_dist_adj
        self.max_paths = max_paths
        # self.hop_distribution = [1 << i for i in range(int(math.log2(n_head) - 1), -1, -1)]
        # self.hop_distribution[-1] += 1
        # assert sum(self.hop_distribution) == self.num_heads

        self.head_dim = embed_dim // n_head
        assert self.head_dim * n_head == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.edge_encoders = nn.ModuleList([
            nn.Embedding(max_paths + 1, n_head, padding_idx=0) for _ in range(n_graph_type)
        ])
        self.spatial_encoder = GaussianBondLayer(n_head, means=(0, 3), stds=(0.1, 10))
        if self.use_3d_info:
            self.spatial3d_encoder = GaussianBondLayer(n_head, means=(0, 3), stds=(0.1, 10))

        # self.norm = nn.LayerNorm(n_head)

        self.need_graph_token = need_graph_token
        if need_graph_token:
            self.graph_token = nn.Embedding(1, n_head)
            self.contrast_token = nn.Embedding(1, n_head)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, bond_adj, dist_adj, center_cnt=None, dist3d_adj=None, contrast=False):
        """
        :param bond_adj: [bsz, n_atom, n_atom]
        :param dist_adj: [bsz, n_atom, n_atom]
        :param rxn_type: [bsz]
        :return: attention bias with mask [bsz*n_head, n_atom, n_atom]
        """
        # bsz, n_hop, n_type, n_atom, _ = bond_adj.size()
        # bond_embed = self.edge_encoders[0](bond_adj[:, :, 0].int()).sum(dim=1)
        # for i in range(1, self.n_graph_type):
        #     bond_embed += self.edge_encoders[i](bond_adj[:, :, i].int()).sum(dim=1)

        # [bsz, n_hop, n_type, n_atom, n_atom, n_head] -> # [bsz, n_atom, n_atom, n_head]

        # bond_embed = torch.cat([
        #     bond_embed[:, i].unsqueeze(1).expand(bsz, hop, n_atom, n_atom)
        #     for i, hop in enumerate(self.hop_distribution)
        # ], dim=1)  # [bsz, n_hop, n_type, n_atom, n_atom] -> # [bsz, n_head, n_atom, n_atom]

        bsz, n_atom, _ = bond_adj.size()
        if self.use_dist_adj and dist_adj is not None:
            comb_embed = self.spatial_encoder(dist_adj)  # [bsz, n_atom, n_atom] -> [bsz, n_atom, n_atom, n_head]
        else:
            comb_embed = torch.zeros(bsz, n_atom, n_atom, self.num_heads, device=bond_adj.device)
        if self.use_3d_info and dist3d_adj is not None:
            comb_embed += self.spatial3d_encoder(dist3d_adj)

        if self.max_single_hop > 0:
            for i in range(self.n_graph_type):
                j_hop_embed = bond_adj.long()
                # decode to multi sense embedding
                j_hop_embed = torch.where(j_hop_embed > 0, ((j_hop_embed - 1) >> i) % 2, 0).float()
                base_hop_embed = j_hop_embed
                comb_embed += self.edge_encoders[i](j_hop_embed.int())
                for j in range(1, self.max_single_hop):
                    # generate multi atom environment embedding
                    j_hop_embed = torch.bmm(j_hop_embed, base_hop_embed)
                    # print(j_hop_embed.max())
                    j_hop_embed = - F.relu(-j_hop_embed + self.max_paths) + self.max_paths
                    comb_embed += self.edge_encoders[i](j_hop_embed.int())

        comb_embed = comb_embed.permute(0, 3, 1, 2)
        if self.need_graph_token:
            graph_token = self.graph_token.weight[0].clone()
            if contrast:
                graph_token += self.contrast_token.weight[0]
            graph_token = graph_token.view(1, self.num_heads, 1, 1).repeat(bsz, 1, 1, 1)

            if self.known_rxn_cnt:
                graph_token += self.cnt_token(center_cnt).view(bsz, self.num_heads, 1, 1)

            comb_embed = torch.cat([graph_token.expand(bsz, self.num_heads, n_atom, 1),
                                    comb_embed], dim=-1)
            comb_embed = torch.cat([graph_token.expand(bsz, self.num_heads, 1, n_atom + 1),
                                    comb_embed], dim=-2)

            mask = torch.where(bond_adj == 0, False, True)
            mask = torch.cat([mask[:, :, 0].unsqueeze(-1), mask], dim=-1)
            mask = torch.cat([mask[:, 0, :].unsqueeze(-2), mask], dim=-2)

            # [bsz, n_head, n_atom+1, n_atom+1]
            n_atom += 1
            comb_embed.masked_fill_(
                mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1),
                -1e3
            )

        else:
            comb_embed.masked_fill_(
                (bond_adj == 0).unsqueeze(1).repeat(1, self.num_heads, 1, 1),
                -1e3
            )
        return comb_embed.contiguous().view(bsz * self.num_heads, n_atom, n_atom)


def gaussian(x, mean, std):
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (A * std)


class GaussianAtomLayer(nn.Module):
    def __init__(self, d_model=128, means=(0, 3), stds=(0.1, 10)):
        super().__init__()
        self.d_model = d_model
        self.means = nn.Embedding(1, d_model)
        self.stds = nn.Embedding(1, d_model)
        self.mul = nn.Embedding(1, 1)
        self.bias = nn.Embedding(1, 1)
        nn.init.uniform_(self.means.weight, means[0], means[1])
        nn.init.uniform_(self.stds.weight, stds[0], stds[1])
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def real_forward(self, x):
        mul = self.mul.weight
        bias = self.bias.weight
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, self.d_model)
        mean = self.means.weight[0]
        std = self.stds.weight[0].abs() + 1e-5
        return gaussian(x, mean, std).type_as(self.means.weight)

    def forward(self, x):
        """
        :param x: [bsz, n_atom]
        :return: [bsz, n_atom, d_model]
        """
        out = self.real_forward(x)
        return torch.where(x.unsqueeze(-1).expand_as(out) != 0, out, torch.zeros_like(out))


class GaussianBondLayer(nn.Module):
    def __init__(self, nhead=16, means=(0, 3), stds=(0.1, 10)):
        super().__init__()
        self.nhead = nhead
        self.means = nn.Embedding(1, nhead)
        self.stds = nn.Embedding(1, nhead)
        self.mul = nn.Embedding(1, 1)
        self.bias = nn.Embedding(1, 1)
        nn.init.uniform_(self.means.weight, means[0], means[1])
        nn.init.uniform_(self.stds.weight, stds[0], stds[1])
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def real_forward(self, x):
        mul = self.mul.weight
        bias = self.bias.weight
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.nhead)
        mean = self.means.weight[0]
        std = self.stds.weight[0].abs() + 1e-5
        return gaussian(x, mean, std).type_as(self.means.weight)

    def forward(self, x):
        """
        :param x: [bsz, n_atom, n_atom]
        :return: [bsz, n_atom, n_atom, nhead]
        """
        out = self.real_forward(x)
        return torch.where(x.unsqueeze(-1).expand_as(out) != 0, out, torch.zeros_like(out))


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
