# !/usr/bin/python3
# @File: base_prompt_models.py
# --coding:utf-8--
# @Author:yuwang
# @Email:as1003208735@foxmail.com
# @Time: 2023/5/15 21:17
import torch

from model.Embeddings import AtomFeaEmbedding
from torch_geometric.nn import GAT, GraphSAGE, GIN, GCN, pool


class MolGNNLayers(torch.nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.4,  mode='prompt_gat_v2_classification', num_tasks=1, num_layers=4):
        super(MolGNNLayers, self).__init__()
        self.atom_encoder = AtomFeaEmbedding(d_model)
        self.edge_encoder = BondEncoder(nhead)
        self.down_stream_out_fn = torch.nn.Linear(d_model, num_tasks)

        if 'prompt_gat_v2' in mode:
            self.gnn = GAT(in_channels=d_model,
                           hidden_channels=d_model,
                           num_layers=num_layers,
                           out_channel=d_model,
                           dropout=dropout,
                           v2=True,
                           heads=nhead,
                           edge_dim=nhead,
                           )
        elif 'prompt_gat_v1' in mode:
            self.gnn = GAT(in_channels=d_model,
                           hidden_channels=d_model,
                           num_layers=num_layers,
                           out_channel=d_model,
                           dropout=dropout,
                           v2=False,
                           heads=nhead,
                           edge_dim=nhead,
                           )
        elif 'prompt_gcn' in mode:
            self.gnn = GCN(in_channels=d_model,
                           hidden_channels=d_model,
                           num_layers=num_layers,
                           out_channel=d_model,
                           dropout=dropout,
                           edge_dim=nhead,
                           )
        elif 'prompt_gin' in mode:
            self.gnn = GIN(in_channels=d_model,
                           hidden_channels=d_model,
                           num_layers=num_layers,
                           out_channel=d_model,
                           dropout=dropout,
                           edge_dim=nhead,
                           )
        elif 'prompt_graphsage' in mode:
            self.gnn = GraphSAGE(in_channels=d_model,
                                 hidden_channels=d_model,
                                 num_layers=num_layers,
                                 out_channel=d_model,
                                 dropout=dropout,
                                 edge_dim=nhead,
                                 )
        else:
            raise ValueError('Invalid mol_encoder: %s' % mode)

    def forward(self, batch, atom_bias=None, bond_bias=None, blend_type='add'):
        x = self.atom_encoder(batch.x)
        edge_attr = self.edge_encoder(batch.edge_attr)
        if bond_bias is not None:
            if blend_type == 'add':
                edge_attr += bond_bias
            else:
                edge_attr *= bond_bias
        if atom_bias is not None:
            if blend_type == 'add':
                x += atom_bias
            else:
                x *= atom_bias
        x = self.gnn(x, edge_index=batch.edge_index, edge_attr=edge_attr)
        x = pool.global_add_pool(x, batch.batch)
        x = self.down_stream_out_fn(x)
        return x


class BondEncoder(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(BondEncoder, self).__init__()
        self.bond_embedding = torch.nn.Embedding(10, embedding_dim)
        torch.nn.init.xavier_uniform_(self.bond_embedding.weight.data)

    def forward(self, edge_attr):
        """
        :param edge_attr: edge attributes, shape (num_batched_edges, num_bond_fea)
        :return: bond embeddings, shape (num_batched_edges, embedding_dim)
        """
        return self.bond_embedding(edge_attr.long())
