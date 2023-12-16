# !/usr/bin/python3
# @File: RetroAGT.py
# --coding:utf-8--
# @Author:yuwang
# @Email:as1003208735@foxmail.com
# @Time: 2022.03.20.22
import sys

sys.path.append("..")
sys.path.append("./model")

import pytorch_lightning as pl
import torch
from torch import nn, Tensor
from model.LearningRate import PolynomialDecayLR
from model.Embeddings import EmbeddingLayer
from model.Modules import EncoderLayer, Encoder, MultiHeadAtomAdj
from torcheval.metrics import functional as TMF
from torchmetrics.classification import Accuracy, Precision, Recall
from torcheval.metrics import BinaryAUROC
from data.data_utils import NUM_ATOM_FEAT, concat_padded_mol_dict, concat_prompt_mol_dict
# from model.util import LossRecorderList, build_gnn_base_model
import torch.nn.functional as F
from itertools import chain
from model.base_prompt_models import MolGNNLayers
from sklearn.metrics import cohen_kappa_score
from model import util

import nni

log_metric = []


class IMLearner(pl.LightningModule):
    def __init__(self,
                 d_model=512,
                 nhead=32,
                 num_encoder_layer=12,
                 dim_feedforward=512,
                 dropout=0.1,
                 max_single_hop=4,
                 n_layers=1,
                 batch_first=True,
                 norm_first=True,
                 activation='gelu',
                 warmup_updates=6e4,
                 tot_updates=1e6,
                 peak_lr=2e-4,
                 end_lr=1e-9,
                 weight_decay=0.99,
                 use_3d_info=False,
                 use_dist_adj=True,
                 mode='pretrain',
                 num_tasks=3,
                 batch_considered=50,
                 loss_type='bce',
                 prompt_mol='',
                 n_downstream_layer=6,
                 return_embedding=False, ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layer = num_encoder_layer
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.activation = activation
        self.warmup_updates = warmup_updates
        self.peak_lr = peak_lr
        self.tot_updates = tot_updates
        self.end_lr = end_lr
        self.weight_decay = weight_decay
        self.use_3d_info = use_3d_info
        self.use_dist_adj = use_dist_adj
        self.mode = mode
        self.num_tasks = num_tasks
        self.use_adaptive_multi_task = True
        self.batch_considered = batch_considered
        self.loss_type = loss_type
        self.prompt_mol = prompt_mol
        self.n_downstream_layer = n_downstream_layer
        self.return_embedding = return_embedding
        self.predict_state = False
        encoder_layer = EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                     batch_first=batch_first, norm_first=norm_first)
        self.emb = EmbeddingLayer(d_model, nhead, max_single_hop, n_layers, need_graph_token=False,
                                  use_3d_info=use_3d_info, dropout=dropout, use_dist_adj=use_dist_adj)

        self.encoder = Encoder(encoder_layer, num_encoder_layer, nn.LayerNorm(d_model))
        self.bond_fn = MultiHeadAtomAdj(d_model, nhead, bias=True, batch_first=batch_first)

        if self.mode == 'pretrain':
            self.loss_init = None
            self.loss_last = None
            self.loss_last2 = None

            self.out_fn_bond = nn.Sequential(
                nn.Linear(nhead, 13)
            )
            self.out_fn_atoms = nn.ModuleList(
                [nn.Linear(d_model, 50) for _ in range(NUM_ATOM_FEAT)]
            )
            self.criterion = nn.CrossEntropyLoss()

        elif "finetune" in self.mode:
            # self.down_stream_encoder = Encoder(encoder_layer, n_downstream_layer, nn.LayerNorm(d_model))
            # #
            # # if self.mode in ['finetune_DDI', 'finetune_DDS']:
            # #     self.down_stream_decoder = Decoder(decoder_layer, 3, nn.LayerNorm(d_model))
            # # self.down_stream_out_fn = nn.Sequential(nn.Linear(nhead, num_tasks))
            # self.down_stream_out_fn = nn.Sequential(
            #     # nn.Linear(d_model, d_model // 2),
            #     # nn.GELU(),
            #     nn.Dropout(self.dropout),
            #     # nn.Dropout(self.dropout),
            #     nn.Linear(d_model, num_tasks),
            # )
            pass

        elif "prompt" in self.mode:
            self.prompt_fn = MolGNNLayers(d_model, nhead, dropout, mode, num_tasks, num_layers=4)

        else:
            raise ValueError('Unrecognized mode: {}'.format(self.mode))

        if self.mode in ['finetune_classification', 'finetune_DDI'] or 'classification' in self.mode:
            if self.loss_type == 'bce':
                self.down_stream_criterion = nn.BCEWithLogitsLoss()
            else:
                self.down_stream_criterion = nn.CrossEntropyLoss()

        elif 'regression' in self.mode:
            self.down_stream_criterion = nn.MSELoss()

        elif self.mode != 'pretrain':
            raise ValueError('Unrecognized mode: {}'.format(self.mode))

        self.test_recorder = None

        self.save_hyperparameters()

    def pre_forward_step(self, mol_dict):
        if self.mode == 'pretrain' or 'not_freeze' in self.mode:
            pred = self.forward(mol_dict)
        elif 'finetune' or 'prompt' in self.mode:
            graph_data = mol_dict['graph_data'] if 'graph_data' in mol_dict else None

            if 'empty' in self.mode:
                return self.forward(graph_data=graph_data), mol_dict

            if 'DDI' in self.mode or 'DDS' in self.mode:
                mol_dict_1, mol_dict_2 = mol_dict
                mol_dict = concat_padded_mol_dict(mol_dict_1, mol_dict_2)

            if self.prompt_mol:
                mol_dict = concat_prompt_mol_dict(mol_dict, self.prompt_mol)

            with torch.no_grad():
                raw_fea, edge_fea = self.emb(mol_dict['atom_fea'], mol_dict['bond_adj'], mol_dict['dist_adj'])
                n_atom = raw_fea.size(1)
                atom_fea = self.encoder(raw_fea, edge_fea)
                bond_prob = self.bond_fn(atom_fea, atom_fea)  # [batch_size, n_atom, n_atom, nhead]

                if graph_data is not None:
                    edge_fea = edge_fea.reshape(raw_fea.size(0), self.nhead, n_atom, n_atom).permute(0, 2, 3,
                                                                                                     1)  # [batch_size, n_atom, n_atom, nhead]
                    edge_fea += bond_prob  # 相加原因：edge_fea是原始的，bond_prob是预测的，两者相加，相当于将预测的信息加入到原始的edge_fea中
                    # 删去无效的边
                    bond_indices = edge_index = torch.where(mol_dict['bond_adj'] > 1, 1, 0).nonzero().t()
                    edge_fea = edge_fea[
                        bond_indices[0], bond_indices[1], bond_indices[2]]  # edge_fea: [batch_size, n_edge, nhead]
                    # 删去无效的原子
                    atom_indices = torch.where(mol_dict['atom_fea'][:, 0] > 0, 1, 0).nonzero().t()
                    atom_fea = atom_fea[atom_indices[0], atom_indices[1]]  # atom_fea: [batch_size, n_atom, d_model]
                    atom_fea += raw_fea[atom_indices[0], atom_indices[1]]
                else:
                    edge_fea = bond_prob.permute(0, 3, 1, 2).reshape(-1, n_atom,
                                                                     n_atom) + edge_fea  # edge_fea: [batch_size, n_atom, n_atom, nhead]
                    atom_fea += raw_fea
            pred = self.forward(atom_fea=atom_fea, edge_fea=edge_fea, graph_data=graph_data)

        else:
            raise ValueError('Unrecognized mode: {}'.format(self.mode))
        return pred, mol_dict

    def forward(self, mol_dict=None, atom_fea=None, bond_prob=None, edge_fea=None, graph_data=None):

        if self.mode in ['pretrain'] or 'not_freeze' in self.mode:
            raw_fea, edge_fea = self.emb(mol_dict['atom_fea'], mol_dict['bond_adj'], mol_dict['dist_adj'])
            atom_fea = self.encoder(raw_fea, edge_fea)
            bond_prob = self.bond_fn(atom_fea, atom_fea)

            if self.mode in ['pretrain']:
                atom_probs = []
                for fn in self.out_fn_atoms:
                    atom_probs.append(fn(atom_fea))
                bond_prob = self.out_fn_bond(bond_prob)
                return atom_probs, bond_prob
            else:
                atom_fea = self.down_stream_encoder(atom_fea, edge_fea)
                pred_prob = self.down_stream_out_fn(atom_fea.sum(dim=1))

        elif 'finetune' in self.mode:
            # atom_fea = self.down_stream_linear(atom_fea)
            if self.predict_state:
                atom_fea, intermedia = self.down_stream_encoder(atom_fea, edge_fea, need_weights=True)
            else:
                atom_fea = self.down_stream_encoder(atom_fea, edge_fea)
            pred_prob = self.down_stream_out_fn(atom_fea.sum(dim=1))


        elif 'prompt' in self.mode:
            pred_prob = self.prompt_fn(batch=graph_data, atom_bias=atom_fea, bond_bias=edge_fea)
        else:
            raise ValueError('Unrecognized mode: {}'.format(self.mode))

        if self.predict_state:
            return pred_prob, intermedia

        return pred_prob

    def training_step(self, batch, batch_idx):
        mol_dict, target = batch
        pred, mol_dict = self.pre_forward_step(mol_dict)
        loss = self.calc_loss_and_acc(pred, target, bond_adj=mol_dict['bond_adj'],
                                      prefix='train', is_eval=batch_idx % 100 == 0)
        return loss

    def validation_step(self, batch, batch_idx):
        mol_dict, target = batch
        pred, mol_dict = self.pre_forward_step(mol_dict)
        bast_point = self.calc_loss_and_acc(pred, target, bond_adj=mol_dict['bond_adj'], prefix='valid')
        return {
            "best_point": bast_point,
        }

    def validation_epoch_end(self, outputs):
        avg_performance = self._avg_dicts(outputs)
        # log_metric.append(avg_performance['best_point'])
        nni.report_intermediate_result(avg_performance['best_point'])
        self._log_dict(avg_performance)

    def test_step(self, batch, batch_idx):
        mol_dict, target = batch
        pred, mol_dict = self.pre_forward_step(mol_dict)
        # performance = self.calc_loss_and_acc(pred, target, bond_adj=mol_dict['bond_adj'], prefix='test')
        if self.test_recorder is None:
            self.test_recorder = {'pred': pred, 'target': target}
        else:
            self.test_recorder['pred'] = torch.cat([self.test_recorder['pred'], pred], dim=0)
            self.test_recorder['target'] = torch.cat([self.test_recorder['target'], target], dim=0)
        return self.test_recorder

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        self.predict_state = True
        mol_dict, target = batch
        pred, intermedia = self.pre_forward_step(mol_dict)
        return pred, intermedia, target

    def predict_epoch_end(self, result):
        pred_all, intermedia_all, target_all = [], [], []
        for pred, intermedia, target in result:
            pred_all.extend(pred.tolist())
            intermedia_all.extend(intermedia.tolist(dim=0))
            target_all.extend(target.tolist())
        return pred_all, intermedia_all, target_all

    def test_epoch_end(self, outputs):
        performance = self.calc_loss_and_acc(self.test_recorder['pred'], self.test_recorder['target'], prefix='test')
        print(performance)
        nni.report_intermediate_result(performance)
        self._log_dict(performance)
        test_recorder = self.test_recorder
        self.test_recorder = None
        return {"performance": performance, "recorder": test_recorder}

    def calc_loss_and_acc(self, pred, target, bond_adj=None, prefix="train", is_eval=True):
        performance = {}
        if self.mode in ['pretrain', 'finetune_reaction']:
            atom_pred, bond_pred = pred
            bond_mask = torch.where(bond_adj != 0, True, False)
            atom_mask = torch.where(target['atom'][:, 0, :] > -1000, True, False)

            # print(target['bond'][bond_mask].size(), target['bond'][bond_mask][:100])
            cur_pred = bond_pred[bond_mask.unsqueeze(-1).repeat(1, 1, 1, 13)].reshape(-1, 13)
            cur_target = (target['bond'][bond_mask] * 2 + 6).long()
            loss_list = [self.criterion(cur_pred, cur_target)]

            if is_eval:
                acc = TMF.multiclass_accuracy(cur_pred, cur_target, num_classes=13)
                auc = TMF.multiclass_auroc(cur_pred, cur_target, num_classes=13)
                auprc = TMF.multiclass_auprc(cur_pred, cur_target, num_classes=13)
                self.log(f'pretrain_{prefix}_bond_acc', acc, on_step=True, prog_bar=True)
                self.log(f'pretrain_{prefix}_bond_auc', auc, on_step=True, prog_bar=True)
                self.log(f'pretrain_{prefix}_bond_auprc', auprc, on_step=True, prog_bar=True)

            task_name = ['charge', 'num_H', 'chiral']

            for fea_idx in range(NUM_ATOM_FEAT - 2):
                cur_pred = atom_pred[fea_idx][atom_mask.unsqueeze(-1).repeat(1, 1, 50)].view(-1, 50)
                cur_target = (target['atom'][:, fea_idx, :][atom_mask] + 25).long()
                # print(cur_target[:10])
                loss_list.append(self.criterion(cur_pred, cur_target))
                if is_eval:
                    acc = TMF.multiclass_accuracy(cur_pred, cur_target, num_classes=50)
                    f1 = TMF.multiclass_f1_score(cur_pred, cur_target, num_classes=50)
                    auc = TMF.multiclass_auroc(cur_pred, cur_target, num_classes=50)
                    self.log(f'pretrain_{prefix}_{task_name[fea_idx]}_acc', acc, on_epoch=True, prog_bar=True)
                    self.log(f'pretrain_{prefix}_{task_name[fea_idx]}_f1', f1, on_epoch=True, prog_bar=False)
                    self.log(f'pretrain_{prefix}_{task_name[fea_idx]}_auc', auc, on_epoch=True, prog_bar=False)

            # print(loss_list)

            loss = self.calc_mt_loss(loss_list)
            self.log("pretrain_" + prefix + "loss", loss, prog_bar=True, logger=True)
            self.log("pretrain_" + prefix + "bond_loss", loss_list[0].item(), logger=True)
            self.log("pretrain_" + prefix + "charge_loss", loss_list[1].item(), logger=True)
            self.log("pretrain_" + prefix + "num_h_loss", loss_list[2].item(), logger=True)
            self.log("pretrain_" + prefix + "chiral_loss", loss_list[3].item(), logger=True)

            if prefix == 'test':
                performance['bond'] = loss_list[0].item()
                performance['charge'] = loss_list[1].item()
                performance['num_h'] = loss_list[2].item()
                performance['chiral'] = loss_list[3].item()
                performance['batch_size'] = target.size(0)

        elif self.mode == "finetune_DDI" or "classification" in self.mode:

            if self.loss_type == 'bce':
                mask = torch.where(target != -1, True, False)
                if prefix != 'test':
                    loss = self.down_stream_criterion(pred[mask].view(-1), target[mask].view(-1).float())

                cur_pred = pred.sigmoid()
                # print(pred[mask][0], target[mask][0])
                acc = TMF.binary_accuracy(cur_pred[mask].view(-1), target[mask].view(-1).long())
                # prc = TMF.binary_precision(cur_pred, target[mask].view(-1).long())
                # rec = TMF.binary_recall(cur_pred, target[mask].view(-1).long())
                # auc = TMF.binary_auroc(cur_pred[mask].view(-1), target[mask].view(-1).long())
                auc = 0
                for i in range(self.num_tasks):
                    auc += TMF.binary_auroc(cur_pred[:, i][mask[:, i]].view(-1),
                                            target[:, i][mask[:, i]].view(-1).long())
                auc /= self.num_tasks
                if prefix != 'test':
                    self.log("finetune_cls_" + prefix + "_bce_loss", loss, prog_bar=True, logger=True)
                    self.log("finetune_cls_" + prefix + "_acc", acc.item(), prog_bar=True, logger=True)
                    # self.log("finetune_cls_" + prefix + "_prc", prc.item(), prog_bar=False, logger=True)
                    # self.log("finetune_cls_" + prefix + "_rec", rec.item(), prog_bar=False, logger=True)
                    self.log("finetune_cls_" + prefix + "_auc", auc.item(), prog_bar=True, logger=True)
                performance['acc'] = acc.item()
                performance['auc'] = auc.item()
                performance['batch_size'] = mask.sum().item()
                performance['best_point'] = performance['auc']

            else:
                cur_pred = pred.view(-1, self.num_tasks)
                cur_target = target.view(-1).long()
                loss = self.down_stream_criterion(cur_pred, cur_target)
                # print(pred[mask][0], target[mask][0])
                # print(cur_target.max(), self.num_tasks)
                cur_pred = cur_pred.softmax(dim=-1)
                acc = TMF.multiclass_accuracy(cur_pred, cur_target, num_classes=self.num_tasks)
                f1 = TMF.multiclass_f1_score(cur_pred, cur_target, num_classes=self.num_tasks)
                # auc = TMF.multiclass_auroc(cur_pred, cur_target, num_classes=self.num_tasks)
                kappa_score = cohen_kappa_score(cur_pred.argmax(dim=-1).numpy(force=True), cur_target.numpy(force=True))
                if prefix != 'test':
                    self.log("finetune_cls_" + prefix + "_mt_loss", loss, prog_bar=True, logger=True)
                    self.log("finetune_cls_" + prefix + "_mt_acc", acc.item(), prog_bar=True, logger=True)
                    self.log("finetune_cls_" + prefix + "_mt_f1", f1.item(), prog_bar=False, logger=True)
                    self.log("finetune_cls_" + prefix + "_mt_kappa_score", kappa_score, prog_bar=True, logger=True)
                performance['acc'] = acc.item()
                performance['f1'] = f1.item()
                performance['kappa_score'] = kappa_score
                performance['batch_size'] = target.size(0)
                performance['best_point'] = performance['f1']


        elif 'regression' in self.mode:
            loss = self.down_stream_criterion(pred.view(-1).float(), target.view(-1).float())
            rmse = loss.sqrt().float()

            if prefix != 'test':
                self.log("finetune_reg_" + prefix + "_mse_loss", loss, prog_bar=True, logger=True)
                self.log("finetune_reg_" + prefix + "_rmse_loss", rmse, prog_bar=True, logger=True)

            performance['rmse'] = rmse.item()
            performance['best_point'] = -performance['rmse']

        else:
            raise ValueError('Unrecognized mode: {}'.format(self.mode))

        if prefix == 'train':
            return loss
        elif prefix == 'valid':
            return performance['best_point'] if "best_point" in performance.keys() else -loss
        else:
            return performance

    @staticmethod
    def _avg_dicts(colls):
        complete_dict = {key: [] for key, val in colls[0].items() if key != 'batch_size'}
        if "batch_size" in colls[0].keys():
            batch_sizes = [coll['batch_size'] for coll in colls]
        else:
            batch_sizes = [1] * len(colls)

        for col_index, coll in enumerate(colls):
            for key in complete_dict.keys():
                complete_dict[key].append(coll[key])
                complete_dict[key][col_index] *= batch_sizes[col_index]

        avg_dict = {key: sum(l) / sum(batch_sizes) for key, l in complete_dict.items()}
        return avg_dict

    def _log_dict(self, coll):
        for key, val in coll.items():
            self.log(key, val, sync_dist=True)

    def calc_mt_loss(self, loss_list):
        loss_list = torch.stack(loss_list)
        # if not self.use_adaptive_multi_task or self.num_shared_layer == 0:
        #     return loss_list.sum()

        if self.loss_init is None:
            if self.training:
                self.loss_init = util.LossRecorderList(loss_list=loss_list.detach().tolist(), device=loss_list.device)
                self.loss_last2 = loss_list.detach()
                loss_t = (loss_list / self.loss_init.get_mean()).mean()
            else:
                loss_t = (loss_list / loss_list.detach()).mean()

        elif self.loss_last is None:
            if self.training:
                self.loss_last = loss_list.detach()
                self.loss_init.update(loss_list.detach().tolist())
                loss_t = (loss_list / self.loss_init.get_mean()).mean()
            else:
                loss_t = (loss_list / loss_list.detach()).mean()
        else:
            w = F.softmax(self.loss_last / self.loss_last2, dim=-1).detach()
            loss_t = (loss_list / self.loss_init.get_mean() * w).sum()

            if self.training:
                self.loss_init.update(loss_list.detach().tolist())
                self.loss_last2 = self.loss_last
                self.loss_last = loss_list.detach()
        return loss_t

    # def calc_mt_loss(self, loss_list):
    #     loss_list = torch.stack(loss_list)
    #     if not self.use_adaptive_multi_task:
    #         return loss_list.sum()
    #
    #     if self.cur_loss_step == 0:
    #         if self.training:
    #             self.loss_init[:, 0] = loss_list.detach()
    #             self.loss_last2[:, 0] = loss_list.detach()
    #             self.cur_loss_step += 1
    #             loss_t = (loss_list / self.loss_init[:, 0]).mean()
    #         else:
    #             loss_t = (loss_list / loss_list.detach()).mean()
    #
    #     elif self.cur_loss_step == 1:
    #         if self.training:
    #             self.loss_last[:, 0] = loss_list.detach()
    #             self.loss_init[:, 1] = loss_list.detach()
    #             self.cur_loss_step += 1
    #             loss_t = (loss_list / self.loss_init[:, :2].mean(dim=-1)).mean()
    #         else:
    #             loss_t = (loss_list / loss_list.detach()).mean()
    #     else:
    #         cur_loss_init = self.loss_init[:, :self.cur_loss_step].mean(dim=-1)
    #         cur_loss_last = self.loss_last[:, :self.cur_loss_step - 1].mean(dim=-1)
    #         cur_loss_last2 = self.loss_last2[:, :self.cur_loss_step - 1].mean(dim=-1)
    #         w = F.softmax(cur_loss_last / cur_loss_last2, dim=-1).detach()
    #         loss_t = (loss_list / cur_loss_init * w).sum()
    #
    #         if self.training:
    #             cur_init_idx = self.cur_loss_step.item() % self.batch_considered
    #             self.loss_init[:, cur_init_idx] = loss_list.detach()
    #
    #             cur_loss_last2_step = (self.cur_loss_step.item() - 1) % (self.batch_considered // 10)
    #             self.loss_last2[:, cur_loss_last2_step] = self.loss_last[:, cur_loss_last2_step - 1]
    #             self.loss_last[:, cur_loss_last2_step] = loss_list.detach()
    #             self.cur_loss_step += 1
    #     return loss_t

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--seed', type=int, default=123)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--d_model', type=int, default=512)
        parser.add_argument('--nhead', type=int, default=16)
        parser.add_argument('--num_encoder_layer', type=int, default=2)
        parser.add_argument('--dim_feedforward', type=int, default=512)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--batch_second', default=False, action='store_true')
        parser.add_argument('--known_rxn_cnt', default=False, action='store_true')
        parser.add_argument('--norm_first', default=False, action='store_true')
        parser.add_argument('--activation', type=str, default='gelu')
        parser.add_argument("--warmup_updates", type=float, default=6000)
        parser.add_argument("--tot_updates", type=float, default=200000)
        parser.add_argument("--peak_lr", type=float, default=2e-4)
        parser.add_argument("--end_lr", type=float, default=1e-9)
        parser.add_argument('--weight_decay', type=float, default=1e-2)
        parser.add_argument('--max_single_hop', type=int, default=4)
        parser.add_argument('--not_use_dist_adj', default=False, action='store_true')
        # parser.add_argument('--not_use_contrastive', default=False, action='store_true')
        parser.add_argument('--prompt_mol', default='', type=str)
        parser.add_argument('--n_downstream_layer', default=3, type=int)
        return parser

    def configure_optimizers(self):
        if self.mode in ['pretrain'] or 'not_freeze' in self.mode:
            target_params = self.parameters()
        elif 'finetune' in self.mode:
            target_params = chain(
                self.down_stream_encoder.parameters(),
                self.down_stream_out_fn.parameters(), )
        elif 'prompt' in self.mode:
            target_params = self.prompt_fn.parameters()
        else:
            raise NotImplementedError
        #
        # if 'DDI' in self.mode:
        #     target_params = chain(target_params, self.down_stream_bond_fn.parameters())

        optimizer = torch.optim.AdamW(
            target_params, lr=self.peak_lr, weight_decay=self.weight_decay
        )
        lr_scheduler = {
            "scheduler": PolynomialDecayLR(
                optimizer,
                warmup_updates=self.warmup_updates,
                tot_updates=self.tot_updates,
                lr=self.peak_lr,
                end_lr=self.end_lr,
                power=1.0,
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]
