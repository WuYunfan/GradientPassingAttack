from attacker.basic_attacker import BasicAttacker
import torch.nn as nn
from model import LightGCN, BasicModel
from utils import get_sparse_tensor, generate_daj_mat, AverageMeter, ce_loss
import scipy.sparse as sp
import numpy as np
import torch
from torch.nn.init import normal_
import dgl
import copy
from torch.utils.data import TensorDataset, DataLoader
from trainer import BasicTrainer
from utils import AverageMeter
import torch.nn.functional as F
import sys
from torch.autograd import Function
from torch.optim import SGD
import gc
from torch.optim.lr_scheduler import StepLR
from attacker.wrmf_sgd_attacker import WRMFSGD
import higher


class TorchSparseMat:
    def __init__(self, row, col, shape, device):
        self.shape = shape
        self.device = device
        row = torch.tensor(row, dtype=torch.int64, device=device)
        col = torch.tensor(col, dtype=torch.int64, device=device)
        self.g = dgl.graph((col, row), num_nodes=max(shape), device=device)
        self.inv_g = dgl.graph((row, col), num_nodes=max(shape), device=device)

    def spmm(self, r_mat, value_tensor, norm=None):
        n_non_zeros = self.g.num_edges()
        values = torch.ones([n_non_zeros - value_tensor.shape[0]], dtype=torch.float32, device=self.device)
        values = torch.cat([values, value_tensor], dim=0)

        padding_tensor = torch.empty([max(self.shape) - r_mat.shape[0], r_mat.shape[1]],
                                     dtype=torch.float32, device=self.device)
        padded_r_mat = torch.cat([r_mat, padding_tensor], dim=0)

        x = dgl.ops.gspmm(self.g, 'mul', 'sum', lhs_data=padded_r_mat, rhs_data=values)
        if norm is not None:
            row_sum = dgl.ops.gspmm(self.g, 'copy_rhs', 'sum', lhs_data=None, rhs_data=values)
            if norm == 'left':
                x = x / (row_sum[:, None] + 1.e-5)
            if norm == 'both':
                col_sum = dgl.ops.gspmm(self.inv_g, 'copy_rhs', 'sum', lhs_data=None, rhs_data=values)
                x = x / (torch.pow(row_sum[:, None], 0.5) + 1.e-5) / (torch.pow(col_sum[:, None], 0.5) + 1.e-5)
        return x[:self.shape[0], :]


class IGCN(BasicModel):
    def __init__(self, model_config):
        super(IGCN, self).__init__(model_config)
        self.n_fake_users = model_config['n_fake_users']
        self.n_norm_users = self.n_users - self.n_fake_users
        self.embedding_size = model_config['embedding_size']
        self.n_layers = model_config['n_layers']

        self.adj_mat = self.generate_graph(model_config['dataset'])
        self.feat_mat = self.generate_feat(model_config['dataset'])

        self.embedding = nn.Embedding(self.feat_mat.shape[1], self.embedding_size)
        normal_(self.embedding.weight, std=0.1)
        self.to(device=self.device)

    def generate_graph(self, dataset):
        adj_mat = generate_daj_mat(dataset).tocoo()
        row, col = adj_mat.row, adj_mat.col
        fake_row = np.arange(self.n_fake_users, dtype=np.int64) + self.n_users - self.n_fake_users
        fake_row = fake_row[:, None].repeat(self.n_items, axis=1).flatten()
        fake_col = np.arange(self.n_items, dtype=np.int64) + self.n_users
        fake_col = fake_col.repeat(self.n_fake_users, axis=0)
        row = np.concatenate([row, fake_row, fake_col])
        col = np.concatenate([col, fake_col, fake_row])
        adj_mat = TorchSparseMat(row, col, (self.n_users + self.n_items,
                                            self.n_users + self.n_items), self.device)
        return adj_mat

    def generate_feat(self, dataset):
        indices = []
        for user, item in dataset.train_array:
            indices.append([user, self.n_norm_users + item])
            indices.append([self.n_users + item, user])
        for user in range(self.n_users):
            indices.append([user, self.n_norm_users + self.n_items])
        for item in range(self.n_items):
            indices.append([self.n_users + item, self.n_norm_users + self.n_items + 1])
        indices = np.array(indices)
        row, col = indices[:, 0], indices[:, 1]
        fake_row = np.arange(self.n_fake_users, dtype=np.int64) + self.n_users - self.n_fake_users
        fake_row = fake_row[:, None].repeat(self.n_items, axis=1).flatten()
        fake_col = np.arange(self.n_items, dtype=np.int64) + self.n_norm_users
        fake_col = fake_col.repeat(self.n_fake_users, axis=0)
        row = np.concatenate([row, fake_row])
        col = np.concatenate([col, fake_col])
        feat_mat = TorchSparseMat(row, col, (self.n_users + self.n_items,
                                             self.n_norm_users + self.n_items), self.device)
        return feat_mat

    def inductive_rep_layer(self, fake_tensor):
        x = self.feat_mat.spmm(self.embedding.weight, fake_tensor.flatten(), norm='left')
        return x

    def get_rep(self, fake_tensor=None):
        if fake_tensor is None:
            fake_tensor = torch.zeros([self.n_fake_users, self.n_items], dtype=torch.float32, device=self.device)

        representations = self.inductive_rep_layer(fake_tensor)
        all_layer_rep = [representations]
        for _ in range(self.n_layers):
            representations = self.adj_mat.spmm(representations, fake_tensor.flatten().repeat(2), norm='both')
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def predict(self, users):
        return LightGCN.predict(self, users)


class IMF(IGCN):
    def __init__(self, model_config):
        super(IMF, self).__init__(model_config)

    def get_rep(self, fake_tensor=None):
        if fake_tensor is None:
            fake_tensor = torch.zeros([self.n_fake_users, self.n_items], dtype=torch.float32, device=self.device)
        representations = self.inductive_rep_layer(fake_tensor)
        return representations


class IGCNTrainer(BasicTrainer):
    def __init__(self, trainer_config):
        super(IGCNTrainer, self).__init__(trainer_config)

        self.dataloader = DataLoader(self.dataset, batch_size=trainer_config['batch_size'],
                                     num_workers=trainer_config['dataloader_num_workers'],
                                     persistent_workers=True)
        self.initialize_optimizer()
        self.l2_reg = trainer_config['l2_reg']
        self.aux_reg = trainer_config['aux_reg']

    def train_one_epoch(self):
        losses = AverageMeter()
        for batch_data in self.dataloader:
            rep = self.model.get_rep()
            inputs = batch_data.to(device=self.device, dtype=torch.int64)

            users, pos_items = inputs[:, 0, 0], inputs[:, 0, 1]
            users_r = rep[users, :]
            pos_items_r = rep[self.model.n_users + pos_items, :]
            bce_loss_p = -F.softplus(torch.sum(users_r * pos_items_r, dim=1))
            l2_norm_sq_p = torch.norm(users_r, p=2, dim=1) ** 2 + torch.norm(pos_items_r, p=2, dim=1) ** 2

            users_e = self.model.embedding(users)
            aux_loss = -F.softplus(torch.sum(users_r * users_e, dim=1)).mean()

            inputs = inputs.reshape(-1, 3)
            users, neg_items = inputs[:, 0], inputs[:, 2]
            users_r = rep[users, :]
            neg_items_r = rep[self.model.n_users + neg_items, :]
            bce_loss_n = F.softplus(torch.sum(users_r * neg_items_r, dim=1))
            l2_norm_sq_n = torch.norm(neg_items_r, p=2, dim=1) ** 2

            l2_norm_sq = torch.cat([l2_norm_sq_p, l2_norm_sq_n], dim=0)
            bce_loss = torch.cat([bce_loss_p, bce_loss_n], dim=0).mean()
            reg_loss = self.l2_reg * l2_norm_sq.mean() + self.aux_reg * aux_loss
            loss = bce_loss + reg_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.update(loss.item(), l2_norm_sq.shape[0])
        return losses.avg


class ParameterPropagation(Function):
    @staticmethod
    def forward(ctx, rep, mat, order, fake_tensor):
        ctx.order = order
        ctx.mat = mat
        ctx.save_for_backward(fake_tensor)
        return rep

    @staticmethod
    def backward(ctx, grad_out):
        order = ctx.order
        mat = ctx.mat
        fake_tensor = ctx.saved_tensors[0]
        grad = grad_out
        for _ in range(order):
            grad = mat.spmm(grad, fake_tensor.flatten()) + grad + grad_out
        return grad, None, None, None


class ERAP4(BasicAttacker):
    def __init__(self, attacker_config):
        super(ERAP4, self).__init__(attacker_config)
        self.surrogate_model_config = attacker_config['surrogate_model_config']
        self.surrogate_model_config['device'] = self.device
        self.surrogate_model_config['dataset'] = self.dataset
        self.surrogate_model_config['n_fake_users'] = self.n_fakes

        self.surrogate_trainer_config = attacker_config['surrogate_trainer_config']
        self.surrogate_trainer_config['device'] = self.device
        self.surrogate_trainer_config['dataset'] = self.dataset

        self.propagation_order = attacker_config['propagation_order']
        self.retraining_lr = attacker_config['re_lr']
        self.initial_lr = attacker_config['lr']
        self.momentum = attacker_config['momentum']
        self.adv_epochs = attacker_config['adv_epochs']
        self.topk = attacker_config['topk']

        self.data_mat = sp.coo_matrix((np.ones((len(self.dataset.train_array),)), np.array(self.dataset.train_array).T),
                                      shape=(self.n_users, self.n_items), dtype=np.float32).tocsr()
        self.fake_tensor = self.init_fake_tensor()
        self.adv_opt = SGD([self.fake_tensor], lr=self.initial_lr, momentum=self.momentum)
        self.scheduler = StepLR(self.adv_opt, step_size=self.adv_epochs / 3, gamma=0.1)

        self.dataset.train_data += [[]] * self.n_fakes
        self.dataset.val_data += [[]] * self.n_fakes
        self.dataset.n_users += self.n_fakes

        self.surrogate_model = getattr(sys.modules[__name__], self.surrogate_model_config['name'])
        self.surrogate_model = self.surrogate_model(self.surrogate_model_config)
        self.surrogate_trainer_config['dataset'] = self.dataset
        self.surrogate_trainer_config['model'] = self.surrogate_model
        self.surrogate_trainer = IGCNTrainer(self.surrogate_trainer_config)
        self.surrogate_trainer.train()
        self.retrain_opt = SGD(self.surrogate_model.parameters(), lr=self.retraining_lr)
        test_user = TensorDataset(torch.arange(self.n_users, dtype=torch.int64, device=self.device))
        self.user_loader = DataLoader(test_user, batch_size=self.surrogate_trainer_config['batch_size'])

    def init_fake_tensor(self):
        return WRMFSGD.init_fake_tensor(self)

    def project_fake_tensor(self):
        WRMFSGD.project_fake_tensor(self)

    def retrain_surrogate(self):
        with higher.innerloop_ctx(self.surrogate_model, self.retrain_opt) as (fmodel, diffopt):
            fmodel.eval()
            rep = fmodel.get_rep(self.fake_tensor)
            rep = ParameterPropagation.apply(rep, fmodel.adj_mat, self.propagation_order, self.fake_tensor)
            users_r = rep[self.n_users:self.n_users + self.n_fakes, :]
            all_items_r = rep[self.n_users + self.n_fakes:, :]
            scores = F.softplus(torch.mm(users_r, all_items_r.t()))
            loss = (scores - 2 * scores * self.fake_tensor).mean()
            diffopt.step(loss)

            scores = []
            for users in self.user_loader:
                users = users[0]
                scores.append(fmodel.predict(users))
            scores = torch.cat(scores, dim=0)
            adv_loss = ce_loss(scores, self.target_item)
            _, topk_items = scores.topk(self.topk, dim=1)
            hr = torch.eq(topk_items, self.target_item).float().sum(dim=1).mean()
            adv_grads = torch.autograd.grad(adv_loss, self.fake_tensor)[0]
        return adv_loss.item(), hr.item(), adv_grads

    def generate_fake_users(self, verbose=True, writer=None):
        WRMFSGD.generate_fake_users(self, verbose, writer)
        self.dataset.train_data = self.dataset.train_data[:-self.n_fakes]
        self.dataset.val_data = self.dataset.val_data[:-self.n_fakes]
        self.dataset.n_users -= self.n_fakes
