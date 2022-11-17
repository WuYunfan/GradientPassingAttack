from abc import ABC

from attacker.basic_attacker import BasicAttacker
import torch.nn as nn
from model import LightGCN, BasicModel
from utils import get_sparse_tensor, generate_adj_mat, AverageMeter, ce_loss, mse_loss, bce_loss, topk_loss
import scipy.sparse as sp
import numpy as np
import torch
from torch.nn.init import normal_
import dgl
import copy
from torch.utils.data import TensorDataset, DataLoader
from trainer import BasicTrainer
import torch.nn.functional as F
import sys
from torch.autograd import Function
import gc
from torch.optim.lr_scheduler import StepLR
from attacker.wrmf_sgd_attacker import WRMFSGD
import higher
from torch.optim import Adam, SGD


class TorchSparseMat:
    def __init__(self, row, col, shape, device):
        self.shape = shape
        self.device = device
        row = torch.tensor(row, dtype=torch.int64, device=device)
        col = torch.tensor(col, dtype=torch.int64, device=device)
        self.g = dgl.graph((col, row), num_nodes=max(shape), device=device)
        self.inv_g = dgl.graph((row, col), num_nodes=max(shape), device=device)
        self.n_non_zeros = self.g.num_edges()

    def spmm(self, r_mat, value_tensor, norm=None):
        values = torch.ones([self.n_non_zeros - value_tensor.shape[0]], dtype=torch.float32, device=self.device)
        values = torch.cat([values, value_tensor], dim=0)

        padding_tensor = torch.empty([max(self.shape) - r_mat.shape[0], r_mat.shape[1]],
                                     dtype=torch.float32, device=self.device)
        padded_r_mat = torch.cat([r_mat, padding_tensor], dim=0)

        x = dgl.ops.gspmm(self.g, 'mul', 'sum', lhs_data=padded_r_mat, rhs_data=values)
        if norm is not None:
            row_sum = dgl.ops.gspmm(self.g, 'copy_rhs', 'sum', lhs_data=None, rhs_data=values)
            if norm == 'left':
                x = x / (row_sum[:, None] + 1.e-8)
            if norm == 'both':
                col_sum = dgl.ops.gspmm(self.inv_g, 'copy_rhs', 'sum', lhs_data=None, rhs_data=values)
                x = x / (torch.pow(row_sum[:, None], 0.5) + 1.e-8) / (torch.pow(col_sum[:, None], 0.5) + 1.e-8)
        return x[:self.shape[0], :]


class SurrogateEPRA4MF(BasicModel):
    def __init__(self, model_config):
        super(SurrogateEPRA4MF, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.embedding = nn.Embedding(self.n_users + self.n_items, self.embedding_size)
        normal_(self.embedding.weight, std=0.1)
        self.to(device=self.device)

    def predict(self, users):
        user_e = self.embedding.weight[users, :]
        scores = torch.mm(user_e, self.embedding.weight[-self.n_items:, :].t())
        return scores


class EPRA4MF(BasicAttacker):
    def __init__(self, attacker_config):
        super(EPRA4MF, self).__init__(attacker_config)
        self.surrogate_config = attacker_config['surrogate_config']
        self.surrogate_config['device'] = self.device
        self.surrogate_config['dataset'] = self.dataset

        self.adv_epochs = attacker_config['adv_epochs']
        self.train_epochs = attacker_config['train_epochs']
        self.unroll_steps = attacker_config['unroll_steps']
        self.initial_lr = attacker_config['lr']
        self.momentum = attacker_config['momentum']
        self.propagation_order = attacker_config['propagation_order']
        self.alpha = attacker_config['alpha']
        self.kappa = attacker_config['kappa']

        self.data_mat = sp.coo_matrix((np.ones((len(self.dataset.train_array),)), np.array(self.dataset.train_array).T),
                                      shape=(self.n_users, self.n_items), dtype=np.float32).tocsr()
        self.fake_tensor = self.init_fake_tensor()
        self.adv_opt = SGD([self.fake_tensor], lr=self.initial_lr, momentum=self.momentum)
        self.scheduler = StepLR(self.adv_opt, step_size=self.adv_epochs / 3, gamma=0.1)

        self.dataset.train_data += [[]] * self.n_fakes
        self.dataset.val_data += [[]] * self.n_fakes
        self.dataset.n_users += self.n_fakes

        poisoned_data_mat = torch.tensor(self.data_mat.toarray(), dtype=torch.float32, device=self.device)
        self.poisoned_data_mat = torch.cat([poisoned_data_mat, self.fake_tensor], dim=0)
        test_users = TensorDataset(torch.arange(self.n_users + self.n_fakes, dtype=torch.int64, device=self.device))
        self.user_loader = DataLoader(test_users, batch_size=self.surrogate_config['batch_size'],
                                      shuffle=True)
        target_users = [user for user in range(self.n_users) if self.target_item not in self.dataset.train_data[user]]
        self.target_users = torch.tensor(target_users, dtype=torch.int64, device=self.device)

    def init_fake_tensor(self):
        return WRMFSGD.init_fake_tensor(self)

    def project_fake_tensor(self):
        WRMFSGD.project_fake_tensor(self)

    def retrain_surrogate(self):
        surrogate_model = getattr(sys.modules[__name__], self.surrogate_config['name'])
        surrogate_model = surrogate_model(self.surrogate_config)
        surrogate_model.train()
        train_opt = Adam(surrogate_model.parameters(), lr=self.surrogate_config['lr'],
                         weight_decay=self.surrogate_config['l2_reg'])

        for _ in range(self.train_epochs - self.unroll_steps):
            for users in self.user_loader:
                users = users[0]
                batch_data = self.poisoned_data_mat[users, :]
                scores = surrogate_model.forward(users)
                loss = bce_loss(batch_data, scores, self.alpha)
                train_opt.zero_grad()
                loss.backward()
                train_opt.step()

        with higher.innerloop_ctx(surrogate_model, train_opt) as (fmodel, diffopt):
            fmodel.train()
            for _ in range(self.unroll_steps):
                for users in self.user_loader:
                    users = users[0]
                    batch_data = self.poisoned_data_mat[users, :]
                    scores = fmodel.forward(users)
                    loss = bce_loss(batch_data, scores, self.alpha)
                    diffopt.step(loss)

            fmodel.eval()
            scores = fmodel.forward(self.target_users)
            adv_loss = topk_loss(scores, self.target_item, self.topk, self.kappa)
            _, topk_items = scores.topk(self.topk, dim=1)
            hr = torch.eq(topk_items, self.target_item).float().sum(dim=1).mean()
            adv_grads = torch.autograd.grad(adv_loss, self.fake_tensor)[0]
        gc.collect()
        return adv_loss.item(), hr.item(), adv_grads

    def generate_fake_users(self, verbose=True, writer=None):
        WRMFSGD.generate_fake_users(self, verbose, writer)
        self.dataset.train_data = self.dataset.train_data[:-self.n_fakes]
        self.dataset.val_data = self.dataset.val_data[:-self.n_fakes]
        self.dataset.n_users -= self.n_fakes

