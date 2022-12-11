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
            grad = mat.spmm(grad, fake_tensor.flatten().repeat(2)) + grad + grad_out
        return grad, None, None, None


class SurrogateERAP4MF(BasicModel):
    def __init__(self, model_config):
        super(SurrogateERAP4MF, self).__init__(model_config)
        self.n_fake_users = model_config['n_fake_users']
        self.embedding_size = model_config['embedding_size']
        self.propagation_order = model_config['propagation_order']
        self.embedding = nn.Embedding(self.n_users + self.n_items, self.embedding_size)
        self.adj_mat = self.generate_graph(model_config['dataset'])
        normal_(self.embedding.weight, std=0.1)
        self.to(device=self.device)

    def predict(self, users, fake_tensor=None):
        if fake_tensor is None:
            embeddings = self.embedding.weight
        else:
            embeddings = ParameterPropagation.apply(self.embedding.weight, self.adj_mat,
                                                    self.propagation_order, fake_tensor)
        user_e = embeddings[users, :]
        scores = torch.mm(user_e, embeddings[-self.n_items:, :].t())
        return scores

    def generate_graph(self, dataset):
        adj_mat = generate_adj_mat(dataset).tocoo()
        row, col = adj_mat.row, adj_mat.col
        fake_row = np.arange(self.n_fake_users, dtype=np.int64) + self.n_users - self.n_fake_users
        fake_row = fake_row[:, None].repeat(self.n_items, axis=1).flatten()
        fake_col = np.arange(self.n_items, dtype=np.int64) + self.n_users
        fake_col = fake_col[None, :].repeat(self.n_fake_users, axis=0).flatten()
        row = np.concatenate([row, fake_row, fake_col])
        col = np.concatenate([col, fake_col, fake_row])
        adj_mat = TorchSparseMat(row, col, (self.n_users + self.n_items,
                                            self.n_users + self.n_items), self.device)
        return adj_mat


class EPAR4Trainer(BasicTrainer):
    def __init__(self, trainer_config):
        super(EPAR4Trainer, self).__init__(trainer_config)

        test_users = TensorDataset(torch.arange(self.dataset.n_users, dtype=torch.int64, device=self.device))
        self.user_loader = DataLoader(test_users, batch_size=self.config['test_batch_size'], shuffle=True)
        self.poisoned_data_mat = self.config['poisoned_data_mat']
        self.initialize_optimizer()
        self.l2_reg = trainer_config['l2_reg']
        self.weight = trainer_config['weight']

    def train_one_epoch(self):
        losses = AverageMeter()
        for users in self.user_loader:
            users = users[0]
            batch_data = self.poisoned_data_mat[users, :]
            scores = self.model.predict(users)
            loss = mse_loss(batch_data, scores, self.weight)
            loss += self.l2_reg * torch.norm(scores, p=2) ** 2 / (users.shape[0] * self.dataset.n_items)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.update(loss.item(), users.shape[0] * self.dataset.n_items)
        return losses.avg


class ERAP4(BasicAttacker):
    def __init__(self, attacker_config):
        super(ERAP4, self).__init__(attacker_config)
        self.surrogate_model_config = attacker_config['surrogate_model_config']
        self.surrogate_model_config['device'] = self.device
        self.surrogate_model_config['dataset'] = self.dataset
        self.surrogate_model_config['n_fake_users'] = self.n_fakes
        self.surrogate_model_config['verbose'] = False

        self.surrogate_trainer_config = attacker_config['surrogate_trainer_config']
        self.surrogate_trainer_config['device'] = self.device
        self.surrogate_trainer_config['dataset'] = self.dataset
        self.surrogate_trainer_config['topks'] = [self.topk]

        self.adv_epochs = attacker_config['adv_epochs']
        self.unroll_steps = attacker_config['unroll_steps']
        self.initial_lr = attacker_config['lr']
        self.momentum = attacker_config['momentum']
        self.kappa = torch.tensor(attacker_config['kappa'], dtype=torch.float32, device=self.device)

        self.data_mat = sp.coo_matrix((np.ones((len(self.dataset.train_array),)), np.array(self.dataset.train_array).T),
                                      shape=(self.n_users, self.n_items), dtype=np.float32).tocsr()
        self.fake_tensor = self.init_fake_tensor()
        self.adv_opt = SGD([self.fake_tensor], lr=self.initial_lr, momentum=self.momentum)
        self.scheduler = StepLR(self.adv_opt, step_size=self.adv_epochs / 3, gamma=0.1)

        self.dataset.train_data += [[]] * self.n_fakes
        self.dataset.val_data += [[]] * self.n_fakes
        self.dataset.n_users += self.n_fakes

        target_users = [user for user in range(self.n_users) if self.target_item not in self.dataset.train_data[user]]
        self.target_users = torch.tensor(target_users, dtype=torch.int64, device=self.device)

        self.surrogate_model = getattr(sys.modules[__name__], self.surrogate_model_config['name'])
        self.surrogate_model = self.surrogate_model(self.surrogate_model_config)

        self.data_tensor = torch.tensor(self.data_mat.toarray(), dtype=torch.float32, device=self.device)
        poisoned_data_mat = torch.cat([self.data_tensor, torch.zeros_like(self.fake_tensor)], dim=0)
        self.surrogate_trainer_config['model'] = self.surrogate_model
        self.surrogate_trainer_config['poisoned_data_mat'] = poisoned_data_mat
        self.surrogate_trainer = EPAR4Trainer(self.surrogate_trainer_config)
        ckpt_path = self.surrogate_trainer_config.get('ckpt_path', None)
        if ckpt_path is None:
            self.surrogate_trainer.train()
        else:
            self.surrogate_model.load(ckpt_path)

    def init_fake_tensor(self):
        return WRMFSGD.init_fake_tensor(self)

    def project_fake_tensor(self):
        WRMFSGD.project_fake_tensor(self)

    def retrain_surrogate(self):
        with torch.no_grad():
            aggregated_embedding = self.surrogate_model.adj_mat.spmm(self.surrogate_model.embedding.weight,
                                                                     self.fake_tensor.flatten().repeat(2), norm='left')
            self.surrogate_model.embedding.weight[self.n_users:self.n_users + self.n_fakes, :] = \
                aggregated_embedding[self.n_users:self.n_users + self.n_fakes, :]

        poisoned_data_mat = torch.cat([self.data_tensor, self.fake_tensor], dim=0)
        opt = Adam(self.surrogate_model.parameters(), lr=self.config['s_lr'])
        with higher.innerloop_ctx(self.surrogate_model, opt) as (fmodel, diffopt):
            fmodel.train()
            for i in range(self.unroll_steps):
                for users in self.surrogate_trainer.user_loader:
                    users = users[0]
                    batch_data = poisoned_data_mat[users, :]
                    scores = fmodel.predict(users, self.fake_tensor.detach())
                    loss = mse_loss(batch_data, scores, self.surrogate_trainer.weight)
                    loss += self.surrogate_trainer.l2_reg * torch.norm(scores, p=2) ** 2 / \
                            (users.shape[0] * self.dataset.n_items)
                    diffopt.step(loss)

            fmodel.eval()
            scores = fmodel.predict(self.target_users)
            adv_loss = ce_loss(scores, self.target_item)
            adv_grads = torch.autograd.grad(adv_loss, self.fake_tensor)[0]
            _, topk_items = scores.topk(self.topk, dim=1)
            hr = torch.eq(topk_items, self.target_item).float().sum(dim=1).mean()
        gc.collect()
        return adv_loss.item(), hr.item(), adv_grads

    def generate_fake_users(self, verbose=True, writer=None):
        WRMFSGD.generate_fake_users(self, verbose, writer)
        self.dataset.train_data = self.dataset.train_data[:-self.n_fakes]
        self.dataset.val_data = self.dataset.val_data[:-self.n_fakes]
        self.dataset.n_users -= self.n_fakes

