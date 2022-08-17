from attacker.basic_attacker import BasicAttacker
import torch.nn as nn
from model import LightGCN
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


class TorchSparseMat:
    def __init__(self, coo_mat, shape, device):
        coo_mat = coo_mat.tocoo()
        self.shape = shape
        self.device = device
        self.row = torch.tensor(coo_mat.row, dtype=torch.int64, device=device)
        self.col = torch.tensor(coo_mat.col, dtype=torch.int64, device=device)
        self.row_degree = torch.tensor(np.array(np.sum(coo_mat, axis=1)).squeeze(),
                                       dtype=torch.float32, device=device)

    def add_new_entry(self, r, c):
        r = torch.tensor([r], dtype=torch.int64, device=self.device)
        c = torch.tensor([c], dtype=torch.int64, device=self.device)
        self.row = torch.cat([self.row, r], dim=0)
        self.col = torch.cat([self.col, c], dim=0)
        self.row_degree[r] += 1.


class IGCN(nn.Module):
    def __init__(self, model_config):
        super(IGCN, self).__init__()
        self.device = model_config['device']
        self.n_users = model_config['dataset'].n_users
        self.n_items = model_config['dataset'].n_items
        self.n_norm_users = model_config['n_norm_users']
        self.embedding_size = model_config['embedding_size']
        self.n_layers = model_config.get['n_layers']

        self.adj_mat = self.generate_graph(model_config['dataset'])
        self.feat_mat = self.generate_feat(model_config['dataset'])

        self.embedding = nn.Embedding(self.feat_mat.shape[1], self.embedding_size)
        normal_(self.embedding.weight, std=0.1)
        self.to(device=self.device)

    def generate_graph(self, dataset):
        adj_mat = generate_daj_mat(dataset).tocoo()
        adj_mat = TorchSparseMat(adj_mat, (self.n_users + self.n_items,
                                           self.n_users + self.n_items), self.device)
        return adj_mat

    def generate_feat(self, dataset):
        indices = []
        for user, item in dataset.train_array:
            indices.append([user, self.n_norm_users + item])
            if user < self.n_norm_users:
                indices.append([self.n_users + item, user])
        for user in range(self.n_users):
            indices.append([user, self.n_norm_users + self.n_items])
        for item in range(self.n_items):
            indices.append([self.n_users + item, self.n_norm_users + self.n_items + 1])
        feat_mat = sp.coo_matrix((np.ones((len(indices),)), np.array(indices).T),
                                 shape=(self.n_users + self.n_items,
                                        self.n_norm_users + self.n_items + 2), dtype=np.float32)
        feat_mat = TorchSparseMat(feat_mat, (self.n_users + self.n_items,
                                             self.n_norm_users + self.n_items + 2), self.device)
        return feat_mat

    def inductive_rep_layer(self):
        padding_tensor = torch.empty([max(self.feat_mat.shape) - self.feat_mat.shape[1], self.embedding_size],
                                     dtype=torch.float32, device=self.device)
        padding_features = torch.cat([self.embedding.weight, padding_tensor], dim=0)

        row, col = self.feat_mat.row, self.feat_mat.col
        values = torch.pow(self.feat_mat.row_degree[row], -1.)
        g = dgl.graph((col, row), num_nodes=max(self.feat_mat.shape), device=self.device)
        x = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=padding_features, rhs_data=values)
        x = x[:self.feat_mat.shape[0], :]
        return x

    def get_rep(self):
        representations = self.inductive_rep_layer()

        row, col = self.adj_mat.row, self.adj_mat.col
        values = torch.pow(self.adj_mat.row_degree[row], -0.5) * torch.pow(self.adj_mat.row_degree[col], -0.5)
        values[torch.isinf(values)] = 1.
        g = dgl.graph((col, row), num_nodes=self.adj_mat.shape[0], device=self.device)
        all_layer_rep = [representations]
        for _ in range(self.n_layers):
            representations = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=representations, rhs_data=values)
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def bpr_forward(self, users, pos_items, neg_items):
        rep = self.get_rep()
        users_r = rep[users, :]
        pos_items_r, neg_items_r = rep[self.n_users + pos_items, :], rep[self.n_users + neg_items, :]
        l2_norm_sq = torch.norm(users_r, p=2, dim=1) ** 2 + torch.norm(pos_items_r, p=2, dim=1) ** 2 \
                     + torch.norm(neg_items_r, p=2, dim=1) ** 2
        return users_r, pos_items_r, neg_items_r, l2_norm_sq

    def predict(self, users):
        return LightGCN.predict(self, users)


class IMF(IGCN):
    def __init__(self, model_config):
        super(IMF, self).__init__(model_config)

    def get_rep(self):
        representations = self.inductive_rep_layer()
        return representations


class IGCNTrainer(BasicTrainer):
    def __init__(self, trainer_config):
        super(IGCNTrainer, self).__init__(trainer_config)

        self.dataloader = DataLoader(self.dataset, batch_size=trainer_config['batch_size'],
                                     num_workers=trainer_config['dataloader_num_workers'])
        self.initialize_optimizer()
        self.l2_reg = trainer_config['l2_reg']
        self.aux_reg = trainer_config['aux_reg']

    def train_one_epoch(self):
        losses = AverageMeter()
        for batch_data in self.dataloader:
            inputs = batch_data[:, 0, :].to(device=self.device, dtype=torch.int64)
            users, pos_items, neg_items = inputs[:, 0],  inputs[:, 1],  inputs[:, 2]
            users_r, pos_items_r, neg_items_r, l2_norm_sq = self.model.bpr_forward(users, pos_items, neg_items)
            pos_scores = torch.sum(users_r * pos_items_r, dim=1)
            neg_scores = torch.sum(users_r * neg_items_r, dim=1)
            bpr_loss = F.softplus(neg_scores - pos_scores).mean()

            neg_users = users[torch.randperm(users.shape[0])]
            users_e = self.model.embedding(users)
            neg_users_e = self.model.embedding(neg_users)
            pos_scores = torch.sum(users_r * users_e, dim=1)
            neg_scores = torch.sum(users_r * neg_users_e, dim=1)
            aux_loss = F.softplus(neg_scores - pos_scores).mean()

            reg_loss = self.l2_reg * l2_norm_sq.mean() + self.aux_reg * aux_loss
            loss = bpr_loss + reg_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.update(loss.item(), inputs.shape[0])
        return losses.avg


class ParameterPropagation(Function):
    @staticmethod
    def forward(ctx, rep, mat, order):
        ctx.order = order
        ctx.save_for_backward(mat.row, mat.col)
        return rep

    @staticmethod
    def backward(ctx, grad_out):
        row, col = ctx.saved_tensors
        values = torch.ones_like(row, dtype=torch.float32, device=row.device)
        g = dgl.graph((col, row), num_nodes=grad_out.shape[0], device=row.device)
        order = ctx.order
        grad = grad_out
        for _ in range(order):
            grad = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=grad, rhs_data=values) + grad_out
        return grad, None, None


class ERAP4(BasicAttacker):
    def __init__(self, attacker_config):
        super(ERAP4, self).__init__(attacker_config)
        self.surrogate_model_config = attacker_config['surrogate_model_config']
        self.surrogate_model_config['device'] = self.device
        self.surrogate_model_config['dataset'] = self.dataset
        self.surrogate_model_config['n_norm_users'] = self.n_users
        self.surrogate_trainer_config = attacker_config['surrogate_trainer_config']
        self.surrogate_trainer_config['device'] = self.device
        self.surrogate_trainer_config['dataset'] = self.dataset
        self.propagation_order = attacker_config['propagation_order']

        self.dataset.n_users += self.n_fakes
        propagation_mat = generate_daj_mat(self.dataset)
        self.dataset.n_users -= self.n_fakes
        idxes = np.arange(propagation_mat.shape[1])
        propagation_mat[idxes, idxes] = 1.
        self.propagation_mat = TorchSparseMat(propagation_mat,
                                              (self.n_users + self.n_fakes + self.n_items,
                                               self.n_users + self.n_fakes + self.n_items), self.device)

        test_user = TensorDataset(torch.arange(self.n_users, dtype=torch.int64, device=self.device))
        self.test_user_loader = DataLoader(test_user, batch_size=self.surrogate_trainer_config['test_batch_size'])

    def retraining_approximate(self, model, opt, u, i):
        model.feat_mat.add_new_entry(u, model.n_norm_users + i)
        model.adj_mat.add_new_entry(u, model.n_users + i)
        model.adj_mat.add_new_entry(model.n_users + i, u)
        self.propagation_mat.add_new_entry(u, self.n_users + self.n_fakes + i)
        self.propagation_mat.add_new_entry(self.n_users + self.n_fakes + i, u)

        model.train()
        rep = model.get_rep()
        padding_tensor = torch.empty([self.n_users + self.n_fakes - model.n_users, model.embedding_size],
                                     dtype=torch.float32, device=self.device)
        padding_rep = torch.cat([rep[:model.n_users, :], padding_tensor, rep[model.n_users:, :]], dim=0)
        rep = ParameterPropagation.apply(padding_rep, self.propagation_mat, self.propagation_order)
        loss = -F.softplus((rep[u, :] * rep[self.n_users + self.n_fakes + i, :]).sum())
        opt.zero_grad()
        loss.backward()
        opt.step()

    def backup(self, model):
        feat_mat = copy.deepcopy(model.feat_mat)
        adj_mat = copy.deepcopy(model.adj_mat)
        propagation_mat = copy.deepcopy(self.propagation_mat)
        params = []
        for param in model.parameters():
            params.append(copy.copy(param.data))
        return feat_mat, adj_mat, propagation_mat, params

    def restore(self, model, feat_mat, adj_mat, propagation_mat, params):
        model.feat_mat = feat_mat
        model.adj_mat = adj_mat
        self.propagation_mat = propagation_mat
        with torch.no_grad():
            for param, param_b in zip(model.parameters(), params):
                param.data = param_b

    def evaluate_score(self, model, opt, u, i):
        feat_mat, adj_mat, propagation_mat, params = self.backup(model)
        self.retraining_approximate(model, opt, u, i)

        model.eval()
        adv_loss = AverageMeter()
        with torch.no_grad():
            for users in self.test_user_loader:
                users = users[0]
                scores = self.model.predict(users)
                adv_loss.update(ce_loss(scores, self.target_item).item(), users.shape[0])
        self.restore(model, feat_mat, adj_mat, propagation_mat, params)
        return adv_loss.avg

    def generate_fake_users(self, verbose=True, writer=None):
        self.dataset.attack_data = [[] for _ in range(self.n_users)]
        for u in range(self.n_users):
            if self.target_item not in self.dataset.train_data[u]:
                self.dataset.attack_data[u].append(self.target_item)

        for f_u in range(self.n_fakes):
            self.dataset.n_users += 1
            self.dataset.train_data.append([])
            self.dataset.val_data.append([])
            self.dataset.attack_data.append([])

            model = getattr(sys.modules['erap4_attacker'], self.surrogate_model_config['name'])
            model = model(self.surrogate_model_config['name'])
            self.surrogate_trainer_config['model'] = model
            trainer = IGCNTrainer(self.surrogate_trainer_config)
            trainer.train(verbose=True)

            _, metrics = self.trainer.eval('attack')
            print('Hit ratio after injecting {:d} fake users, {:.3f}%@{:d}.'.
                  format(f_u, metrics['Recall'][trainer.topks[0]] * 100, trainer.topks[0]))

            opt = SGD(model.parameters(), lr=1.)
            for _ in range(self.n_inters):
                scores = np.zeros([self.n_items], dtype=np.float32)
                for i in range(self.n_items):
                    if i in self.dataset.train_data[-1]:
                        scores[i] = -np.inf
                    else:
                        scores[i] = self.evaluate_score(model, opt, self.n_users + f_u, i)
                i = np.argmax(scores)
                self.dataset.train_data[-1].append(i)
                self.retraining_approximate(model, opt, self.n_users + f_u, i)

