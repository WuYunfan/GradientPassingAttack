import torch
from torch.optim import Adam, SGD
import scipy.sparse as sp
import numpy as np
import torch.nn as nn
from utils import get_sparse_tensor, AverageMeter
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import time
from attacker.basic_attacker import BasicAttacker
from torch.nn.init import kaiming_uniform_, calculate_gain, normal_, zeros_, ones_
from model import BasicModel
import torch_sparse
import torch_scatter
from torch.optim.lr_scheduler import StepLR


class IGCN(BasicModel):
    def __init__(self, model_config):
        super(IGCN, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.n_layers = model_config['n_layers']
        self.dropout = model_config['dropout']
        self.feature_ratio = model_config['feature_ratio']
        self.adj = self.generate_graph(model_config['dataset'])
        self.feat, self.user_map, self.item_map = self.generate_feat(model_config['dataset'])

        self.dense_layer = nn.Linear(self.feat.shape[1], self.embedding_size)
        normal_(self.dense_layer.weight, std=0.1)
        zeros_(self.dense_layer.bias)
        self.to(device=self.device)

    def generate_graph(self, dataset):
        sub_mat = sp.coo_matrix((np.ones((len(dataset.train_array),)), np.array(dataset.train_array).T),
                                shape=(self.n_users, self.n_items), dtype=np.float32)
        adj_mat = sp.lil_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat[:self.n_users, self.n_users:] = sub_mat
        adj_mat[self.n_users:, :self.n_users] = sub_mat.T

        adj_mat = get_sparse_tensor(adj_mat, self.device)
        return adj_mat

    def generate_feat(self, dataset, is_updating=False):
        if not is_updating:
            sub_mat = sp.coo_matrix((np.ones((len(dataset.train_array),)), np.array(dataset.train_array).T),
                                    shape=(self.n_users, self.n_items), dtype=np.float32)
            user_degree = np.array(np.sum(sub_mat, axis=1)).squeeze()
            popular_users = np.argsort(user_degree)[-int(self.n_users * self.feature_ratio):]
            print('User degree ratio {:.3f}%'.format(np.sum(user_degree[popular_users]) * 100. / np.sum(user_degree)))
            user_map = dict()
            for idx, user in enumerate(popular_users):
                user_map[user] = idx
            item_degree = np.array(np.sum(sub_mat, axis=0)).squeeze()
            popular_items = np.argsort(item_degree)[-int(self.n_items * self.feature_ratio):]
            print('Item degree ratio {:.3f}%'.format(np.sum(item_degree[popular_items]) * 100. / np.sum(item_degree)))
            item_map = dict()
            for idx, item in enumerate(popular_items):
                item_map[item] = idx
        else:
            user_map = self.user_map
            item_map = self.item_map

        user_dim, item_dim = len(user_map), len(item_map)
        indices = []
        for user, item in dataset.train_array:
            if item in item_map:
                indices.append([user, user_dim + item_map[item]])
            if user in user_map:
                indices.append([self.n_users + item, user_map[user]])
        for user in range(self.n_users):
            indices.append([user, user_dim + item_dim])
        for item in range(self.n_items):
            indices.append([self.n_users + item, user_dim + item_dim + 1])

        feat = sp.coo_matrix((np.ones((len(indices),)), np.array(indices).T),
                             shape=(self.n_users + self.n_items, user_dim + item_dim + 2), dtype=np.float32)
        degree = np.array(np.sum(feat, axis=1)).squeeze()
        print('User feat number {:.3f}, Item feat number {:.3f}'
              .format(degree[:self.n_users].mean(), degree[self.n_users:].mean()))
        feat = get_sparse_tensor(feat, self.device)
        return feat, user_map, item_map

    def dropout_sp_mat(self, mat):
        if not self.training:
            return mat
        random_tensor = 1 - self.dropout
        random_tensor += torch.rand(mat._nnz()).to(self.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = mat.indices()
        v = mat.values()

        i = i[:, dropout_mask]
        v = v[dropout_mask] / (1. - self.dropout)
        out = torch.sparse.FloatTensor(i, v, mat.shape).coalesce()
        return out

    def merge_fake_users(self, fake_indices, fake_values):
        if fake_indices is None:
            return self.adj, self.feat
        row, column = fake_indices
        row = row + self.n_users + self.n_items
        column = column + self.n_users

        indices, values = self.adj.indeces(), self.adj.values()
        indices = torch.cat([indices, torch.stack([row, column], dim=0), torch.stack([column, row], dim=0)], dim=1)
        values = torch.cat([values, fake_values, fake_values], dim=0)
        n_rows = torch.max(row)
        degree = torch_scatter.scatter(values, indices[0, :], dim=0, reduce='sum')
        inv_degree = torch.pow(degree, -0.5)
        inv_degree[torch.isinf(inv_degree)] = inv_degree[torch.logical_not(torch.isinf(inv_degree))].mean()
        values = values * inv_degree[indices[0, :]] * inv_degree[indices[1, :]]
        adj = torch.sparse.FloatTensor(indices, values, torch.Size([n_rows, n_rows])).coalesce()

        user_dim, item_dim = len(self.user_map), len(self.item_map)
        new_row = []
        new_column = []
        new_values = []
        for edge_idx in range(row.shape[0]):
            r, c = row[edge_idx].item(), column[edge_idx].item() - self.n_users
            if c in self.item_map:
                new_row.append(r)
                new_column.append(user_dim + self.item_map[c])
                new_values.append(fake_values[edge_idx])
        for fake_u in range(self.n_users + self.n_items, n_rows):
            new_row.append(fake_u)
            new_column.append(user_dim + item_dim)
            new_values.append(torch.tensor(1., dtype=torch.float32, device=self.device))
        new_row = torch.tensor(new_row, dtype=torch.int64, device=self.device)
        new_column = torch.tensor(new_column, dtype=torch.int64, device=self.device)
        new_values = torch.cat(new_values, dim=0)
        indices, values = self.feat.indeces(), self.feat.values()
        indices = torch.cat([indices, torch.stack([new_row, new_column], dim=0)], dim=1)
        values = torch.cat([values, new_values], dim=0)
        degree = torch_scatter.scatter(values, indices[0, :], dim=0, reduce='sum')
        values = values / degree[indices[0, :]]
        feat = torch.sparse.FloatTensor(indices, values, torch.Size([n_rows, user_dim + item_dim + 2])).coalesce()
        return adj, feat

    def get_rep(self, fake_indices, fake_values):
        adj, feat = self.merge_fake_users(fake_indices, fake_values)

        feat = self.dropout_sp_mat(feat)
        representations = torch.sparse.mm(feat, self.dense_layer.weight.t()) + self.dense_layer.bias[None, :]

        all_layer_rep = [representations]
        dropped_adj = self.dropout_sp_mat(adj)
        for _ in range(self.n_layers):
            representations = torch.sparse.mm(dropped_adj, representations)
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep[:self.n_users + self.n_items, :]

    def bpr_forward(self, users, pos_items, neg_items):
        rep = self.get_rep(None, None)
        users_r = rep[users, :]
        pos_items_r, neg_items_r = rep[self.n_users + pos_items, :], rep[self.n_users + neg_items, :]
        l2_norm_sq = torch.norm(users_r, p=2, dim=1) ** 2 + torch.norm(pos_items_r, p=2, dim=1) ** 2 \
                     + torch.norm(neg_items_r, p=2, dim=1) ** 2
        return users_r, pos_items_r, neg_items_r, l2_norm_sq

    def predict(self, users, fake_indices=None, fake_values=None):
        rep = self.get_rep(fake_indices, fake_values)
        users_r = rep[users, :]
        all_items_r = rep[self.n_users:, :]
        scores = torch.mm(users_r, all_items_r.t())
        return scores

    def save(self, path):
        params = {'sate_dict': self.state_dict(), 'user_map': self.user_map, 'item_map': self.item_map}
        torch.save(params, path)

    def load(self, path):
        params = torch.load(path)
        self.load_state_dict(params['sate_dict'])
        self.user_map = params['user_map']
        self.item_map = params['item_map']
        self.feat, _, _ = self.generate_feat(self.config['dataset'], is_updating=True)


class GBFUG(BasicAttacker):
    def __init__(self, attacker_config):
        super(GBFUG, self).__init__(attacker_config)
        self.topk = attacker_config['topk']
        self.b = attacker_config.get('b', 0.01)
        self.candidate_item_rate = attacker_config.get('candidate_item_rate', 1.)
        self.initial_lr = attacker_config['lr']
        self.train_epoch = attacker_config['train_epoch']
        self.adv_epochs = attacker_config['adv_epochs']

        self.fake_indices, self.fake_tensor = self.init_fake_data()
        self.fake_tensor.requires_grad_()
        self.adv_opt = SGD([self.fake_tensor], lr=self.initial_lr, momentum=attacker_config['momentum'])
        self.scheduler = StepLR(self.adv_opt, step_size=self.adv_epochs / 3, gamma=0.1)

        igcn_config = attacker_config['igcn_config']
        igcn_config['name'] = 'IGCN'
        igcn_config['dataset'] = self.dataset
        igcn_config['device'] = self.device
        self.igcn_model = IGCN(igcn_config)
        self.dataloader = DataLoader(self.dataset, batch_size=attacker_config['batch_size'],
                                     shuffle=True, num_workers=attacker_config['dataloader_num_workers'])
        self.train_opt = Adam(self.igcn_model.parameters(), lr=igcn_config['lr'])
        self.l2_reg = igcn_config['l2_reg']

        test_users = TensorDataset(torch.arange(self.n_users, dtype=torch.int64, device=self.device))
        self.test_user_loader = DataLoader(test_users, batch_size=attacker_config['test_batch_size'], shuffle=True)

    def init_fake_data(self):
        data_mat = sp.coo_matrix((np.ones((len(self.dataset.train_array),)), np.array(self.dataset.train_array).T),
                                 shape=(self.n_users, self.n_items), dtype=np.float32).tocsr()
        item_degree = np.array(np.sum(data_mat, axis=0)).squeeze()
        popular_items = np.argsort(item_degree)[::-1].copy()[:int(self.n_items * self.candidate_item_rate)]
        data_mat = data_mat[:, popular_items]

        user_degree = np.array(np.sum(data_mat, axis=1)).squeeze()
        qualified_users = data_mat[user_degree <= self.n_inters, :]
        sample_idx = np.random.choice(qualified_users.shape[0], self.n_fakes, replace=False)
        fake_tensor = qualified_users[sample_idx, :].toarray()
        fake_tensor = torch.tensor(fake_tensor, dtype=torch.float32, device=self.device)

        popular_items = torch.tensor(popular_items, dtype=torch.int64, device=self.device)
        row = torch.arange(self.n_fakes, dtype=torch.int64, device=self.device)[:, None].\
            repeat(1, popular_items.shape[0]).flatten()
        column = popular_items.repeat(self.n_fakes)
        fake_indices = torch.stack([row, column], dim=0)
        return fake_indices, fake_tensor

    def project_fake_tensor(self):
        with torch.no_grad():
            _, items = self.fake_tensor.topk(self.n_inters, dim=1)
            self.fake_tensor.zero_()
            for fake_user in range(self.n_fakes):
                self.fake_tensor[fake_user, items[fake_user, :]] = 1.

    @staticmethod
    def ce_loss(scores, target_item):
        log_probs = F.log_softmax(scores, dim=1)
        return -log_probs[:, target_item].mean()

    def wmw_loss(self, scores, target_item):
        top_scores, _ = scores.topk(self.topk, dim=1)
        target_scores = scores[:, target_item]
        loss = top_scores - target_scores[:, None]
        loss = torch.sigmoid(loss / self.b).mean()
        return loss

    def generate_fake_users(self, verbose=True, writer=None, igcn_model=None):
        if igcn_model is None:
            self.igcn_model.train()
            for epoch in range(self.train_epoch):
                start_time = time.time()
                losses = AverageMeter()
                for batch_data in self.dataloader:
                    inputs = batch_data[:, 0, :].to(device=self.device, dtype=torch.int64)
                    users, pos_items, neg_items = inputs[:, 0], inputs[:, 1], inputs[:, 2]

                    users_r, pos_items_r, neg_items_r, l2_norm_sq = self.igcn_model.bpr_forward(users, pos_items, neg_items)
                    pos_scores = torch.sum(users_r * pos_items_r, dim=1)
                    neg_scores = torch.sum(users_r * neg_items_r, dim=1)

                    bpr_loss = F.softplus(neg_scores - pos_scores).mean()
                    reg_loss = self.l2_reg * l2_norm_sq.mean()
                    loss = bpr_loss + reg_loss
                    self.train_opt.zero_grad()
                    loss.backward()
                    self.train_opt.step()
                    losses.update(loss.item(), inputs.shape[0])

                consumed_time = time.time() - start_time
                if verbose:
                    print('Epoch {:d}/{:d}, IGCN Loss: {:.3f}, Time: {:.3f}s'.
                          format(epoch, self.train_epoch, losses.avg, consumed_time))
                if writer:
                    writer.add_scalar('IGCN/Loss', losses.avg, epoch)
        else:
            self.igcn_model = igcn_model

        self.igcn_model.eval()
        for epoch in range(self.adv_epochs):
            self.scheduler.step()

            start_time = time.time()
            hit_k = AverageMeter()
            adv_losses = AverageMeter()

            for users in self.test_user_loader:
                users = users[0]
                scores = self.igcn_model.predict(users, self.fake_indices, self.fake_tensor.flatten())
                adv_loss = self.wmw_loss(scores, self.target_item)
                adv_loss.backward()

                _, topk_items = scores.topk(self.topk, dim=1)
                hr = torch.eq(topk_items, self.target_item).float().sum(dim=1).mean()
                hit_k.update(hr.item(), users.shape[0])
                adv_losses.update(adv_loss.item(), users.shape[0])

            self.fake_tensor.grad = F.normalize(self.fake_tensor.grad, p=2, dim=1)
            self.adv_opt.step()
            self.adv_opt.zero_grad()
            self.project_fake_tensor()

            adv_loss = adv_losses.avg
            hit_k = hit_k.avg
            consumed_time = time.time() - start_time
            if verbose:
                print('Epoch {:d}/{:d}, Adv Loss: {:.3f}, Hit Ratio@{:d}: {:.3f}, Time: {:.3f}s'.
                      format(epoch, self.adv_epochs, adv_loss, self.topk, hit_k, consumed_time))
            if writer:
                writer.add_scalar('GBFUG/Adv_Loss', adv_loss, epoch)
                writer.add_scalar('GBFUG/Hit_Ratio@{:d}'.format(self.topk), hit_k, epoch)

        self.fake_users = torch.sparse.FloatTensor(self.fake_indices, self.fake_tensor,
                                                   torch.Size(self.n_fakes, self.n_items)).coalesce()
        self.fake_users = self.fake_users.to_dense().detach().cpu().numpy()
