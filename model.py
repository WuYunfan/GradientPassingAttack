import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
from utils import generate_adj_mat
from torch.nn.init import kaiming_uniform_, xavier_normal, normal_, zeros_, ones_
import sys
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from torch.utils.checkpoint import checkpoint
import dgl
import multiprocessing as mp
import time


def get_model(config, dataset):
    config = config.copy()
    config['dataset'] = dataset
    model = getattr(sys.modules['model'], config['name'])
    model = model(config)
    return model


def init_one_layer(in_features, out_features):
    layer = nn.Linear(in_features, out_features)
    kaiming_uniform_(layer.weight)
    zeros_(layer.bias)
    return layer


class BasicModel(nn.Module):
    def __init__(self, model_config):
        super(BasicModel, self).__init__()
        if model_config.get('verbose', True):
            print(model_config)
        self.config = model_config
        self.name = model_config['name']
        self.device = model_config['device']
        self.n_users = model_config['dataset'].n_users
        self.n_items = model_config['dataset'].n_items
        self.trainable = True

    def predict(self, users):
        raise NotImplementedError

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))


class MF(BasicModel):
    def __init__(self, model_config):
        super(MF, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.embedding = nn.Embedding(self.n_users + self.n_items, self.embedding_size)
        normal_(self.embedding.weight, std=0.1)
        self.to(device=self.device)

    def bpr_forward(self, users, pos_items, neg_items):
        users_e = self.embedding(users)
        pos_items_e, neg_items_e = self.embedding(self.n_users + pos_items), self.embedding(self.n_users + neg_items)
        l2_norm_sq = torch.norm(users_e, p=2, dim=1) ** 2 + torch.norm(pos_items_e, p=2, dim=1) ** 2 \
                     + torch.norm(neg_items_e, p=2, dim=1) ** 2
        return users_e, pos_items_e, neg_items_e, l2_norm_sq

    def predict(self, users):
        user_e = self.embedding(users)
        scores = torch.mm(user_e, self.embedding.weight[self.n_users:].t())
        return scores


class LightGCN(BasicModel):
    def __init__(self, model_config):
        super(LightGCN, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.n_layers = model_config['n_layers']
        self.embedding = nn.Embedding(self.n_users + self.n_items, self.embedding_size)
        self.adj_mat = self.generate_graph(model_config['dataset'])
        normal_(self.embedding.weight, std=0.1)
        self.to(device=self.device)

    def generate_graph(self, dataset):
        adj_mat = generate_adj_mat(dataset, self.device)
        return adj_mat

    def get_rep(self):
        representations = self.embedding.weight
        all_layer_rep = [representations]
        for _ in range(self.n_layers):
            representations = self.adj_mat.spmm(representations, norm='both')
            all_layer_rep.append(representations)
        all_layer_rep = torch.stack(all_layer_rep, dim=0)
        final_rep = all_layer_rep.mean(dim=0)
        return final_rep

    def bpr_forward(self, users, pos_items, neg_items):
        rep = self.get_rep()
        users_e = self.embedding(users)
        pos_items_e, neg_items_e = self.embedding(self.n_users + pos_items), self.embedding(self.n_users + neg_items)
        l2_norm_sq = torch.norm(users_e, p=2, dim=1) ** 2 + torch.norm(pos_items_e, p=2, dim=1) ** 2 \
                     + torch.norm(neg_items_e, p=2, dim=1) ** 2
        users_r = rep[users, :]
        pos_items_r, neg_items_r = rep[self.n_users + pos_items, :], rep[self.n_users + neg_items, :]
        return users_r, pos_items_r, neg_items_r, l2_norm_sq

    def predict(self, users):
        rep = self.get_rep()
        users_r = rep[users, :]
        all_items_r = rep[self.n_users:, :]
        scores = torch.mm(users_r, all_items_r.t())
        return scores


class ItemKNN(BasicModel):
    def __init__(self, model_config):
        super(ItemKNN, self).__init__(model_config)
        self.k = model_config['k']
        self.data_mat, self.sim_mat = self.calculate_similarity(model_config['dataset'])
        self.trainable = False

    def calculate_similarity(self, dataset):
        data_mat = sp.coo_matrix((np.ones((len(dataset.train_array),)), np.array(dataset.train_array).T),
                                 shape=(self.n_users, self.n_items), dtype=np.float32).tocsr()
        item_degree = np.array(np.sum(data_mat, axis=0)).squeeze()
        row = []
        col = []
        data = []
        for item in range(dataset.n_items):
            intersections = data_mat.T.dot(data_mat[:, item]).toarray().squeeze()
            with np.errstate(invalid='ignore'):
                sims = intersections / (item_degree + item_degree[item] - intersections)
            sims[np.isnan(sims)] = 0.
            row.extend([item] * self.k)
            topk_items = np.argsort(sims)[-self.k:]
            col.extend(topk_items.tolist())
            data.extend(sims[topk_items].tolist())
        sim_mat = sp.coo_matrix((data, (row, col)), shape=(self.n_items, self.n_items), dtype=np.float32).tocsr()
        return data_mat, sim_mat

    def predict(self, users):
        users = users.cpu().numpy()
        profiles = self.data_mat[users, :]
        scores = torch.tensor(profiles.dot(self.sim_mat).toarray(), dtype=torch.float32, device=self.device)
        return scores


class MultiVAE(BasicModel):
    def __init__(self, model_config):
        super(MultiVAE, self).__init__(model_config)
        self.dropout = model_config['dropout']
        self.normalized_data_mat = self.get_data_mat(model_config['dataset'])

        self.e_layer_sizes = model_config['layer_sizes'].copy()
        self.e_layer_sizes.insert(0, self.normalized_data_mat.shape[1])
        self.d_layer_sizes = self.e_layer_sizes[::-1].copy()
        self.mid_size = self.e_layer_sizes[-1]
        self.e_layer_sizes[-1] = self.mid_size * 2
        self.encoder_layers = []
        self.decoder_layers = []
        for layer_idx in range(1, len(self.e_layer_sizes)):
            encoder_layer = init_one_layer(self.e_layer_sizes[layer_idx - 1], self.e_layer_sizes[layer_idx])
            self.encoder_layers.append(encoder_layer)
            decoder_layer = init_one_layer(self.d_layer_sizes[layer_idx - 1], self.d_layer_sizes[layer_idx])
            self.decoder_layers.append(decoder_layer)
        self.encoder_layers = nn.ModuleList(self.encoder_layers)
        self.decoder_layers = nn.ModuleList(self.decoder_layers)
        self.to(device=self.device)

    def get_data_mat(self, dataset):
        data_mat = sp.coo_matrix((np.ones((len(dataset.train_array),)), np.array(dataset.train_array).T),
                                 shape=(self.n_users, self.n_items), dtype=np.float32).tocsr()

        normalized_data_mat = normalize(data_mat, axis=1, norm='l2')
        return normalized_data_mat

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

    def ml_forward(self, users):
        users = users.cpu().numpy()
        profiles = self.normalized_data_mat[users, :]
        representations = get_sparse_tensor(profiles, self.device)

        representations = self.dropout_sp_mat(representations)
        representations = torch.sparse.mm(representations, self.encoder_layers[0].weight.t())
        representations += self.encoder_layers[0].bias[None, :]
        l2_norm_sq = torch.norm(self.encoder_layers[0].weight, p=2)[None] ** 2
        for layer in self.encoder_layers[1:]:
            representations = layer(torch.tanh(representations))
            l2_norm_sq += torch.norm(layer.weight, p=2)[None] ** 2

        mean, log_var = representations[:, :self.mid_size], representations[:, -self.mid_size:]
        std = torch.exp(0.5 * log_var)
        kl = torch.sum(-log_var + torch.exp(log_var) + mean ** 2, dim=1)
        epsilon = torch.randn(mean.shape[0], mean.shape[1], device=self.device)
        representations = mean + float(self.training) * epsilon * std

        for layer in self.decoder_layers[:-1]:
            representations = torch.tanh(layer(representations))
            l2_norm_sq += torch.norm(layer.weight, p=2)[None] ** 2
        scores = self.decoder_layers[-1](representations)
        l2_norm_sq += torch.norm(self.decoder_layers[-1].weight, p=2)[None] ** 2
        return scores, kl, l2_norm_sq

    def predict(self, users):
        scores, _, _ = self.ml_forward(users)
        if scores.shape[1] < self.n_items:
            padding = torch.full([scores.shape[0], self.n_items - scores.shape[1]], -np.inf, device=self.device)
            scores = torch.cat([scores, padding], dim=1)
        return scores


class NeuMF(BasicModel):
    def __init__(self, model_config):
        super(NeuMF, self).__init__(model_config)
        self.embedding_size = model_config['embedding_size']
        self.layer_sizes = model_config['layer_sizes']
        self.mf_embedding = nn.Embedding(self.n_users + self.n_items, self.embedding_size)
        self.mlp_embedding = nn.Embedding(self.n_users + self.n_items, self.layer_sizes[0] // 2)
        self.mlp_layers = []
        for layer_idx in range(1, len(self.layer_sizes)):
            dense_layer = nn.Linear(self.layer_sizes[layer_idx - 1], self.layer_sizes[layer_idx])
            self.mlp_layers.append(dense_layer)
        self.mlp_layers = nn.ModuleList(self.mlp_layers)
        self.output_layer = nn.Linear(self.layer_sizes[-1] + self.embedding_size, 1, bias=False)

        kaiming_uniform_(self.mf_embedding.weight)
        kaiming_uniform_(self.mlp_embedding.weight)
        self.init_mlp_layers()
        self.arch = 'gmf'
        self.to(device=self.device)

    def init_mlp_layers(self):
        for layer in self.mlp_layers:
            kaiming_uniform_(layer.weight)
            zeros_(layer.bias)
        ones_(self.output_layer.weight)

    def bce_forward(self, users, items):
        users_mf_e, items_mf_e = self.mf_embedding(users), self.mf_embedding(self.n_users + items)
        users_mlp_e, items_mlp_e = self.mlp_embedding(users), self.mlp_embedding(self.n_users + items)

        mf_vectors = users_mf_e * items_mf_e
        mlp_vectors = torch.cat([users_mlp_e, items_mlp_e], dim=1)
        for layer in self.mlp_layers:
            mlp_vectors = F.leaky_relu(layer(mlp_vectors))

        if self.arch == 'gmf':
            vectors = [mf_vectors, torch.zeros_like(mlp_vectors, device=self.device, dtype=torch.float32)]
        elif self.arch == 'mlp':
            vectors = [torch.zeros_like(mf_vectors, device=self.device, dtype=torch.float32), mlp_vectors]
        else:
            vectors = [mf_vectors, mlp_vectors]
        predict_vectors = torch.cat(vectors, dim=1)
        scores = predict_vectors * self.output_layer.weight
        l2_norm_sq = torch.norm(scores, p=2, dim=1) ** 2
        scores = scores.sum(dim=1)
        return scores, l2_norm_sq

    def predict(self, users):
        items = torch.arange(self.n_items, dtype=torch.int64, device=self.device).repeat(users.shape[0])
        users = users[:, None].repeat(1, self.n_items).flatten()
        scores, _ = self.bce_forward(users, items)
        scores = scores.reshape(-1, self.n_items)
        return scores
