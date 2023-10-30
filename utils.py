import numpy as np
import torch
import random
import os
import sys
import scipy.sparse as sp
import torch.nn.functional as F
import dgl


def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_run(log_path, seed):
    set_seed(seed)
    if not os.path.exists(log_path): os.mkdir(log_path)
    f = open(os.path.join(log_path, 'log.txt'), 'w')
    f = Unbuffered(f)
    sys.stderr = f
    sys.stdout = f


def generate_adj_mat(train_array, model):
    train_array = torch.tensor(train_array, dtype=torch.int64, device=model.device)
    users, items = train_array[:, 0], train_array[:, 1]
    row = torch.cat([users, items + model.n_users])
    col = torch.cat([items + model.n_users, users])
    adj_mat = TorchSparseMat(row, col, (model.n_users + model.n_items,
                                        model.n_users + model.n_items), model.device)
    return adj_mat


class TorchSparseMat:
    def __init__(self, row, col, shape, device):
        self.shape = shape
        self.device = device
        self.row = row
        self.col = col
        self.g = dgl.graph((self.col, self.row), num_nodes=max(shape), device=device)
        self.n_non_zeros = self.row.shape[0]

        eps = torch.tensor(1.e-8, dtype=torch.float32, device=device)
        values = torch.ones([self.n_non_zeros], dtype=torch.float32, device=device)
        degree = dgl.ops.gspmm(self.g, 'copy_rhs', 'sum', lhs_data=None, rhs_data=values)
        degree = torch.where(degree > 0, degree, eps)
        self.inv_deg = torch.pow(degree, -0.5)

    def spmm(self, r_mat, value_tensor=None, norm=None):
        if value_tensor is None:
            values = torch.ones([self.n_non_zeros], dtype=torch.float32, device=self.device)
        else:
            values = value_tensor

        assert r_mat.shape[0] == self.shape[1]
        padding_tensor = torch.empty([max(self.shape) - r_mat.shape[0], r_mat.shape[1]],
                                     dtype=torch.float32, device=self.device)
        padded_r_mat = torch.cat([r_mat, padding_tensor], dim=0)

        if norm == 'both':
            values = values * self.inv_deg[self.row] * self.inv_deg[self.col]
        x = dgl.ops.gspmm(self.g, 'mul', 'sum', lhs_data=padded_r_mat, rhs_data=values)
        return x[:self.shape[0], :]


class AverageMeter:
    def __init__(self):
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


def get_target_items(dataset, top_ratio=1., num=10):
    data_mat = sp.coo_matrix((np.ones((len(dataset.train_array),)), np.array(dataset.train_array).T),
                             shape=(dataset.n_users, dataset.n_items), dtype=np.float32).tocsr()
    item_degree = np.array(np.sum(data_mat, axis=0)).squeeze()
    selected_items = np.argsort(item_degree)[-int(dataset.n_items * top_ratio):]
    target_items = np.random.choice(selected_items, size=num, replace=False)
    return target_items


def mse_loss(profiles, scores):
    n_profiles = 1. - profiles
    loss_p = ((scores - 1) ** 2 * profiles).sum() / profiles.sum()
    loss_n = ((scores + 1) ** 2 * n_profiles).sum() / n_profiles.sum()
    return loss_p + loss_n


def ce_loss(scores, target_item):
    log_probs = F.log_softmax(scores, dim=1)
    return -log_probs[:, target_item].mean()


def topk_loss(scores, target_item, topk, kappa):
    top_scores, _ = scores.topk(topk, dim=1)
    target_scores = scores[:, target_item]
    loss = F.logsigmoid(top_scores[:, -1]) - F.logsigmoid(target_scores)
    loss = torch.max(loss, -kappa).mean()
    return loss
