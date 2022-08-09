import numpy as np
import torch
import random
import os
import sys
import scipy.sparse as sp
import torch.nn.functional as F


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


def get_sparse_tensor(mat, device):
    coo = mat.tocoo()
    indexes = np.stack([coo.row, coo.col], axis=0)
    indexes = torch.tensor(indexes, dtype=torch.int64, device=device)
    data = torch.tensor(coo.data, dtype=torch.float32, device=device)
    sp_tensor = torch.sparse.FloatTensor(indexes, data, torch.Size(coo.shape)).coalesce()
    return sp_tensor


def generate_daj_mat(dataset):
    train_array = np.array(dataset.train_array)
    users, items = train_array[:, 0], train_array[:, 1]
    row = np.concatenate([users, items + dataset.n_users], axis=0)
    column = np.concatenate([items + dataset.n_users, users], axis=0)
    adj_mat = sp.coo_matrix((np.ones(row.shape), np.stack([row, column], axis=0)),
                            shape=(dataset.n_users + dataset.n_items, dataset.n_users + dataset.n_items),
                            dtype=np.float32).tocsr()
    return adj_mat


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


def get_target_items(dataset):
    data_mat = sp.coo_matrix((np.ones((len(dataset.train_array),)), np.array(dataset.train_array).T),
                             shape=(dataset.n_users, dataset.n_items), dtype=np.float32).tocsr()
    item_degree = np.array(np.sum(data_mat, axis=0)).squeeze()
    selected_items = np.argsort(item_degree)[int(dataset.n_items * 0.75):]
    target_items = np.random.choice(selected_items, size=10, replace=False)
    return target_items


def mse_loss(profiles, scores, device, weight):
    weights = torch.ones_like(profiles, dtype=torch.float32, device=device)
    weights[profiles > 0] = weight
    loss = weights * (profiles - scores) ** 2
    loss = torch.mean(loss)
    return loss


def ce_loss(scores, target_item):
    log_probs = F.log_softmax(scores, dim=1)
    return -log_probs[:, target_item].mean()


def wmw_loss(scores, target_item, topk, b):
    top_scores, _ = scores.topk(topk, dim=1)
    target_scores = scores[:, target_item]
    loss = top_scores - target_scores[:, None]
    loss = torch.sigmoid(loss / b).mean()
    return loss
