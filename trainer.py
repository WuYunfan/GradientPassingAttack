import scipy
import torch
import sys
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, SGD
import time
import numpy as np
import os
from utils import AverageMeter, generate_adj_mat
import torch.nn.functional as F
import scipy.sparse as sp
import optuna
from utils import mse_loss


def get_trainer(config, model):
    config = config.copy()
    config['dataset'] = model.dataset
    config['device'] = model.device
    config['model'] = model
    trainer = getattr(sys.modules['trainer'], config['name'])
    trainer = trainer(config)
    return trainer


class PPConfig:
    def __init__(self, trainer_config):
        self.order = trainer_config.get('pp_step', 0)
        if self.order != 0:
            self.mat = generate_adj_mat(trainer_config['dataset'], trainer_config['device'])
            self.threshold = trainer_config['pp_threshold']


class BasicTrainer:
    def __init__(self, trainer_config):
        if trainer_config.get('verbose', True):
            print(trainer_config)
        self.config = trainer_config
        self.name = trainer_config['name']
        self.dataset = trainer_config['dataset']
        self.model = trainer_config['model']
        self.topks = trainer_config['topks']
        self.device = trainer_config['device']
        self.n_epochs = trainer_config['n_epochs']
        self.negative_sample_ratio = trainer_config.get('neg_ratio', 1)
        self.max_patience = trainer_config.get('max_patience', 100)
        self.val_interval = trainer_config.get('val_interval', 1)
        self.epoch = 0
        self.best_ndcg = -np.inf
        self.save_path = None
        self.opt = None
        self.pp_config = PPConfig(trainer_config)

        test_user = TensorDataset(torch.arange(self.dataset.n_users, dtype=torch.int64, device=self.device))
        self.test_user_loader = DataLoader(test_user, batch_size=trainer_config['test_batch_size'])

    def initialize_optimizer(self):
        opt = getattr(sys.modules[__name__], self.config['optimizer'])
        self.opt = opt(self.model.parameters(), lr=self.config['lr'])

    def train_one_epoch(self):
        raise NotImplementedError

    def record(self, writer, stage, metrics):
        for metric in metrics:
            for k in self.topks:
                writer.add_scalar('{:s}_{:s}/{:s}_{:s}@{:d}'
                                  .format(self.model.name, self.name, stage, metric, k)
                                  , metrics[metric][k], self.epoch)

    def train(self, verbose=True, writer=None, extra_eval=None, trial=None):
        if not self.model.trainable:
            results, metrics = self.eval('val')
            if verbose:
                print('Validation result. {:s}'.format(results))
            ndcg = metrics['NDCG'][self.topks[0]]
            return ndcg

        self.dataset.negative_sample_ratio = self.negative_sample_ratio
        if not os.path.exists('checkpoints'): os.mkdir('checkpoints')
        patience = self.max_patience
        for self.epoch in range(self.n_epochs):
            start_time = time.time()
            self.model.train()
            loss = self.train_one_epoch()
            consumed_time = time.time() - start_time
            if verbose:
                print('Epoch {:d}/{:d}, Loss: {:.6f}, Time: {:.3f}s'
                      .format(self.epoch, self.n_epochs, loss, consumed_time))
            if writer:
                writer.add_scalar('{:s}_{:s}/train_loss'.format(self.model.name, self.name), loss, self.epoch)

            if (self.epoch + 1) % self.val_interval != 0:
                continue

            start_time = time.time()
            results, metrics = self.eval('val')
            consumed_time = time.time() - start_time
            if verbose:
                print('Validation result. {:s}Time: {:.3f}s'.format(results, consumed_time))
            if writer:
                self.record(writer, 'validation', metrics)

            ndcg = metrics['NDCG'][self.topks[0]]
            if ndcg > self.best_ndcg:
                if self.save_path:
                    os.remove(self.save_path)
                self.save_path = os.path.join('checkpoints', '{:s}_{:s}_{:s}_{:.3f}.pth'
                                              .format(self.model.name, self.name, self.dataset.name, ndcg * 100))
                self.best_ndcg = ndcg
                self.model.save(self.save_path)
                patience = self.max_patience
                if verbose:
                    print('Best NDCG, save model to {:s}'.format(self.save_path))
            else:
                patience -= self.val_interval
                if patience <= 0:
                    print('Early stopping at epoch {:d}!'.format(self.epoch))
                    break

            if extra_eval is not None:
                recall = extra_eval[0](self, *extra_eval[1], verbose)
            if trial is not None:
                if extra_eval is not None:
                    trial.report(recall, self.epoch)
                else:
                    trial.report(ndcg, self.epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        if self.best_ndcg != -np.inf:
            self.model.load(self.save_path)
            print('Best NDCG {:.3f}'.format(self.best_ndcg))
        return self.best_ndcg

    def calculate_metrics(self, eval_data, rec_items):
        results = {'Precision': {}, 'Recall': {}, 'NDCG': {}}
        hit_matrix = np.zeros_like(rec_items, dtype=np.float32)
        for user in range(rec_items.shape[0]):
            for item_idx in range(rec_items.shape[1]):
                if rec_items[user, item_idx] in eval_data[user]:
                    hit_matrix[user, item_idx] = 1.
        eval_data_len = np.array([len(items) for items in eval_data], dtype=np.int32)

        for k in self.topks:
            hit_num = np.sum(hit_matrix[:, :k], axis=1)
            precisions = hit_num / k
            with np.errstate(invalid='ignore'):
                recalls = hit_num / eval_data_len

            max_hit_num = np.minimum(eval_data_len, k)
            max_hit_matrix = np.zeros_like(hit_matrix[:, :k], dtype=np.float32)
            for user, num in enumerate(max_hit_num):
                max_hit_matrix[user, :num] = 1.
            denominator = np.log2(np.arange(2, k + 2, dtype=np.float32))[None, :]
            dcgs = np.sum(hit_matrix[:, :k] / denominator, axis=1)
            idcgs = np.sum(max_hit_matrix / denominator, axis=1)
            with np.errstate(invalid='ignore'):
                ndcgs = dcgs / idcgs

            user_masks = (max_hit_num > 0)
            results['Precision'][k] = precisions[user_masks].mean()
            results['Recall'][k] = recalls[user_masks].mean()
            results['NDCG'][k] = ndcgs[user_masks].mean()
        return results

    def get_rec_items(self, val_or_test, banned_items, k=None):
        rec_items = []
        with torch.no_grad():
            for users in self.test_user_loader:
                users = users[0]
                scores = self.model.predict(users)

                if val_or_test != 'train':
                    users = users.cpu().numpy().tolist()
                    exclude_user_indexes = []
                    exclude_items = []
                    for user_idx, user in enumerate(users):
                        items = self.dataset.train_data[user]
                        exclude_user_indexes.extend([user_idx] * len(items))
                        exclude_items.extend(items)
                    scores[exclude_user_indexes, exclude_items] = -np.inf
                if banned_items is not None:
                    scores[:, banned_items] = -np.inf

                if k is None:
                    k = max(self.topks)
                _, items = torch.topk(scores, k=k)
                rec_items.append(items.cpu().numpy())

        rec_items = np.concatenate(rec_items, axis=0)
        return rec_items

    def eval(self, val_or_test, banned_items=None):
        self.model.eval()
        eval_data = getattr(self.dataset, val_or_test + '_data')

        rec_items = self.get_rec_items(val_or_test, banned_items)
        metrics = self.calculate_metrics(eval_data, rec_items)

        precision = ''
        recall = ''
        ndcg = ''
        for k in self.topks:
            precision += '{:.3f}%@{:d}, '.format(metrics['Precision'][k] * 100., k)
            recall += '{:.3f}%@{:d}, '.format(metrics['Recall'][k] * 100., k)
            ndcg += '{:.3f}%@{:d}, '.format(metrics['NDCG'][k] * 100., k)
        results = 'Precision: {:s}Recall: {:s}NDCG: {:s}'.format(precision, recall, ndcg)
        return results, metrics


class BPRTrainer(BasicTrainer):
    def __init__(self, trainer_config):
        super(BPRTrainer, self).__init__(trainer_config)

        self.dataloader = DataLoader(self.dataset, batch_size=trainer_config['batch_size'],
                                     num_workers=trainer_config['dataloader_num_workers'],
                                     persistent_workers=True)
        self.initialize_optimizer()
        self.l2_reg = trainer_config['l2_reg']

    def train_one_epoch(self):
        losses = AverageMeter()
        for batch_data in self.dataloader:
            inputs = batch_data[:, 0, :].to(device=self.device, dtype=torch.int64)
            users, pos_items, neg_items = inputs[:, 0],  inputs[:, 1],  inputs[:, 2]

            users_r, pos_items_r, neg_items_r, l2_norm_sq = \
                self.model.bpr_forward(users, pos_items, neg_items, self.pp_config)
            pos_scores = torch.sum(users_r * pos_items_r, dim=1)
            neg_scores = torch.sum(users_r * neg_items_r, dim=1)

            bpr_loss = F.softplus(neg_scores - pos_scores).mean()
            reg_loss = self.l2_reg * l2_norm_sq.mean()
            loss = bpr_loss + reg_loss
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
            losses.update(loss.item(), inputs.shape[0])
        return losses.avg


class APRTrainer(BasicTrainer):
    def __init__(self, trainer_config):
        super(APRTrainer, self).__init__(trainer_config)

        self.dataloader = DataLoader(self.dataset, batch_size=trainer_config['batch_size'],
                                     num_workers=trainer_config['dataloader_num_workers'],
                                     persistent_workers=True)
        self.initialize_optimizer()
        self.l2_reg = trainer_config['l2_reg']
        self.adv_reg = trainer_config['adv_reg']
        self.eps = trainer_config['eps']
        self.model.load(trainer_config['ckpt_path'])

    def train_one_epoch(self):
        losses = AverageMeter()
        for batch_data in self.dataloader:
            inputs = batch_data[:, 0, :].to(device=self.device, dtype=torch.int64)
            users, pos_items, neg_items = inputs[:, 0],  inputs[:, 1],  inputs[:, 2]

            users_r, pos_items_r, neg_items_r, l2_norm_sq = \
                self.model.bpr_forward(users, pos_items, neg_items, self.pp_config)
            pos_scores = torch.sum(users_r * pos_items_r, dim=1)
            neg_scores = torch.sum(users_r * neg_items_r, dim=1)
            bpr_loss = F.softplus(neg_scores - pos_scores).mean()

            delta_users_r, delta_pos_items_r, delta_neg_items_r = \
                torch.autograd.grad(bpr_loss, (users_r, pos_items_r, neg_items_r), retain_graph=True)
            delta_users_r = F.normalize(delta_users_r, p=2, dim=1) * self.eps
            delta_pos_items_r = F.normalize(delta_pos_items_r, p=2, dim=1) * self.eps
            delta_neg_items_r = F.normalize(delta_neg_items_r, p=2, dim=1) * self.eps
            pos_scores = torch.sum((users_r + delta_users_r) * (pos_items_r + delta_pos_items_r), dim=1)
            neg_scores = torch.sum((users_r + delta_users_r) * (neg_items_r + delta_neg_items_r), dim=1)
            adv_loss = self.adv_reg * F.softplus(neg_scores - pos_scores).mean()

            reg_loss = self.l2_reg * l2_norm_sq.mean()
            loss = bpr_loss + reg_loss + adv_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.update(loss.item(), inputs.shape[0])
        return losses.avg


class BCETrainer(BasicTrainer):
    def __init__(self, trainer_config):
        super(BCETrainer, self).__init__(trainer_config)

        self.dataloader = DataLoader(self.dataset, batch_size=trainer_config['batch_size'],
                                     num_workers=trainer_config['dataloader_num_workers'],
                                     persistent_workers=True)
        self.initialize_optimizer()
        self.l2_reg = trainer_config['l2_reg']
        if self.model.name == 'NeuMF':
            self.mf_pretrain_epochs = trainer_config['mf_pretrain_epochs']
            self.mlp_pretrain_epochs = trainer_config['mlp_pretrain_epochs']

    def change_arch(self):
        if self.epoch == self.mf_pretrain_epochs and self.model.arch == 'gmf':
            self.model.arch = 'mlp'
            self.initialize_optimizer()
            self.best_ndcg = -np.inf
            self.model.load(self.save_path)
        if self.epoch == self.mf_pretrain_epochs + self.mlp_pretrain_epochs and self.model.arch == 'mlp':
            self.model.arch = 'neumf'
            self.initialize_optimizer()
            self.best_ndcg = -np.inf
            self.model.load(self.save_path)
            self.model.init_mlp_layers()

    def train_one_epoch(self):
        if self.model.name == 'NeuMF':
            self.change_arch()
        losses = AverageMeter()
        for batch_data in self.dataloader:
            inputs = batch_data.to(device=self.device, dtype=torch.int64)
            pos_users, pos_items = inputs[:, 0, 0], inputs[:, 0, 1]
            inputs = inputs.reshape(-1, 3)
            neg_users, neg_items = inputs[:, 0], inputs[:, 2]
            pos_scores, neg_scores, l2_norm_sq = self.model.bce_forward(pos_users, pos_items,
                                                                        neg_users, neg_items, self.pp_config)
            bce_loss_p = F.softplus(-pos_scores)
            bce_loss_n = F.softplus(neg_scores)

            bce_loss = torch.cat([bce_loss_p, bce_loss_n], dim=0).mean()
            reg_loss = self.l2_reg * l2_norm_sq.mean()
            loss = bce_loss + reg_loss
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
            losses.update(loss.item(), l2_norm_sq.shape[0])
        return losses.avg


class MLTrainer(BasicTrainer):
    def __init__(self, trainer_config):
        super(MLTrainer, self).__init__(trainer_config)

        train_user = TensorDataset(torch.arange(self.dataset.n_users, dtype=torch.int64, device=self.device))
        self.train_user_loader = DataLoader(train_user, batch_size=trainer_config['batch_size'], shuffle=True)
        self.data_mat = sp.coo_matrix((np.ones((len(self.dataset.train_array),)), np.array(self.dataset.train_array).T),
                                      shape=(self.dataset.n_users, self.dataset.n_items), dtype=np.float32).tocsr()
        self.initialize_optimizer()
        self.l2_reg = trainer_config['l2_reg']
        self.kl_reg = trainer_config['kl_reg']

    def train_one_epoch(self):
        kl_reg = min(self.kl_reg, 1. * self.epoch / self.n_epochs)

        losses = AverageMeter()
        for users in self.train_user_loader:
            users = users[0]

            scores, kl, l2_norm_sq = self.model.ml_forward(users)
            scores = F.log_softmax(scores, dim=1)
            users = users.cpu().numpy()
            profiles = self.data_mat[users, :]
            profiles = torch.tensor(profiles.toarray(), dtype=torch.float32, device=self.device)
            ml_loss = -torch.sum(profiles * scores, dim=1).mean()

            reg_loss = kl_reg * kl.mean() + self.l2_reg * l2_norm_sq.mean()
            loss = ml_loss + reg_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.update(loss.item(), users.shape[0])
        return losses.avg


class MSETrainer(BasicTrainer):
    def __init__(self, trainer_config):
        super(MSETrainer, self).__init__(trainer_config)

        train_user = TensorDataset(torch.arange(self.dataset.n_users, dtype=torch.int64, device=self.device))
        self.train_user_loader = DataLoader(train_user, batch_size=trainer_config['batch_size'], shuffle=True)
        self.data_mat = sp.coo_matrix((np.ones((len(self.dataset.train_array),)), np.array(self.dataset.train_array).T),
                                      shape=(self.dataset.n_users, self.dataset.n_items), dtype=np.float32).tocsr()
        self.initialize_optimizer()
        self.l2_reg = trainer_config['l2_reg']
        self.weight = trainer_config['weight']
        self.merged_data_mat = None

    def merge_fake_tensor(self, fake_tensor):
        row = torch.nonzero(fake_tensor)[:, 0].cpu().numpy()
        col = torch.nonzero(fake_tensor)[:, 1].cpu().numpy()
        fake_mat = sp.coo_matrix((np.ones((row.shape[0],)), np.vstack((row, col))),
                                 shape=list(fake_tensor.shape), dtype=np.float32).tocsr()
        self.merged_data_mat = sp.vstack([self.data_mat, fake_mat])

    def train_one_epoch(self):
        if self.merged_data_mat is not None:
            data_mat = self.merged_data_mat
        else:
            data_mat = self.data_mat

        losses = AverageMeter()
        for users in self.train_user_loader:
            users = users[0]

            scores = self.model.mse_forward(users, self.pp_config)
            reg_loss = torch.norm(self.model.embedding(users), p=2) ** 2
            reg_loss += torch.norm(self.model.embedding.weight[-self.model.n_items:, :], p=2) ** 2
            users = users.cpu().numpy()
            profiles = data_mat[users, :]
            profiles = torch.tensor(profiles.toarray(), dtype=torch.float32, device=self.device)
            m_loss = mse_loss(profiles, scores, self.weight)

            loss = m_loss + self.l2_reg * reg_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.update(loss.item(), users.shape[0])
        return losses.avg


