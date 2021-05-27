import torch
import sys
from dataset import ML1MDataset
from model import MF, LightGCN
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, SGD
from torch.optim.adagrad import Adagrad
import time
import numpy as np
import os
from utils import AverageMeter
import torch.nn.functional as F


def get_trainer(config, dataset, model):
    config = config.copy()
    config['dataset'] = dataset
    config['model'] = model
    trainer = getattr(sys.modules['trainer'], config['name'])
    trainer = trainer(config)
    return trainer


class BasicTrainer:
    def __init__(self, trainer_config):
        print(trainer_config)
        self.config = trainer_config
        self.name = trainer_config['name']
        self.dataset = trainer_config['dataset']
        self.model = trainer_config['model']
        self.topks = trainer_config['topks']
        self.device = trainer_config['device']
        self.n_epochs = trainer_config['n_epochs']
        self.max_patience = trainer_config.get('max_patience', 50)
        self.val_interval = trainer_config.get('val_interval', 1)
        self.epoch = 0
        self.best_ndcg = -np.inf
        self.save_path = None

        test_user = TensorDataset(torch.arange(self.dataset.n_users, dtype=torch.int64, device=self.device))
        self.test_user_loader = DataLoader(test_user, batch_size=trainer_config['test_batch_size'])

    def train_one_epoch(self):
        raise NotImplementedError

    def record(self, writer, stage, metrics):
        for metric in metrics:
            for k in self.topks:
                writer.add_scalar('{:s}_{:s}/{:s}_{:s}@{:d}'
                                  .format(self.model.name, self.name, stage, metric, k)
                                  , metrics[metric][k], self.epoch)

    def train(self, verbose=True, writer=None):
        if not self.model.trainable:
            results, metrics = self.eval('val')
            if verbose:
                print('Validation result. {:s}'.format(results))
            ndcg = metrics['NDCG'][self.topks[0]]
            return ndcg

        if not os.path.exists('checkpoints'): os.mkdir('checkpoints')
        patience = self.max_patience
        for self.epoch in range(self.n_epochs):
            start_time = time.time()
            self.model.train()
            loss = self.train_one_epoch()
            _, metrics = self.eval('train')
            consumed_time = time.time() - start_time
            if verbose:
                print('Epoch {:d}/{:d}, Loss: {:.6f}, Time: {:.3f}s'
                      .format(self.epoch, self.n_epochs, loss, consumed_time))
            if writer:
                writer.add_scalar('{:s}_{:s}/train_loss'.format(self.model.name, self.name), loss, self.epoch)
                self.record(writer, 'train', metrics)

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
                print('Best NDCG, save model to {:s}'.format(self.save_path))
            else:
                patience -= self.val_interval
                if patience <= 0:
                    print('Early stopping!')
                    break
        self.model.load(self.save_path)
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

    def get_rec_items(self, train_or_val):
        self.model.eval()
        rec_items = []
        with torch.no_grad():
            for users in self.test_user_loader:
                users = users[0]
                scores = self.model.predict(users)

                if train_or_val != 'train':
                    users = users.cpu().numpy().tolist()
                    exclude_user_indexes = []
                    exclude_items = []
                    for user_idx, user in enumerate(users):
                        items = self.dataset.train_data[user]
                        exclude_user_indexes.extend([user_idx] * len(items))
                        exclude_items.extend(items)
                    scores[exclude_user_indexes, exclude_items] = -np.inf

                _, items = torch.topk(scores, k=max(self.topks))
                rec_items.append(items.cpu().numpy())

        rec_items = np.concatenate(rec_items, axis=0)
        return rec_items

    def eval(self, train_or_val):
        eval_data = getattr(self.dataset, train_or_val + '_data')
        rec_items = self.get_rec_items(train_or_val)
        metrics = self.calculate_metrics(eval_data, rec_items)

        precison = ''
        recall = ''
        ndcg = ''
        for k in self.topks:
            precison += '{:.3f}%@{:d}, '.format(metrics['Precision'][k] * 100., k)
            recall += '{:.3f}%@{:d}, '.format(metrics['Recall'][k] * 100., k)
            ndcg += '{:.3f}%@{:d}, '.format(metrics['NDCG'][k] * 100., k)
        results = 'Precision: {:s}Recall: {:s}NDCG: {:s}'.format(precison, recall, ndcg)
        return results, metrics


class BPRTrainer(BasicTrainer):
    def __init__(self, trainer_config):
        super(BPRTrainer, self).__init__(trainer_config)

        self.dataloader = DataLoader(self.dataset, batch_size=trainer_config['batch_size'],
                                     num_workers=trainer_config['dataloader_num_workers'])
        self.opt = getattr(sys.modules[__name__], trainer_config['optimizer'])
        self.opt = self.opt(self.model.parameters(), lr=trainer_config['lr'])
        self.l2_reg = trainer_config['l2_reg']

    def train_one_epoch(self):
        losses = AverageMeter()
        for batch_data in self.dataloader:
            inputs = batch_data[:, 0, :].to(device=self.device, dtype=torch.int64)
            users, pos_items, neg_items = inputs[:, 0],  inputs[:, 1],  inputs[:, 2]

            users_r, pos_items_r, neg_items_r, l2_norm_sq = self.model.bpr_forward(users, pos_items, neg_items)
            pos_scores = torch.sum(users_r * pos_items_r, dim=1)
            neg_scores = torch.sum(users_r * neg_items_r, dim=1)

            bpr_loss = F.softplus(neg_scores - pos_scores).mean()
            reg_loss = self.l2_reg * l2_norm_sq.mean()
            loss = bpr_loss + reg_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.update(loss.item(), inputs.shape[0])
        return losses.avg


class BCETrainer(BasicTrainer):
    def __init__(self, trainer_config):
        super(BCETrainer, self).__init__(trainer_config)

        self.dataloader = DataLoader(self.dataset, batch_size=trainer_config['batch_size'],
                                     num_workers=trainer_config['dataloader_num_workers'])
        self.opt = getattr(sys.modules[__name__], trainer_config['optimizer'])
        self.opt = self.opt(self.model.parameters(), lr=trainer_config['lr'])
        self.l2_reg = trainer_config['l2_reg']
        self.mf_pretrain_epochs = trainer_config['mf_pretrain_epochs']
        self.mlp_pretrain_epochs = trainer_config['mlp_pretrain_epochs']

    def train_one_epoch(self):
        if self.epoch == self.mf_pretrain_epochs:
            self.model.arch = 'mlp'
            self.best_ndcg = -np.inf
            self.model.load(self.save_path)
        if self.epoch == self.mf_pretrain_epochs + self.mlp_pretrain_epochs:
            self.model.arch = 'neumf'
            self.opt = getattr(sys.modules[__name__], self.config['optimizer'])
            self.opt = self.opt(self.model.parameters(), lr=self.config['lr'])
            self.best_ndcg = -np.inf
            self.model.load(self.save_path)
            self.model.init_mlp_layers()
        losses = AverageMeter()
        for batch_data in self.dataloader:
            inputs = batch_data.to(device=self.device, dtype=torch.int64)
            users, pos_items = inputs[:, 0, 0], inputs[:, 0, 1]
            logits, l2_norm_sq_p = self.model.bce_forward(users, pos_items)
            bce_loss_p = F.softplus(-logits)

            inputs = inputs.reshape(-1, 3)
            users, neg_items = inputs[:, 0], inputs[:, 2]
            logits, l2_norm_sq_n = self.model.bce_forward(users, neg_items)
            bce_loss_n = F.softplus(logits)

            bce_loss = torch.cat([bce_loss_p, bce_loss_n], dim=0).mean()
            l2_norm_sq = torch.cat([l2_norm_sq_p, l2_norm_sq_n], dim=0)
            reg_loss = self.l2_reg * l2_norm_sq.mean()
            loss = bce_loss + reg_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.update(loss.item(), l2_norm_sq.shape[0])
        return losses.avg


class APRTrainer(BasicTrainer):
    def __init__(self, trainer_config):
        super(APRTrainer, self).__init__(trainer_config)

        self.dataloader = DataLoader(self.dataset, batch_size=trainer_config['batch_size'],
                                     num_workers=trainer_config['dataloader_num_workers'])
        self.opt = getattr(sys.modules[__name__], trainer_config['optimizer'])
        self.opt = self.opt(self.model.parameters(), lr=trainer_config['lr'])
        self.l2_reg = trainer_config['l2_reg']
        self.adv_reg = trainer_config['adv_reg']
        self.eps = trainer_config['eps']
        self.model.load(trainer_config['ckpt_path'])

    def train_one_epoch(self):
        losses = AverageMeter()
        for batch_data in self.dataloader:
            inputs = batch_data[:, 0, :].to(device=self.device, dtype=torch.int64)
            users, pos_items, neg_items = inputs[:, 0],  inputs[:, 1],  inputs[:, 2]

            users_r, pos_items_r, neg_items_r, l2_norm_sq = self.model.bpr_forward(users, pos_items, neg_items)
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



