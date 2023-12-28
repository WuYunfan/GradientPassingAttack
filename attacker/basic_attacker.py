import random
import numpy as np
from model import get_model
from trainer import get_trainer
from torch.utils.data import Dataset
import torch


class BasicAttacker:
    def __init__(self, attacker_config):
        print(attacker_config)
        self.config = attacker_config
        self.name = attacker_config['name']
        self.dataset = attacker_config['dataset']
        self.n_users = self.dataset.n_users
        self.n_items = self.dataset.n_items
        self.n_fakes = attacker_config['n_fakes']
        self.n_inters = attacker_config['n_inters']
        self.n_train_inters = int(self.n_inters * attacker_config.get('train_ratio', 0.8))
        self.target_items = np.array(attacker_config['target_items'])
        self.device = attacker_config['device']
        self.topk = attacker_config['topk']
        self.fake_users = None
        self.model = None
        self.trainer = None
        self.consumed_time = 0.
        self.retrain_time = 0.

    def generate_fake_users(self, verbose=True, writer=None):
        self.fake_users = np.zeros([self.n_fakes, self.n_items], dtype=np.float32)

    def eval(self, model_config, trainer_config, verbose=True, writer=None, retrain=True):
        if self.dataset.attack_data is None:
            self.dataset.attack_data = [[] for _ in range(self.n_users)]
            for u in range(self.n_users):
                for item in self.target_items:
                    if item not in self.dataset.train_data[u]:
                        self.dataset.attack_data[u].append(item)

            for fake_u in range(self.n_fakes):
                items = np.where(self.fake_users[fake_u, :] > 0.5)[0].tolist()
                assert len(items) <= self.n_inters
                random.shuffle(items)
                train_items = items[:self.n_train_inters]
                val_items = items[self.n_train_inters:]
                self.dataset.train_data.append(set(train_items))
                self.dataset.val_data.append(set(val_items))
                self.dataset.attack_data.append([])
                self.dataset.train_array.extend([[fake_u + self.n_users, item] for item in train_items])
            self.dataset.n_users += self.n_fakes

        if self.model is None or retrain:
            self.model = get_model(model_config, self.dataset)
            self.trainer = get_trainer(trainer_config, self.model)
            self.trainer.train(verbose=False, writer=writer)
        _, metrics = self.trainer.eval('attack')

        if verbose:
            hit_ratio = ''
            ndcg = ''
            hit_one = ''
            for k in self.trainer.topks:
                hit_ratio += '{:.3f}%@{:d}, '.format(metrics['Recall'][k] * 100, k)
                ndcg += '{:.3f}%@{:d}, '.format(metrics['NDCG'][k] * 100, k)
                hit_one += '{:.3f}%@{:d}, '.format(metrics['HitOne'][k] * 100, k)
            results = 'Hit Ratio: {:s}NDCG: {:s}Hit One: {:s}'.format(hit_ratio, ndcg, hit_one)
            print('Target Items: {:s}'.format(str(self.target_items)))
            print('Attack result. {:s}'.format(results))
            print('Consumed time: {:.3f}s, retrain time: {:.3f}'.format(self.consumed_time, self.retrain_time))

        hr = metrics['Recall'][self.trainer.topks[0]]
        return hr







