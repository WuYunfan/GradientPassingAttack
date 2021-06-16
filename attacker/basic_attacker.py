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
        self.target_item = attacker_config['target_item']
        self.device = attacker_config['device']
        self.fake_users = None
        self.model = None
        self.trainer = None

    def generate_fake_users(self, verbose=True, writer=None):
        raise NotImplementedError

    def eval(self, model_config, trainer_config, verbose=True, writer=None, retrain=True):
        if self.dataset.attack_data is None:
            self.dataset.attack_data = [[] for _ in range(self.n_users)]
            for u in range(self.n_users):
                if self.target_item not in self.dataset.train_data[u]:
                    self.dataset.attack_data[u].append(self.target_item)

            for fake_u in range(self.n_fakes):
                items = np.nonzero(self.fake_users[fake_u, :])[0].tolist()
                self.dataset.train_data.append(items)
                self.dataset.val_data.append([])
                self.dataset.attack_data.append([])
                self.dataset.train_array.extend([[fake_u + self.n_users, item] for item in items])
            self.dataset.n_users += self.n_fakes

        if self.model is None or retrain:
            self.model = get_model(model_config, self.dataset)
            self.trainer = get_trainer(trainer_config, self.dataset, self.model)
            self.trainer.train(verbose=verbose, writer=writer)
        _, metrics = self.trainer.eval('attack')

        if verbose:
            hit_ratio = ''
            ndcg = ''
            for k in self.trainer.topks:
                hit_ratio += '{:.3f}%@{:d}, '.format(metrics['Recall'][k] * 100, k)
                ndcg += '{:.3f}%@{:d}, '.format(metrics['NDCG'][k] * 100, k)
            results = 'Hit Ratio: {:s}NDCG: {:s}'.format(hit_ratio, ndcg)
            print('Attack result. {:s}'.format(results))

        ndcg = metrics['NDCG'][self.trainer.topks[0]]
        return ndcg


class PoisonedDataset(Dataset):
    def __init__(self, data_mat, fake_tensor, device):
        self.data_mat = data_mat
        self.fake_tensor = fake_tensor
        self.device = device
        self.n_users = data_mat.shape[0]
        self.n_fakes = fake_tensor.shape[0]

    def __len__(self):
        return self.n_users + self.n_fakes

    def __getitem__(self, index):
        if index < self.n_users:
            return index, \
                   torch.tensor(self.data_mat[index, :].toarray().squeeze(), dtype=torch.float32, device=self.device)
        return index, self.fake_tensor[index - self.n_users, :]




