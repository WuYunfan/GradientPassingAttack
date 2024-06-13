import gc
import random
import torch
from torch.utils.data import TensorDataset, DataLoader
from attacker.basic_attacker import BasicAttacker
import numpy as np
from model import get_model
from trainer import get_trainer
from utils import AverageMeter, topk_loss, initial_parameter
import torch.nn.functional as F
import time
from dataset import BiasedSampledDataset
import os
from utils import PartialDataLoader
import scipy.sparse as sp
from attacker.dpa2dl_attacker import DPA2DLAttacker


class RAPURAttacker(BasicAttacker):
    def __init__(self, attacker_config):
        super(RAPURAttacker, self).__init__(attacker_config)
        self.surrogate_model_config = attacker_config['surrogate_model_config']
        self.surrogate_trainer_config = attacker_config['surrogate_trainer_config']

        self.step = attacker_config['step']

        n_top_items = int(self.n_items * attacker_config['top_rate'])
        data_mat = sp.coo_matrix((np.ones((len(self.dataset.train_array),)), np.array(self.dataset.train_array).T),
                                 shape=(self.n_users, self.n_items), dtype=np.float32).tocsr()
        item_popularity = np.array(np.sum(data_mat, axis=0)).squeeze()
        popularity_rank = np.argsort(item_popularity)[::-1].copy()
        popular_items = popularity_rank[:n_top_items]
        self.popular_candidate_tensor = torch.tensor(list(set(popular_items) - set(self.target_items)),
                                                     dtype=torch.int64, device=self.device)
        self.target_item_tensor = torch.tensor(self.target_items, dtype=torch.int64, device=self.device)
        self.target_user_tensor = torch.arange(self.n_users, dtype=torch.int64, device=self.device)
        self.target_user_loader = DataLoader(TensorDataset(self.target_user_tensor),
                                             batch_size=self.surrogate_trainer_config['test_batch_size'], shuffle=False)

    def get_target_hr(self, surrogate_model):
        return DPA2DLAttacker.get_target_hr(self, surrogate_model)

    def choose_filler_items(self, surrogate_model, temp_fake_user_array):
        surrogate_model.eval()
        with torch.no_grad():
            rep = surrogate_model.get_rep()

        epsilon = rep[self.target_user_tensor, :].mean(dim=0)
        candidate_reps = rep[surrogate_model.n_users + self.popular_candidate_tensor, :]
        filler_items_4target = []
        for target_item in self.target_items:
            target_item_rep = rep[surrogate_model.n_users + target_item, :]
            p_m = (epsilon + target_item_rep).unsqueeze(0)
            candidate_scores = torch.mm(p_m, candidate_reps.t()).squeeze()

            _, filler_items = torch.topk(candidate_scores, self.n_inters - self.target_items.shape[0], dim=0)
            filler_items = self.popular_candidate_tensor[filler_items]
            filler_items = filler_items.cpu().numpy()
            filler_items = np.concatenate([filler_items, self.target_items], axis=0)
            filler_items_4target.append(filler_items)

        for fake_u in temp_fake_user_array:
            filler_items = filler_items_4target[fake_u % self.target_items.shape[0]]
            self.dataset.train_data.append(set(filler_items))
            self.dataset.val_data.append({})
            self.dataset.train_array += [[fake_u, item] for item in filler_items]
            self.dataset.n_users += 1
            self.fake_users[fake_u - self.n_users, filler_items] = 1.

    def generate_fake_users(self, verbose=True, writer=None):
        self.fake_users = np.zeros([self.n_fakes, self.n_items], dtype=np.float32)
        fake_user_end_indices = list(np.arange(0, self.n_fakes, self.step, dtype=np.int64)) + [self.n_fakes]
        for i_step in range(1, len(fake_user_end_indices)):
            step_start_time = time.time()
            fake_nums_str = '{}-{}'.format(fake_user_end_indices[i_step - 1], fake_user_end_indices[i_step])
            print('Start generating poison #{:s} !'.format(fake_nums_str))
            temp_fake_user_array = np.arange(fake_user_end_indices[i_step - 1],
                                             fake_user_end_indices[i_step]) + self.n_users

            surrogate_model = get_model(self.surrogate_model_config, self.dataset)
            surrogate_trainer = get_trainer(self.surrogate_trainer_config, surrogate_model)

            start_time = time.time()
            surrogate_trainer.train(verbose=verbose)
            consumed_time = time.time() - start_time
            self.retrain_time += consumed_time
            os.remove(surrogate_trainer.save_path)

            target_hr = self.get_target_hr(surrogate_model)
            if verbose:
                print('Target Hit Ratio {:.6f}%'. format(target_hr * 100.))
            if writer:
                writer.add_scalar('{:s}/Hit_Ratio@{:d}'.format(self.name, self.topk), target_hr, i_step - 1)

            self.choose_filler_items(surrogate_model, temp_fake_user_array)
            print('Poison #{:s} has been generated!'.format(fake_nums_str))
            consumed_time = time.time() - step_start_time
            self.consumed_time += consumed_time
            gc.collect()
            torch.cuda.empty_cache()

        self.dataset.train_data = self.dataset.train_data[:-self.n_fakes]
        self.dataset.val_data = self.dataset.val_data[:-self.n_fakes]
        self.dataset.train_array = self.dataset.train_array[:-self.n_fakes * self.n_inters]
        self.dataset.n_users -= self.n_fakes
