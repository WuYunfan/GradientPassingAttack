import gc
import random
import torch
from torch.utils.data import TensorDataset, DataLoader
from attacker.basic_attacker import BasicAttacker
import numpy as np
from model import get_model
from trainer import get_trainer
from utils import AverageMeter, topk_loss
import torch.nn.functional as F
import time
import os


class DPA2DL(BasicAttacker):
    def __init__(self, attacker_config):
        super(DPA2DL, self).__init__(attacker_config)
        self.surrogate_model_config = attacker_config['surrogate_model_config']
        self.surrogate_trainer_config = attacker_config['surrogate_trainer_config']

        self.reg_u = attacker_config['reg_u']
        self.prob = attacker_config['prob']
        self.kappa = torch.tensor(attacker_config['kappa'], dtype=torch.float32, device=self.device)
        self.step = attacker_config['step']
        self.alpha = attacker_config['alpha']
        self.n_rounds = attacker_config['n_rounds']
        self.bernoulli_p = attacker_config.get('bernoulli_p', 0.)
        self.pre_train_weights = None

        target_users = [user for user in range(self.n_users) if self.target_item not in self.dataset.train_data[user]]
        target_users = TensorDataset(torch.tensor(target_users, dtype=torch.int64, device=self.device))
        self.target_user_loader = DataLoader(target_users, batch_size=self.surrogate_trainer_config['test_batch_size'],
                                             shuffle=True)

    def get_target_hr(self, surrogate_model):
        surrogate_model.eval()
        with torch.no_grad():
            scores = []
            for users in self.target_user_loader:
                users = users[0]
                scores.append(surrogate_model.predict(users))
            scores = torch.cat(scores, dim=0)
            _, topk_items = scores.topk(self.topk, dim=1)
            hr = torch.eq(topk_items, self.target_item).float().sum(dim=1).mean()
        return hr.item()

    def poison_train(self, surrogate_model, surrogate_trainer, temp_fake_users):
        losses = AverageMeter()
        for users in self.target_user_loader:
            users = users[0]
            scores = surrogate_model.mse_forward(users, surrogate_trainer.pp_config)
            loss = self.alpha * self.reg_u * topk_loss(scores, self.target_item, self.topk, self.kappa)
            surrogate_trainer.opt.zero_grad()
            loss.backward()
            surrogate_trainer.opt.step()
            losses.update(loss.item(), users.shape[0])

        scores = surrogate_model.mse_forward(torch.tensor(temp_fake_users, dtype=torch.int64, device=self.device),
                                             surrogate_trainer.pp_config)
        scores = torch.cat([scores[:, :self.target_item], scores[:, self.target_item + 1:]], dim=1)
        loss = self.alpha * (torch.sigmoid(scores) ** 2).mean()
        surrogate_trainer.opt.zero_grad()
        loss.backward()
        surrogate_trainer.opt.step()
        losses.update(loss.item(), temp_fake_users.shape[0] * (self.n_items - 1))
        return losses.avg

    def choose_filler_items(self, surrogate_model, temp_fake_users, prob):
        surrogate_model.eval()
        with torch.no_grad():
            scores = surrogate_model.predict(torch.tensor(temp_fake_users, dtype=torch.int64, device=self.device))
        for u_idx in range(temp_fake_users.shape[0]):
            row_score = torch.sigmoid(scores[u_idx, :]) * prob
            row_score[self.target_item] = 0.
            filler_items = row_score.topk(self.n_inters - 1).indices
            prob[filler_items] *= self.prob
            if (prob < 1.0).all():
                prob[:] = 1.

            f_u = temp_fake_users[u_idx]
            filler_items = filler_items.cpu().numpy()
            self.fake_users[f_u - self.n_users, filler_items] = 1.

            filler_items = filler_items.tolist()
            self.dataset.train_data[f_u] += filler_items
            self.dataset.train_array += [[f_u, item] for item in filler_items]

    def save_surrogate(self, surrogate_trainer, best_hr):
        surrogate_trainer.save_path = os.path.join('checkpoints', 'DPA2DL_{:s}_{:s}_{:.3f}.pth'.
                                                   format(self.dataset.name, surrogate_trainer.model.name, best_hr))
        surrogate_trainer.model.save(surrogate_trainer.save_path)
        print('Maximal hit ratio, save poisoned model to {:s}.'.format(surrogate_trainer.save_path))

    def generate_fake_users(self, verbose=True, writer=None):
        self.fake_users = np.zeros([self.n_fakes, self.n_items], dtype=np.float32)
        self.fake_users[:, self.target_item] = 1.

        prob = torch.ones(self.n_items, dtype=torch.float32, device=self.device)
        fake_user_end_indices = list(np.arange(0, self.n_fakes, self.step, dtype=np.int64)) + [self.n_fakes]
        for i_step in range(1, len(fake_user_end_indices)):
            step_start_time = time.time()
            fake_nums_str = '{}-{}'.format(fake_user_end_indices[i_step - 1], fake_user_end_indices[i_step])
            print('Start generating poison #{:s} !'.format(fake_nums_str))

            temp_fake_users = np.arange(fake_user_end_indices[i_step - 1], fake_user_end_indices[i_step]) + self.n_users
            n_temp_fakes = temp_fake_users.shape[0]
            self.dataset.train_data += [[self.target_item]] * n_temp_fakes
            self.dataset.val_data += [[]] * n_temp_fakes
            self.dataset.train_array += [[fake_u, self.target_item] for fake_u in temp_fake_users]
            self.dataset.n_users += n_temp_fakes

            surrogate_model = get_model(self.surrogate_model_config, self.dataset)
            if self.pre_train_weights is not None:
                with torch.no_grad():
                    pre_train_weights = torch.clone(surrogate_model.embedding.weight)
                    pre_train_weights.data[:-self.n_items - n_temp_fakes, :] = self.pre_train_weights[:-self.n_items, :]
                    pre_train_weights.data[-self.n_items:, :] = self.pre_train_weights[-self.n_items:, :]
                    prob = torch.full(pre_train_weights.shape, self.bernoulli_p, device=self.device)
                    mask = torch.bernoulli(prob)
                    surrogate_model.embedding.weight.data = \
                        pre_train_weights * mask + surrogate_model.embedding.weight * (1 - mask)
            surrogate_trainer = get_trainer(self.surrogate_trainer_config, surrogate_model)

            start_time = time.time()
            surrogate_trainer.train(verbose=False)
            consumed_time = time.time() - start_time
            self.retrain_time += consumed_time
            if self.bernoulli_p > 0.:
                self.pre_train_weights = torch.clone(surrogate_model.embedding.weight.detach())

            best_hr = self.get_target_hr(surrogate_model)
            print('Initial target HR: {:.4f}'.format(best_hr))
            self.save_surrogate(surrogate_trainer, best_hr)
            for i_round in range(self.n_rounds):
                surrogate_model.train()
                p_loss = self.poison_train(surrogate_model, surrogate_trainer, temp_fake_users)
                t_loss = surrogate_trainer.train_one_epoch()
                target_hr = self.get_target_hr(surrogate_model)
                if verbose:
                    print('Round {:d}/{:d}, Poison Loss: {:.6f}, Train Loss: {:.6f}, Target Hit Ratio {:.6f}'.
                          format(i_round, self.n_rounds, p_loss, t_loss, target_hr))
                if writer:
                    writer.add_scalar('{:s}_{.s}/Poison_Loss'.format(self.name, fake_nums_str), p_loss, i_round)
                    writer.add_scalar('{:s}_{.s}/Train_Loss'.format(self.name, fake_nums_str), t_loss, i_round)
                    writer.add_scalar('{:s}_{.s}/Hit_Ratio@{:d}'.format(self.name, fake_nums_str, self.topk),
                                      target_hr, i_round)
                if target_hr > best_hr:
                    os.remove(surrogate_trainer.save_path)
                    best_hr = target_hr
                    self.save_surrogate(surrogate_trainer, best_hr)

            surrogate_model.load(surrogate_trainer.save_path)
            os.remove(surrogate_trainer.save_path)

            self.choose_filler_items(surrogate_model, temp_fake_users, prob)
            print('Poison #{:s} has been generated!'.format(fake_nums_str))
            consumed_time = time.time() - step_start_time
            self.consumed_time += consumed_time
            torch.cuda.empty_cache()
            gc.collect()

        self.dataset.train_data = self.dataset.train_data[:-self.n_fakes]
        self.dataset.val_data = self.dataset.val_data[:-self.n_fakes]
        self.dataset.train_array = self.dataset.train_array[:-self.n_fakes * self.n_train_inters]
        self.dataset.n_users -= self.n_fakes
