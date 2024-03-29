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


class DPA2DLAttacker(BasicAttacker):
    def __init__(self, attacker_config):
        super(DPA2DLAttacker, self).__init__(attacker_config)
        self.surrogate_model_config = attacker_config['surrogate_model_config']
        self.surrogate_trainer_config = attacker_config['surrogate_trainer_config']

        self.reg_u = attacker_config['reg_u']
        self.prob = attacker_config['prob']
        self.kappa = torch.tensor(attacker_config['kappa'], dtype=torch.float32, device=self.device)
        self.step = attacker_config['step']
        self.alpha = attacker_config['alpha']
        self.n_rounds = attacker_config['n_rounds']

        self.target_item_tensor = torch.tensor(self.target_items, dtype=torch.int64, device=self.device)
        non_target_items = [i for i in range(self.n_items) if i not in self.target_items]
        self.non_target_item_tensor = torch.tensor(non_target_items, dtype=torch.int64, device=self.device)
        target_users = TensorDataset(torch.arange(self.n_users, dtype=torch.int64, device=self.device))
        self.target_user_loader = DataLoader(target_users, batch_size=self.surrogate_trainer_config['test_batch_size'],
                                             shuffle=True)

    def get_target_hr(self, surrogate_model):
        surrogate_model.eval()
        with torch.no_grad():
            hrs = AverageMeter()
            for users in self.target_user_loader:
                users = users[0]
                scores = surrogate_model.predict(users)
                _, topk_items = scores.topk(self.topk, dim=1)
                hr = torch.eq(topk_items.unsqueeze(2), self.target_item_tensor.unsqueeze(0).unsqueeze(0))
                hr = hr.float().sum(dim=1).mean()
                hrs.update(hr.item(), users.shape[0])
        return hrs.avg

    def poison_train(self, surrogate_model, surrogate_trainer, temp_fake_user_tensor):
        losses = AverageMeter()
        for users in self.target_user_loader:
            users = users[0]
            scores = surrogate_model.predict(users)
            loss = self.alpha * self.reg_u * topk_loss(scores, self.target_item_tensor, self.topk, self.kappa)
            surrogate_trainer.opt.zero_grad()
            loss.backward()
            surrogate_trainer.opt.step()
            losses.update(loss.item(), users.shape[0] * self.target_items.shape[0])

        scores = surrogate_model.predict(temp_fake_user_tensor)
        scores = scores[:, self.non_target_item_tensor]
        loss = self.alpha * (torch.sigmoid(scores) ** 2).mean()
        surrogate_trainer.opt.zero_grad()
        loss.backward()
        surrogate_trainer.opt.step()
        losses.update(loss.item(), temp_fake_user_tensor.shape[0] * (self.n_items - self.target_items.shape[0]))
        return losses.avg

    def choose_filler_items(self, surrogate_model, temp_fake_user_tensor, prob):
        surrogate_model.eval()
        with torch.no_grad():
            scores = surrogate_model.predict(temp_fake_user_tensor)
        for u_idx in range(temp_fake_user_tensor.shape[0]):
            row_score = torch.sigmoid(scores[u_idx, :]) * prob
            row_score[self.target_item_tensor] = 0.
            filler_items = row_score.topk(self.n_inters - self.target_items.shape[0]).indices
            prob[filler_items] *= self.prob
            if (prob < 1.0).all():
                prob[:] = 1.

            f_u = temp_fake_user_tensor[u_idx]
            filler_items = filler_items.cpu().numpy()
            self.fake_users[f_u - self.n_users, filler_items] = 1.

            filler_items = set(filler_items.tolist())
            self.dataset.train_data[f_u] |= filler_items
            self.dataset.train_array += [[f_u, item] for item in filler_items]

    def save_surrogate(self, surrogate_trainer, best_hr):
        if surrogate_trainer.save_path:
            os.remove(surrogate_trainer.save_path)
        surrogate_trainer.save_path = os.path.join('checkpoints', 'DPA2DL_{:s}_{:s}_{:.3f}.pth'.
                                                   format(self.dataset.name, surrogate_trainer.model.name, best_hr))
        surrogate_trainer.model.save(surrogate_trainer.save_path)
        print('Maximal hit ratio, save poisoned model to {:s}.'.format(surrogate_trainer.save_path))

    def generate_fake_users(self, verbose=True, writer=None):
        self.fake_users = np.zeros([self.n_fakes, self.n_items], dtype=np.float32)
        self.fake_users[:, self.target_items] = 1.

        prob = torch.ones(self.n_items, dtype=torch.float32, device=self.device)
        fake_user_end_indices = list(np.arange(0, self.n_fakes, self.step, dtype=np.int64)) + [self.n_fakes]
        for i_step in range(1, len(fake_user_end_indices)):
            step_start_time = time.time()
            fake_nums_str = '{}-{}'.format(fake_user_end_indices[i_step - 1], fake_user_end_indices[i_step])
            print('Start generating poison #{:s} !'.format(fake_nums_str))

            temp_fake_user_tensor = np.arange(fake_user_end_indices[i_step - 1], fake_user_end_indices[i_step]) + self.n_users
            temp_fake_user_tensor = torch.tensor(temp_fake_user_tensor, dtype=torch.int64, device=self.device)
            n_temp_fakes = temp_fake_user_tensor.shape[0]
            self.dataset.train_data += [set(self.target_items) for _ in range(n_temp_fakes)]
            self.dataset.val_data += [{} for _ in range(n_temp_fakes)]
            self.dataset.train_array += [[fake_u, item] for item in self.target_items for fake_u in temp_fake_user_tensor]
            self.dataset.n_users += n_temp_fakes

            surrogate_model = get_model(self.surrogate_model_config, self.dataset)
            surrogate_trainer = get_trainer(self.surrogate_trainer_config, surrogate_model)

            start_time = time.time()
            surrogate_trainer.train(verbose=False)
            consumed_time = time.time() - start_time
            self.retrain_time += consumed_time

            best_hr = self.get_target_hr(surrogate_model)
            print('Initial target HR: {:.4f}'.format(best_hr))
            self.save_surrogate(surrogate_trainer, best_hr)
            for i_round in range(self.n_rounds):
                surrogate_model.train()
                p_loss = self.poison_train(surrogate_model, surrogate_trainer, temp_fake_user_tensor)

                start_time = time.time()
                t_loss = surrogate_trainer.train_one_epoch()
                consumed_time = time.time() - start_time
                self.retrain_time += consumed_time

                target_hr = self.get_target_hr(surrogate_model)
                if verbose:
                    print('Round {:d}/{:d}, Poison Loss: {:.6f}, Train Loss: {:.6f}, Target Hit Ratio {:.6f}%'.
                          format(i_round, self.n_rounds, p_loss, t_loss, target_hr * 100.))
                if writer:
                    writer.add_scalar('{:s}_{:s}/Poison_Loss'.format(self.name, fake_nums_str), p_loss, i_round)
                    writer.add_scalar('{:s}_{:s}/Train_Loss'.format(self.name, fake_nums_str), t_loss, i_round)
                    writer.add_scalar('{:s}_{:s}/Hit_Ratio@{:d}'.format(self.name, fake_nums_str, self.topk),
                                      target_hr, i_round)
                if target_hr > best_hr:
                    os.remove(surrogate_trainer.save_path)
                    best_hr = target_hr
                    self.save_surrogate(surrogate_trainer, best_hr)

            surrogate_model.load(surrogate_trainer.save_path)
            os.remove(surrogate_trainer.save_path)

            self.choose_filler_items(surrogate_model, temp_fake_user_tensor, prob)
            print('Poison #{:s} has been generated!'.format(fake_nums_str))
            consumed_time = time.time() - step_start_time
            self.consumed_time += consumed_time
            gc.collect()
            torch.cuda.empty_cache()

        self.dataset.train_data = self.dataset.train_data[:-self.n_fakes]
        self.dataset.val_data = self.dataset.val_data[:-self.n_fakes]
        self.dataset.train_array = self.dataset.train_array[:-self.n_fakes * self.n_inters]
        self.dataset.n_users -= self.n_fakes
