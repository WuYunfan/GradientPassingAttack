import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
import scipy.sparse as sp
import numpy as np
import torch.nn as nn
from utils import mse_loss, ce_loss
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import higher
import time
from attacker.basic_attacker import BasicAttacker
from torch.nn.init import normal_
import gc


class SurrogateWRMF(nn.Module):
    def __init__(self, config):
        super(SurrogateWRMF, self).__init__()
        self.embedding_size = config['embedding_size']
        self.n_users = config['n_users']
        self.n_items = config['n_items']
        self.device = config['device']
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        normal_(self.user_embedding.weight, std=0.1)
        normal_(self.item_embedding.weight, std=0.1)
        self.to(device=self.device)

    def forward(self, users):
        user_e = self.user_embedding(users)
        scores = torch.mm(user_e, self.item_embedding.weight.t())
        return scores


class WRMFSGD(BasicAttacker):
    def __init__(self, attacker_config):
        super(WRMFSGD, self).__init__(attacker_config)
        self.surrogate_config = attacker_config['surrogate_config']
        self.surrogate_config['device'] = self.device
        self.surrogate_config['n_users'] = self.n_users + self.n_fakes
        self.surrogate_config['n_items'] = self.n_items

        self.adv_epochs = attacker_config['adv_epochs']
        self.train_epochs = attacker_config['train_epochs']
        self.unroll_steps = attacker_config['unroll_steps']
        self.weight = attacker_config['weight']
        self.initial_lr = attacker_config['lr']
        self.momentum = attacker_config['momentum']

        self.data_mat = sp.coo_matrix((np.ones((len(self.dataset.train_array),)), np.array(self.dataset.train_array).T),
                                      shape=(self.n_users, self.n_items), dtype=np.float32).tocsr()
        self.fake_tensor = self.init_fake_tensor()
        self.adv_opt = SGD([self.fake_tensor], lr=self.initial_lr, momentum=self.momentum)
        self.scheduler = StepLR(self.adv_opt, step_size=self.adv_epochs / 3, gamma=0.1)

        self.data_tensor = torch.tensor(self.data_mat.toarray(), dtype=torch.float32, device=self.device)
        test_users = TensorDataset(torch.arange(self.n_users + self.n_fakes, dtype=torch.int64, device=self.device))
        self.user_loader = DataLoader(test_users, batch_size=self.surrogate_config['batch_size'],
                                      shuffle=True)
        target_users = [user for user in range(self.n_users) if self.target_item not in self.dataset.train_data[user]]
        self.target_users = torch.tensor(target_users, dtype=torch.int64, device=self.device)

    def init_fake_tensor(self):
        degree = np.array(np.sum(self.data_mat, axis=1)).squeeze()
        qualified_users = self.data_mat[degree <= self.n_inters, :]
        sample_idx = np.random.choice(qualified_users.shape[0], self.n_fakes, replace=False)
        fake_data = qualified_users[sample_idx, :].toarray()
        fake_data = torch.tensor(fake_data, dtype=torch.float32, device=self.device, requires_grad=True)
        return fake_data

    def project_fake_tensor(self):
        with torch.no_grad():
            _, items = self.fake_tensor.topk(self.n_inters, dim=1)
            self.fake_tensor.zero_()
            self.fake_tensor.data = torch.scatter(self.fake_tensor, 1, items, 1.)

    def retrain_surrogate(self):
        surrogate_model = SurrogateWRMF(self.surrogate_config)
        surrogate_model.train()
        train_opt = Adam(surrogate_model.parameters(), lr=self.surrogate_config['lr'],
                         weight_decay=self.surrogate_config['l2_reg'])
        poisoned_data_mat = torch.cat([self.data_tensor, self.fake_tensor], dim=0)

        for _ in range(self.train_epochs - self.unroll_steps):
            for users in self.user_loader:
                users = users[0]
                batch_data = poisoned_data_mat[users, :]
                scores = surrogate_model.forward(users)
                loss = mse_loss(batch_data, scores, self.weight)
                train_opt.zero_grad()
                loss.backward()
                train_opt.step()

        with higher.innerloop_ctx(surrogate_model, train_opt) as (fmodel, diffopt):
            fmodel.train()
            for _ in range(self.unroll_steps):
                for users in self.user_loader:
                    users = users[0]
                    batch_data = poisoned_data_mat[users, :]
                    scores = fmodel.forward(users)
                    loss = mse_loss(batch_data, scores, self.weight)
                    diffopt.step(loss)

            fmodel.eval()
            scores = fmodel.forward(self.target_users)
            adv_loss = ce_loss(scores, self.target_item)
            adv_grads = torch.autograd.grad(adv_loss, self.fake_tensor)[0]
            _, topk_items = scores.topk(self.topk, dim=1)
            hr = torch.eq(topk_items, self.target_item).float().sum(dim=1).mean()
        gc.collect()
        return adv_loss.item(), hr.item(), adv_grads

    def generate_fake_users(self, verbose=True, writer=None):
        min_loss = np.inf
        start_time = time.time()
        for epoch in range(self.adv_epochs):
            adv_loss, hit_k, adv_grads = self.retrain_surrogate()

            consumed_time = time.time() - start_time
            if verbose:
                print('Epoch {:d}/{:d}, Adv Loss: {:.3f}, Hit Ratio@{:d}: {:.3f}%, Time: {:.3f}s'.
                      format(epoch, self.adv_epochs, adv_loss, self.topk, hit_k * 100., consumed_time))
            if writer:
                writer.add_scalar('{:s}/Adv_Loss'.format(self.name), adv_loss, epoch)
                writer.add_scalar('{:s}/Hit_Ratio@{:d}'.format(self.name, self.topk), hit_k, epoch)
            if adv_loss < min_loss:
                print('Minimal loss, save fake users.')
                self.fake_users = self.fake_tensor.detach().cpu().numpy()
                min_loss = adv_loss

            start_time = time.time()
            normalized_adv_grads = F.normalize(adv_grads, p=2, dim=1)
            self.adv_opt.zero_grad()
            self.fake_tensor.grad = normalized_adv_grads
            self.adv_opt.step()
            self.project_fake_tensor()
            self.scheduler.step()
