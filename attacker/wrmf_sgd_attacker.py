import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
import scipy.sparse as sp
import numpy as np
import torch.nn as nn
from utils import AverageMeter, mse_loss, ce_loss
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import higher
import time
from attacker.basic_attacker import BasicAttacker, PoisonedDataset
from torch.nn.init import kaiming_uniform_, calculate_gain, normal_, zeros_, ones_


class SurrogateWRMF(nn.Module):
    def __init__(self, config):
        super(SurrogateWRMF, self).__init__()
        self.embedding_size = config['embedding_size']
        self.n_users = config['n_users']
        self.n_items = config['n_items']
        self.device = config['device']
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.init_weights()
        self.to(device=self.device)

    def forward(self, users):
        user_e = self.user_embedding(users)
        scores = torch.mm(user_e, self.item_embedding.weight.t())
        return scores

    def init_weights(self):
        normal_(self.user_embedding.weight, std=0.1)
        normal_(self.item_embedding.weight, std=0.1)


class WRMFSGD(BasicAttacker):
    def __init__(self, attacker_config):
        super(WRMFSGD, self).__init__(attacker_config)
        self.adv_epochs = attacker_config['adv_epochs']
        self.train_epochs = attacker_config['train_epochs']
        self.unroll_steps = attacker_config['unroll_steps']
        self.weight = attacker_config['weight']
        self.topk = attacker_config['topk']
        self.initial_lr = attacker_config['lr']
        self.data_mat = sp.coo_matrix((np.ones((len(self.dataset.train_array),)), np.array(self.dataset.train_array).T),
                                      shape=(self.n_users, self.n_items), dtype=np.float32).tocsr()

        self.fake_tensor = self.init_fake_data()
        self.adv_opt = SGD([self.fake_tensor], lr=self.initial_lr, momentum=attacker_config['momentum'])
        self.scheduler = StepLR(self.adv_opt, step_size=self.adv_epochs / 3, gamma=0.1)

        self.poisoned_dataset = PoisonedDataset(self.data_mat, self.fake_tensor, self.device)
        self.poisoned_dataloader = DataLoader(self.poisoned_dataset, batch_size=attacker_config['batch_size'],
                                              shuffle=True, num_workers=0)

        test_user = TensorDataset(torch.arange(self.n_users, dtype=torch.int64, device=self.device))
        self.test_user_loader = DataLoader(test_user, batch_size=attacker_config['test_batch_size'])

        self.surrogate_config = attacker_config['surrogate_config']
        self.surrogate_config['device'] = self.device
        self.surrogate_config['n_users'] = self.n_users + self.n_fakes
        self.surrogate_config['n_items'] = self.n_items
        self.surrogate_model = SurrogateWRMF(self.surrogate_config)

    def init_fake_data(self):
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
            for fake_user in range(self.n_fakes):
                self.fake_tensor[fake_user, items[fake_user, :]] = 1.

    def train_adv(self):
        self.surrogate_model.init_weights()
        self.surrogate_model.train()
        train_opt = Adam(self.surrogate_model.parameters(), lr=self.surrogate_config['lr'],
                         weight_decay=self.surrogate_config['l2_reg'])

        for _ in range(self.train_epochs - self.unroll_steps):
            for users, profiles in self.poisoned_dataloader:
                users = users.to(dtype=torch.int64, device=self.device)
                scores = self.surrogate_model.forward(users)
                loss = mse_loss(profiles, scores, self.device, self.weight)
                train_opt.zero_grad()
                loss.backward()
                train_opt.step()

        with higher.innerloop_ctx(self.surrogate_model, train_opt) as (fmodel, diffopt):
            for _ in range(self.unroll_steps):
                for users, profiles in self.poisoned_dataloader:
                    users = users.to(dtype=torch.int64, device=self.device)
                    scores = fmodel.forward(users)
                    loss = mse_loss(profiles, scores, self.device, self.weight)
                    diffopt.step(loss)

            fmodel.eval()
            adv_losses = AverageMeter()
            hrs = AverageMeter()
            adv_grads = torch.zeros_like(self.fake_tensor, dtype=torch.float32, device=self.device)
            for users in self.test_user_loader:
                users = users[0]
                scores = fmodel.forward(users)
                adv_loss = ce_loss(scores, self.target_item)
                _, topk_items = scores.topk(self.topk, dim=1)
                hr = torch.eq(topk_items, self.target_item).float().sum(dim=1).mean()
                adv_grads += torch.autograd.grad(adv_loss, self.fake_tensor, retain_graph=True)[0]
                adv_losses.update(adv_loss.item() / users.shape[0], users.shape[0])
                hrs.update(hr.item(), users.shape[0])
            torch.autograd.grad(adv_loss, self.fake_tensor)
        return adv_losses.avg, hrs.avg, adv_grads

    def generate_fake_users(self, verbose=True, writer=None):
        for epoch in range(self.adv_epochs):
            start_time = time.time()
            adv_loss, hit_k, adv_grads = self.train_adv()

            normalized_adv_grads = F.normalize(adv_grads, p=2, dim=1)
            self.adv_opt.zero_grad()
            self.fake_tensor.grad = normalized_adv_grads
            self.adv_opt.step()
            self.project_fake_tensor()
            consumed_time = time.time() - start_time
            if verbose:
                print('Epoch {:d}/{:d}, Adv Loss: {:.3f}, Hit Ratio@{:d}: {:.3f}%, Time: {:.3f}s'.
                      format(epoch, self.adv_epochs, adv_loss, self.topk, hit_k * 100., consumed_time))
            if writer:
                writer.add_scalar('{:s}/Adv_Loss'.format(self.name), adv_loss, epoch)
                writer.add_scalar('{:s}/Hit_Ratio@{:d}'.format(self.name, self.topk), hit_k, epoch)
            self.scheduler.step()
        self.fake_users = self.fake_tensor.detach().cpu().numpy()
