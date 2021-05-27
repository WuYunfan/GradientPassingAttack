import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
import scipy.sparse as sp
import numpy as np
import torch.nn as nn
from utils import AverageMeter
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import higher
import time
from attacker.basic_attacker import BasicAttacker
from torch.nn.init import kaiming_uniform_, calculate_gain, normal_, zeros_, ones_


class SurrogateWRMF(nn.Module):
    def __init__(self, config):
        super(SurrogateWRMF, self).__init__()
        self.embedding_size = config['embedding_size']
        self.n_users = config['n_users']
        self.n_items = config['n_items']
        self.n_epochs = config['n_epochs']
        self.unroll_steps = config['unroll_steps']
        self.device = config['device']
        self.dataloader = config['dataloader']
        self.weight = config['weight']
        self.topk = config['topk']
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        normal_(self.user_embedding.weight, std=0.1)
        normal_(self.item_embedding.weight, std=0.1)
        self.to(device=self.device)
        self.opt = Adam(self.parameters(), lr=config['lr'], weight_decay=config['l2_reg'])
        test_user = TensorDataset(torch.arange(self.n_users, dtype=torch.int64, device=self.device))
        self.test_user_loader = DataLoader(test_user, batch_size=config['batch_size'])

    def forward(self, users):
        user_e = self.user_embedding(users)
        scores = torch.mm(user_e, self.item_embedding.weight.t())
        return scores

    def mse_loss(self, profiles, scores):
        weights = torch.ones_like(profiles, dtype=torch.float32, device=self.device)
        weights[profiles > 0] = self.weight
        loss = weights * (profiles - scores) ** 2
        loss = torch.mean(loss)
        return loss

    @staticmethod
    def ce_loss(scores, target_item):
        log_probs = F.log_softmax(scores, dim=-1)
        return -log_probs[:, target_item].sum()

    def train_adv(self, target_item, fake_tensor):
        self.train()
        for _ in range(self.n_epochs - self.unroll_steps):
            for users, profiles in self.dataloader:
                users = users.to(dtype=torch.int64, device=self.device)
                scores = self.forward(users)
                loss = self.mse_loss(profiles, scores)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

        with higher.innerloop_ctx(self, self.opt) as (fmodel, diffopt):
            for _ in range(self.unroll_steps):
                for users, profiles in self.dataloader:
                    users = users.to(dtype=torch.int64, device=self.device)
                    scores = fmodel.forward(users)
                    loss = self.mse_loss(profiles, scores)
                    diffopt.step(loss)

            self.eval()
            adv_losses = AverageMeter()
            hrs = AverageMeter()
            adv_grads = torch.zeros_like(fake_tensor, dtype=torch.float32, device=self.device, requires_grad=False)
            for users in self.test_user_loader:
                users = users[0]
                scores = fmodel.forward(users)
                adv_loss = self.ce_loss(scores, target_item)
                _, topk_items = scores.topk(self.topk, dim=1)
                hr = torch.eq(topk_items, target_item).float().sum(dim=1).mean()
                adv_grads += torch.autograd.grad(adv_loss, fake_tensor, retain_graph=True)[0]
                adv_losses.update(adv_loss.item() / users.shape[0], users.shape[0])
                hrs.update(hr.item(), users.shape[0])
            torch.autograd.grad(adv_loss, fake_tensor)
        return adv_losses.avg, hrs.avg, adv_grads


class WRMF_SGD_Dataset(Dataset):
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


class WRMF_SGD(BasicAttacker):
    def __init__(self, attacker_config):
        super(WRMF_SGD, self).__init__(attacker_config)
        self.adv_epochs = attacker_config['adv_epochs']
        self.topk = attacker_config['topk']
        self.initial_lr = attacker_config['lr']
        self.data_mat = sp.coo_matrix((np.ones((len(self.dataset.train_array),)), np.array(self.dataset.train_array).T),
                                      shape=(self.n_users, self.n_items), dtype=np.float32).tocsr()

        self.fake_tensor = self.init_fake_data()
        self.fake_tensor.requires_grad_()
        self.opt = SGD([self.fake_tensor], lr=self.initial_lr, momentum=attacker_config['momentum'])
        self.scheduler = StepLR(self.opt, step_size=self.adv_epochs / 3, gamma=0.1)

        self.surrogate_dataset = WRMF_SGD_Dataset(self.data_mat, self.fake_tensor, self.device)
        self.surrogate_dataloader = DataLoader(self.surrogate_dataset, batch_size=attacker_config['batch_size'],
                                               shuffle=True, num_workers=attacker_config['dataloader_num_workers'])

        self.surrogate_config = attacker_config['surrogate_config']
        self.surrogate_config['device'] = self.device
        self.surrogate_config['n_users'] = self.n_users + self.n_fakes
        self.surrogate_config['n_items'] = self.n_items
        self.surrogate_config['dataloader'] = self.surrogate_dataloader
        self.surrogate_config['topk'] = self.topk
        self.surrogate_config['batch_size'] = attacker_config['batch_size']

    def init_fake_data(self):
        degree = np.array(np.sum(self.data_mat, axis=1)).squeeze()
        qualified_users = self.data_mat[degree <= self.n_inters, :]
        sample_idx = np.random.choice(qualified_users.shape[0], self.n_fakes, replace=False)
        fake_data = qualified_users[sample_idx, :].toarray()
        fake_data = torch.tensor(fake_data, dtype=torch.float32, device=self.device)
        return fake_data

    def project_fake_tensor(self):
        with torch.no_grad():
            _, items = self.fake_tensor.topk(self.n_inters, dim=1)
            self.fake_tensor.zero_()
            for fake_user in range(self.n_fakes):
                self.fake_tensor[fake_user, items[fake_user, :]] = 1.

    def generate_fake_users(self, verbose=True, writer=None):
        for epoch in range(self.adv_epochs):
            self.scheduler.step()
            start_time = time.time()
            surrogate_model = SurrogateWRMF(self.surrogate_config)
            adv_loss, hit_k, adv_grads = surrogate_model.train_adv(self.target_item, self.fake_tensor)

            normalized_adv_grads = F.normalize(adv_grads, p=2, dim=1)
            self.opt.zero_grad()
            self.fake_tensor.grad = normalized_adv_grads
            self.opt.step()
            self.project_fake_tensor()
            consumed_time = time.time() - start_time
            if verbose:
                print('Epoch {:d}/{:d}, Adv Loss: {:.3f}, Hit Ratio@{:d}: {:.3f}, Time: {:.3f}s'.
                      format(epoch, self.adv_epochs, adv_loss, self.topk, hit_k, consumed_time))
            if writer:
                writer.add_scalar('WRMF_SGD/Adv_Loss', adv_loss, epoch)
                writer.add_scalar('WRMF_SGD/Hit_Ratio@{:d}'.format(self.topk), hit_k, epoch)
        self.fake_users = self.fake_tensor.detach().cpu().numpy()
