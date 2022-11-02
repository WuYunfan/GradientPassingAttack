from attacker.basic_attacker import BasicAttacker
import scipy.sparse as sp
import numpy as np
from torch.optim.lr_scheduler import StepLR
from attacker.wrmf_sgd_attacker import WRMFSGD, SurrogateWRMF
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch.optim import Adam, SGD
from utils import mse_loss, ce_loss
import gc


class PGA(BasicAttacker):
    def __init__(self, attacker_config):
        super(PGA, self).__init__(attacker_config)
        self.surrogate_config = attacker_config['surrogate_config']
        self.surrogate_config['device'] = self.device
        self.surrogate_config['n_users'] = self.n_users + self.n_fakes
        self.surrogate_config['n_items'] = self.n_items

        self.adv_epochs = attacker_config['adv_epochs']
        self.train_epochs = attacker_config['train_epochs']
        self.lmd = self.surrogate_config['l2_reg']
        self.weight = attacker_config['weight']
        self.initial_lr = attacker_config['lr']
        self.momentum = attacker_config['momentum']

        self.data_mat = sp.coo_matrix((np.ones((len(self.dataset.train_array),)), np.array(self.dataset.train_array).T),
                                      shape=(self.n_users, self.n_items), dtype=np.float32).tocsr()
        self.fake_tensor = self.init_fake_tensor()
        self.adv_opt = SGD([self.fake_tensor], lr=self.initial_lr, momentum=self.momentum)
        self.scheduler = StepLR(self.adv_opt, step_size=self.adv_epochs / 3, gamma=0.1)

        poisoned_data_mat = torch.tensor(self.data_mat.toarray(), dtype=torch.float32, device=self.device)
        self.poisoned_data_mat = torch.cat([poisoned_data_mat, self.fake_tensor], dim=0)
        test_users = TensorDataset(torch.arange(self.n_users + self.n_fakes, dtype=torch.int64, device=self.device))
        self.user_loader = DataLoader(test_users, batch_size=self.surrogate_config['batch_size'],
                                      shuffle=True,)
        target_users = [user for user in range(self.n_users) if self.target_item not in self.dataset.train_data[user]]
        self.target_users = np.array(target_users)

    def init_fake_tensor(self):
        return WRMFSGD.init_fake_tensor(self)

    def project_fake_tensor(self):
        WRMFSGD.project_fake_tensor(self)

    def retrain_surrogate(self):
        surrogate_model = SurrogateWRMF(self.surrogate_config)
        surrogate_model.train()
        train_opt = Adam(surrogate_model.parameters(), lr=self.surrogate_config['lr'],
                         weight_decay=self.surrogate_config['l2_reg'])

        for _ in range(self.train_epochs):
            for users in self.user_loader:
                users = users[0]
                batch_data = self.poisoned_data_mat[users, :]
                scores = surrogate_model.forward(users)
                loss = mse_loss(batch_data, scores, self.weight)
                train_opt.zero_grad()
                loss.backward()
                train_opt.step()

        surrogate_model.eval()
        scores = []
        all_users = []
        for users in self.user_loader:
            users = users[0]
            scores.append(surrogate_model.forward(users))
            all_users += users.cpu().numpy().tolist()
        norm_user_pos = np.argsort(all_users)[:-self.n_fakes]
        scores = torch.cat(scores, dim=0)[norm_user_pos, :][self.target_users, :]
        adv_loss = ce_loss(scores, self.target_item)
        _, topk_items = scores.topk(self.topk, dim=1)
        hr = torch.eq(topk_items, self.target_item).float().sum(dim=1).mean()

        adv_grads = []
        adv_grads_wrt_item_embeddings = torch.autograd.grad(adv_loss, surrogate_model.item_embedding.weight)[0]
        with torch.no_grad():
            for item in range(self.n_items):
                interacted_users = torch.nonzero(self.poisoned_data_mat[:, item])[:, 0]
                interacted_user_embeddings = surrogate_model.user_embedding(interacted_users)
                sum_v_mat = torch.mm(interacted_user_embeddings.t(), interacted_user_embeddings)
                inv_mat = torch.linalg.inv(sum_v_mat +
                                           self.lmd * torch.eye(surrogate_model.embedding_size, device=self.device))
                item_embedding_wrt_fake_inters = torch.mm(inv_mat,
                                                          surrogate_model.user_embedding.weight[-self.n_fakes:, :].t())
                adv_grad = torch.mm(adv_grads_wrt_item_embeddings[item:item + 1, :], item_embedding_wrt_fake_inters)
                adv_grads.append(adv_grad)
        adv_grads = torch.cat(adv_grads, dim=0).t()
        gc.collect()
        return adv_loss.item(), hr.item(), adv_grads

    def generate_fake_users(self, verbose=True, writer=None):
        WRMFSGD.generate_fake_users(self, verbose, writer)