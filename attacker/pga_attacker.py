from attacker.basic_attacker import BasicAttacker
import scipy.sparse as sp
import numpy as np
from torch.optim.lr_scheduler import StepLR
from attacker.wrmf_sgd_attacker import WRMFSGD
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch.optim import Adam, SGD
from utils import mse_loss, ce_loss
import gc
import time
from model import get_model
from trainer import get_trainer
import torch.nn.functional as F


class PGA(BasicAttacker):
    def __init__(self, attacker_config):
        super(PGA, self).__init__(attacker_config)
        self.surrogate_model_config = attacker_config['surrogate_model_config']
        self.surrogate_trainer_config = attacker_config['surrogate_trainer_config']

        self.adv_epochs = attacker_config['adv_epochs']
        self.lmd = self.surrogate_trainer_config['l2_reg']
        self.lr = attacker_config['lr']
        self.momentum = attacker_config['momentum']

        self.data_mat = sp.coo_matrix((np.ones((len(self.dataset.train_array),)), np.array(self.dataset.train_array).T),
                                      shape=(self.n_users, self.n_items), dtype=np.float32).tocsr()
        self.fake_tensor = self.init_fake_tensor()
        self.adv_opt = SGD([self.fake_tensor], lr=self.lr, momentum=self.momentum)

        target_users = [user for user in range(self.n_users) if self.target_item not in self.dataset.train_data[user]]
        self.target_users = torch.tensor(target_users, dtype=torch.int64, device=self.device)

        self.pre_trained_model = self.load_pretrained_model(self.surrogate_trainer_config.get('pre_train_path', None))
        self.surrogate_model_config['n_fakes'] = self.n_fakes
        self.surrogate_model = get_model(self.surrogate_model_config, self.dataset)
        self.surrogate_trainer = get_trainer(self.surrogate_trainer_config, self.surrogate_model)

        train_user = TensorDataset(torch.arange(self.surrogate_model.n_users, dtype=torch.int64, device=self.device))
        self.surrogate_trainer.train_user_loader = \
            DataLoader(train_user, batch_size=self.surrogate_trainer_config['batch_size'], shuffle=True)

    def load_pretrained_model(self, path):
        return WRMFSGD.load_pretrained_model(self, path)

    def init_fake_tensor(self):
        return WRMFSGD.init_fake_tensor(self)

    def project_fake_tensor(self):
        WRMFSGD.project_fake_tensor(self)

    def retrain_surrogate(self):
        self.surrogate_model.initial_embeddings()
        if self.pre_trained_model is not None:
            self.surrogate_model.initial_pretrained_parameters(self.pre_trained_model)
        self.surrogate_trainer.initialize_optimizer()
        self.surrogate_trainer.merge_fake_tensor(self.fake_tensor)

        start_time = time.time()
        self.surrogate_trainer.train(verbose=False)
        consumed_time = time.time() - start_time
        self.retrain_time += consumed_time

        self.surrogate_model.eval()
        scores = self.surrogate_model.predict(self.target_users)
        _, topk_items = scores.topk(self.topk, dim=1)
        hr = torch.eq(topk_items, self.target_item).float().sum(dim=1).mean()
        adv_loss = ce_loss(scores, self.target_item)

        adv_grads = []
        with torch.no_grad():
            normed_embeddings = F.normalize(self.surrogate_model.embedding.weight, dim=1, p=2)
            adv_grads_wrt_item_embeddings = torch.autograd.grad(adv_loss, self.surrogate_model.embedding.weight)
            adv_grads_wrt_item_embeddings = adv_grads_wrt_item_embeddings[0][-self.n_items:, :]
            item_norm = torch.norm(self.surrogate_model.embedding.weight[-self.n_items:, :], dim=1, p=2)
            adv_grads_wrt_normed_item_embeddings = adv_grads_wrt_item_embeddings * item_norm[:, None]
            normed_fake_user_embeddings = normed_embeddings[self.n_users:-self.n_items, :]
            identity_mat = torch.eye(self.surrogate_model.embedding_size, device=self.device)
            for item in range(self.n_items):
                interacted_users = self.surrogate_trainer.merged_data_mat[:, item].nonzero()[0]
                normed_interacted_user_embeddings = normed_embeddings[interacted_users, :]
                sum_v_mat = torch.mm(normed_interacted_user_embeddings.t(), normed_interacted_user_embeddings)
                inv_mat = torch.linalg.inv(sum_v_mat + self.lmd * identity_mat)
                normed_item_embedding_wrt_fake_inters = torch.mm(inv_mat, normed_fake_user_embeddings.t())
                adv_grad = torch.mm(adv_grads_wrt_normed_item_embeddings[item:item + 1, :], normed_item_embedding_wrt_fake_inters)
                adv_grads.append(adv_grad)
        adv_grads = torch.cat(adv_grads, dim=0).t()
        return adv_loss.item(), hr.item(), adv_grads

    def generate_fake_users(self, verbose=True, writer=None):
        WRMFSGD.generate_fake_users(self, verbose, writer)
