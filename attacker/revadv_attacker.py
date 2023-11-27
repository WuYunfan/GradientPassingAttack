import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
import scipy.sparse as sp
import numpy as np
from utils import mse_loss, ce_loss
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import higher
import time
from attacker.basic_attacker import BasicAttacker
import gc
from model import get_model
from trainer import get_trainer


class RevAdv(BasicAttacker):
    def __init__(self, attacker_config):
        super(RevAdv, self).__init__(attacker_config)
        self.surrogate_model_config = attacker_config['surrogate_model_config']
        self.surrogate_trainer_config = attacker_config['surrogate_trainer_config']

        self.adv_epochs = attacker_config['adv_epochs']
        self.unroll_steps = attacker_config['unroll_steps']
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
        self.data_tensor = torch.tensor(self.surrogate_trainer.data_mat.toarray(),
                                        dtype=torch.float32, device=self.device)

    def load_pretrained_model(self, path):
        if path is None:
            return None
        pre_trained_model_config = self.surrogate_model_config.copy()
        pre_trained_model_config['embedding_size'] = pre_trained_model_config['pretrain_fixed_dim']
        pre_trained_model = get_model(pre_trained_model_config, self.dataset)
        pre_trained_model.load(path)
        return pre_trained_model

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
        self.surrogate_model.initial_embeddings()
        if self.pre_trained_model is not None:
            self.surrogate_model.initial_pretrained_parameters(self.pre_trained_model)
        self.surrogate_trainer.initialize_optimizer()
        self.surrogate_trainer.merge_fake_tensor(self.fake_tensor)
        poisoned_data_tensor = torch.cat([self.data_tensor, self.fake_tensor], dim=0)

        start_time = time.time()
        self.surrogate_trainer.train(verbose=False)

        with higher.innerloop_ctx(self.surrogate_model, self.surrogate_trainer.opt) as (fmodel, diffopt):
            fmodel.train()
            for _ in range(self.unroll_steps):
                for users in self.surrogate_trainer.train_user_loader:
                    users = users[0]
                    profiles = poisoned_data_tensor[users, :]
                    scores, l2_norm_sq = fmodel.mse_forward(users, self.surrogate_trainer.gp_config)
                    m_loss = mse_loss(profiles, scores)
                    loss = m_loss + self.surrogate_trainer.l2_reg * l2_norm_sq
                    diffopt.step(loss)
            consumed_time = time.time() - start_time
            self.retrain_time += consumed_time

            fmodel.eval()
            scores = fmodel.predict(self.target_users)
            _, topk_items = scores.topk(self.topk, dim=1)
            hr = torch.eq(topk_items, self.target_item).float().sum(dim=1).mean()
            adv_loss = ce_loss(scores, self.target_item)
            adv_grads = torch.autograd.grad(adv_loss, self.fake_tensor)[0]
        gc.collect()
        torch.cuda.empty_cache()
        return adv_loss.item(), hr.item(), adv_grads

    def generate_fake_users(self, verbose=True, writer=None):
        max_hr = -np.inf
        start_time = time.time()
        for epoch in range(self.adv_epochs):
            adv_loss, hit_k, adv_grads = self.retrain_surrogate()

            consumed_time = time.time() - start_time
            self.consumed_time += consumed_time
            if verbose:
                print('Epoch {:d}/{:d}, Adv Loss: {:.3f}, Hit Ratio@{:d}: {:.3f}%, Time: {:.3f}s'.
                      format(epoch, self.adv_epochs, adv_loss, self.topk, hit_k * 100., consumed_time))
            if writer:
                writer.add_scalar('{:s}/Adv_Loss'.format(self.name), adv_loss, epoch)
                writer.add_scalar('{:s}/Hit_Ratio@{:d}'.format(self.name, self.topk), hit_k, epoch)
            if hit_k > max_hr:
                print('Maximal hit ratio, save fake users.')
                self.fake_users = self.fake_tensor.detach().cpu().numpy().copy()
                max_hr = hit_k

            start_time = time.time()
            normalized_adv_grads = F.normalize(adv_grads, p=2, dim=1)
            self.adv_opt.zero_grad()
            self.fake_tensor.grad = normalized_adv_grads
            self.adv_opt.step()
            self.project_fake_tensor()
