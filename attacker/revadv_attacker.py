import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
import scipy.sparse as sp
import numpy as np
from utils import diff_bce_loss, ce_loss
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

        self.surrogate_model_config['n_fakes'] = self.n_fakes
        self.surrogate_model = get_model(self.surrogate_model_config, self.dataset)
        self.surrogate_trainer = get_trainer(self.surrogate_trainer_config, self.surrogate_model)

        self.fake_tensor = self.init_fake_tensor(self.surrogate_trainer.data_tensor)
        self.adv_opt = SGD([self.fake_tensor], lr=self.lr, momentum=self.momentum)

        self.target_user_tensor = torch.arange(self.n_users, dtype=torch.int64, device=self.device)
        self.target_item_tensor = torch.tensor(self.target_items, dtype=torch.int64, device=self.device)

    def init_fake_tensor(self, data_tensor):
        degree = torch.sum(data_tensor, dim=1)
        qualified_users = data_tensor[degree <= self.n_inters, :]
        sample_idxes = torch.randint(qualified_users.shape[0], size=[self.n_fakes])
        fake_tensor = qualified_users[sample_idxes, :]
        fake_tensor.requires_grad = True
        return fake_tensor

    def project_fake_tensor(self):
        with torch.no_grad():
            _, items = self.fake_tensor.topk(self.n_inters, dim=1)
            self.fake_tensor.zero_()
            self.fake_tensor.data = torch.scatter(self.fake_tensor, 1, items, 1.)

    def retrain_surrogate(self):
        self.surrogate_model.initial_embeddings()
        self.surrogate_trainer.initialize_optimizer()
        self.surrogate_trainer.merge_fake_tensor(self.fake_tensor)

        start_time = time.time()
        self.surrogate_trainer.train(verbose=False)
        order = self.surrogate_trainer.gp_config.order
        self.surrogate_trainer.gp_config.order = 0
        if order > 0: self.surrogate_trainer.initialize_optimizer()
        with higher.innerloop_ctx(self.surrogate_model, self.surrogate_trainer.opt) as (fmodel, diffopt):
            fmodel.train()
            for _ in range(self.unroll_steps):
                for users in self.surrogate_trainer.train_user_loader:
                    users = users[0]
                    scores, l2_norm_sq = fmodel.forward(users, self.surrogate_trainer.gp_config)
                    profiles = self.surrogate_trainer.merged_data_tensor[users, :]
                    bce_loss = diff_bce_loss(profiles, scores)
                    loss = bce_loss + self.surrogate_trainer.l2_reg * l2_norm_sq.mean()
                    diffopt.step(loss)
            consumed_time = time.time() - start_time
            self.retrain_time += consumed_time

            fmodel.eval()
            scores = fmodel.predict(self.target_user_tensor)
            _, topk_items = scores.topk(self.topk, dim=1)
            hr = torch.eq(topk_items.unsqueeze(2), self.target_item_tensor.unsqueeze(0).unsqueeze(0))
            hr = hr.float().sum(dim=1).mean()
            adv_loss = ce_loss(scores, self.target_item_tensor)
            adv_grads = torch.autograd.grad(adv_loss, self.fake_tensor)[0]
        self.surrogate_trainer.gp_config.order = order
        gc.collect()
        torch.cuda.empty_cache()
        return adv_loss.item(), hr.item(), adv_grads

    def generate_fake_users(self, verbose=True, writer=None):
        max_hr = -np.inf
        for epoch in range(self.adv_epochs):
            start_time = time.time()

            adv_loss, hit_k, adv_grads = self.retrain_surrogate()
            if hit_k > max_hr:
                print('Maximal hit ratio, save fake users.')
                self.fake_users = self.fake_tensor.detach().cpu().numpy().copy()
                max_hr = hit_k

            normalized_adv_grads = F.normalize(adv_grads, p=2, dim=1)
            self.adv_opt.zero_grad()
            self.fake_tensor.grad = normalized_adv_grads
            self.adv_opt.step()
            self.project_fake_tensor()

            consumed_time = time.time() - start_time
            self.consumed_time += consumed_time
            if verbose:
                print('Epoch {:d}/{:d}, Adv Loss: {:.3f}, Hit Ratio@{:d}: {:.3f}%, Time: {:.3f}s'.
                      format(epoch, self.adv_epochs, adv_loss, self.topk, hit_k * 100., consumed_time))
            if writer:
                writer.add_scalar('{:s}/Adv_Loss'.format(self.name), adv_loss, epoch)
                writer.add_scalar('{:s}/Hit_Ratio@{:d}'.format(self.name, self.topk), hit_k, epoch)
