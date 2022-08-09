import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
import scipy.sparse as sp
import numpy as np
import torch.nn as nn
from utils import mse_loss, ce_loss
from torch.utils.data import DataLoader, TensorDataset
import higher
from attacker.basic_attacker import BasicAttacker
from attacker.wrmf_sgd_attacker import WRMFSGD
from model import init_one_layer


class SurrogateItemAE(nn.Module):
    def __init__(self, config):
        super(SurrogateItemAE, self).__init__()
        self.e_layer_sizes = config['layer_sizes']
        self.d_layer_sizes = self.e_layer_sizes[::-1].copy()
        self.device = config['device']
        self.encoder_layers, self.decoder_layers = [], []
        for layer_idx in range(1, len(self.e_layer_sizes)):
            encoder_layer = init_one_layer(self.e_layer_sizes[layer_idx - 1], self.e_layer_sizes[layer_idx])
            self.encoder_layers.append(encoder_layer)
            decoder_layer = init_one_layer(self.d_layer_sizes[layer_idx - 1], self.d_layer_sizes[layer_idx])
            self.decoder_layers.append(decoder_layer)
        self.encoder_layers = nn.ModuleList(self.encoder_layers)
        self.decoder_layers = nn.ModuleList(self.decoder_layers)
        self.to(device=self.device)

    def forward(self, profiles):
        x = profiles
        for layer in self.encoder_layers:
            x = torch.tanh(layer(x))
        for idx, layer in enumerate(self.decoder_layers):
            x = layer(x)
            if idx != len(self.decoder_layers) - 1:
                x = torch.tanh(x)
        return x


class ItemAESGD(BasicAttacker):
    def __init__(self, attacker_config):
        super(ItemAESGD, self).__init__(attacker_config)
        self.adv_epochs = attacker_config['adv_epochs']
        self.train_epochs = attacker_config['train_epochs']
        self.unroll_steps = attacker_config['unroll_steps']
        self.weight = attacker_config['weight']
        self.topk = attacker_config['topk']
        self.initial_lr = attacker_config['lr']
        self.data_mat = sp.coo_matrix((np.ones((len(self.dataset.train_array),)), np.array(self.dataset.train_array).T),
                                      shape=(self.n_users, self.n_items), dtype=np.float32).tocsr()

        self.fake_tensor = self.init_fake_tensor()
        self.adv_opt = SGD([self.fake_tensor], lr=self.initial_lr, momentum=attacker_config['momentum'])
        self.scheduler = StepLR(self.adv_opt, step_size=self.adv_epochs / 3, gamma=0.1)

        poisoned_data_mat = torch.tensor(self.data_mat.toarray(), dtype=torch.float32, device=self.device)
        self.poisoned_data_mat = torch.cat([poisoned_data_mat, self.fake_tensor], dim=0).t()
        test_item = TensorDataset(torch.arange(self.n_items, dtype=torch.int64, device=self.device))
        self.item_loader = DataLoader(test_item, batch_size=attacker_config['batch_size'],
                                      shuffle=True, num_workers=0)

        self.surrogate_config = attacker_config['surrogate_config']
        self.surrogate_config['device'] = self.device
        self.surrogate_config['layer_sizes'].insert(0, self.n_users + self.n_fakes)

    def init_fake_tensor(self):
        return WRMFSGD.init_fake_tensor(self)

    def project_fake_tensor(self):
        WRMFSGD.project_fake_tensor(self)

    def retrain_surrogate(self):
        surrogate_model = SurrogateItemAE(self.surrogate_config)
        surrogate_model.train()
        train_opt = Adam(surrogate_model.parameters(), lr=self.surrogate_config['lr'],
                         weight_decay=self.surrogate_config['l2_reg'])

        for _ in range(self.train_epochs - self.unroll_steps):
            for items in self.item_loader:
                items = items[0]
                batch_data = self.poisoned_data_mat[items, :]
                scores = surrogate_model.forward(batch_data)
                loss = mse_loss(batch_data, scores, self.device, self.weight)
                train_opt.zero_grad()
                loss.backward()
                train_opt.step()

        with higher.innerloop_ctx(surrogate_model, train_opt) as (fmodel, diffopt):
            for _ in range(self.unroll_steps):
                for items in self.item_loader:
                    items = items[0]
                    batch_data = self.poisoned_data_mat[items, :]
                    scores = fmodel.forward(batch_data)
                    loss = mse_loss(batch_data, scores, self.device, self.weight)
                    diffopt.step(loss)

            fmodel.eval()
            scores = []
            all_items = []
            for items in self.item_loader:
                items = items[0]
                batch_data = self.poisoned_data_mat[items, :]
                scores.append(fmodel.forward(batch_data))
                all_items += items.cpu().numpy().tolist()
            scores = torch.cat(scores, dim=0).t()[:-self.n_fakes, :]
            target_item = all_items.index(self.target_item)
            adv_loss = ce_loss(scores, target_item)
            _, topk_items = scores.topk(self.topk, dim=1)
            hr = torch.eq(topk_items, target_item).float().sum(dim=1).mean()
            adv_grads = torch.autograd.grad(adv_loss, self.fake_tensor)[0]
        return adv_loss.item(), hr.item(), adv_grads

    def generate_fake_users(self, verbose=True, writer=None):
        WRMFSGD.generate_fake_users(self, verbose, writer)



