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
from attacker.basic_attacker import BasicAttacker
from torch.nn.init import kaiming_uniform_, calculate_gain, normal_, zeros_, ones_
from attacker.wrmf_sgd_attacker import WRMFSGD


class SurrogateItemAE(nn.Module):
    def __init__(self, config):
        super(SurrogateItemAE, self).__init__()
        self.e_layer_sizes = config['layer_sizes']
        self.d_layer_sizes = self.e_layer_sizes[::-1].copy()
        self.device = config['device']
        self.encoder_layers, self.decoder_layers = [], []
        for layer_idx in range(1, len(self.e_layer_sizes)):
            encoder_layer = nn.Linear(self.e_layer_sizes[layer_idx - 1], self.e_layer_sizes[layer_idx])
            self.encoder_layers.append(encoder_layer)
            decoder_layer = nn.Linear(self.d_layer_sizes[layer_idx - 1], self.d_layer_sizes[layer_idx])
            self.decoder_layers.append(decoder_layer)
        self.encoder_layers = nn.ModuleList(self.encoder_layers)
        self.decoder_layers = nn.ModuleList(self.decoder_layers)
        self.init_weights()
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

    def init_weights(self):
        for layer in self.encoder_layers:
            kaiming_uniform_(layer.weight, nonlinearity='tanh')
            zeros_(layer.bias)
        for layer in self.decoder_layers:
            kaiming_uniform_(layer.weight, nonlinearity='tanh')
            zeros_(layer.bias)


class PoisonedDatasetItem(Dataset):
    def __init__(self, data_mat, fake_tensor, device):
        self.data_mat = data_mat
        self.fake_tensor = fake_tensor
        self.device = device
        self.n_items = data_mat.shape[1]

    def __len__(self):
        return self.n_items

    def __getitem__(self, index):
        profile = torch.tensor(self.data_mat[:, index].toarray().squeeze(), dtype=torch.float32, device=self.device)
        profile = torch.cat([profile, self.fake_tensor[:, index]], dim=0)
        return index, profile


class ItemAESGD(BasicAttacker):
    def __init__(self, attacker_config):
        super(ItemAESGD, self).__init__(attacker_config)
        self.adv_epochs = attacker_config['adv_epochs']
        self.train_epochs = attacker_config['train_epochs']
        self.unroll_steps = attacker_config['unroll_steps']
        self.weight = attacker_config['weight']
        self.max_patience = attacker_config.get('max_patience', 20)
        self.topk = attacker_config['topk']
        self.initial_lr = attacker_config['lr']
        self.data_mat = sp.coo_matrix((np.ones((len(self.dataset.train_array),)), np.array(self.dataset.train_array).T),
                                      shape=(self.n_users, self.n_items), dtype=np.float32).tocsr()

        self.fake_tensor = self.init_fake_data()
        self.adv_opt = SGD([self.fake_tensor], lr=self.initial_lr, momentum=attacker_config['momentum'])
        self.scheduler = StepLR(self.adv_opt, step_size=self.adv_epochs / 3, gamma=0.1)

        self.poisoned_dataset = PoisonedDatasetItem(self.data_mat, self.fake_tensor, self.device)
        self.poisoned_dataloader = DataLoader(self.poisoned_dataset, batch_size=attacker_config['batch_size'],
                                              shuffle=True, num_workers=0)

        self.surrogate_config = attacker_config['surrogate_config']
        self.surrogate_config['device'] = self.device
        self.surrogate_config['layer_sizes'].insert(0, self.n_users + self.n_fakes)
        self.surrogate_model = SurrogateItemAE(self.surrogate_config)

    def init_fake_data(self):
        return WRMFSGD.init_fake_data(self)

    def project_fake_tensor(self):
        WRMFSGD.project_fake_tensor(self)

    def train_adv(self):
        self.surrogate_model.init_weights()
        self.surrogate_model.train()
        train_opt = Adam(self.surrogate_model.parameters(), lr=self.surrogate_config['lr'],
                         weight_decay=self.surrogate_config['l2_reg'])

        for _ in range(self.train_epochs - self.unroll_steps):
            for items, profiles in self.poisoned_dataloader:
                scores = self.surrogate_model.forward(profiles)
                loss = mse_loss(profiles, scores, self.device, self.weight)
                train_opt.zero_grad()
                loss.backward()
                train_opt.step()

        with higher.innerloop_ctx(self.surrogate_model, train_opt) as (fmodel, diffopt):
            for _ in range(self.unroll_steps):
                for items, profiles in self.poisoned_dataloader:
                    scores = fmodel.forward(profiles)
                    loss = mse_loss(profiles, scores, self.device, self.weight)
                    diffopt.step(loss)

            fmodel.eval()
            scores = []
            all_items = []
            for items, profiles in self.poisoned_dataloader:
                scores.append(fmodel.forward(profiles))
                all_items += items.numpy().tolist()
            scores = torch.cat(scores, dim=0).t()[:-self.n_fakes, :]
            target_item = all_items.index(self.target_item)
            adv_loss = ce_loss(scores, target_item)
            _, topk_items = scores.topk(self.topk, dim=1)
            hr = torch.eq(topk_items, target_item).float().sum(dim=1).mean()
            adv_grads = torch.autograd.grad(adv_loss, self.fake_tensor)[0]
        return adv_loss.item() / self.n_users, hr.item(), adv_grads

    def generate_fake_users(self, verbose=True, writer=None):
        WRMFSGD.generate_fake_users(self, verbose, writer)



