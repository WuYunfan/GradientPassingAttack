from attacker.basic_attacker import BasicAttacker
import scipy.sparse as sp
import numpy as np
from attacker.revadv_attacker import GradientAttacker
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch.optim import Adam, SGD
from utils import mse_loss, ce_loss
import gc
import time
from model import get_model
from trainer import get_trainer
import torch.nn.functional as F


class PGAAttacker(GradientAttacker):
    def __init__(self, attacker_config):
        super(PGAAttacker, self).__init__(attacker_config)
        self.lmd = self.surrogate_trainer_config['l2_reg']

    def retrain_surrogate(self):
        self.surrogate_model.initial_embeddings()
        self.surrogate_trainer.initialize_optimizer()
        self.surrogate_trainer.best_ndcg = -np.inf
        self.surrogate_trainer.save_path = None
        self.surrogate_trainer.merge_fake_tensor(self.fake_tensor)

        start_time = time.time()
        self.surrogate_trainer.train(verbose=False)
        consumed_time = time.time() - start_time
        self.retrain_time += consumed_time

        self.surrogate_model.eval()
        scores = self.surrogate_model.predict(self.target_user_tensor)
        _, topk_items = scores.topk(self.topk, dim=1)
        hr = torch.eq(topk_items.unsqueeze(2), self.target_item_tensor.unsqueeze(0).unsqueeze(0))
        hr = hr.float().sum(dim=1).mean()
        adv_loss = ce_loss(scores, self.target_item_tensor)

        adv_grads = []
        with torch.no_grad():
            adv_grads_wrt_item_embeddings = torch.autograd.grad(adv_loss, self.surrogate_model.embedding.weight)
            adv_grads_wrt_item_embeddings = adv_grads_wrt_item_embeddings[0][-self.n_items:, :]
            fake_user_embeddings = self.surrogate_model.embedding.weight[self.n_users:-self.n_items, :]
            identity_mat = torch.eye(self.surrogate_model.embedding_size, device=self.device)
            for item in range(self.n_items):
                interacted_users = torch.where(self.surrogate_trainer.merged_data_tensor[:, item] > 0.5)[0]
                interacted_user_embeddings = self.surrogate_model.embedding.weight[interacted_users, :]
                sum_v_mat = torch.mm(interacted_user_embeddings.t(), interacted_user_embeddings)
                inv_mat = torch.linalg.inv(sum_v_mat + self.lmd * identity_mat)
                item_embedding_wrt_fake_inters = torch.mm(inv_mat, fake_user_embeddings.t())
                adv_grad = torch.mm(adv_grads_wrt_item_embeddings[item:item + 1, :], item_embedding_wrt_fake_inters)
                adv_grads.append(adv_grad)
        adv_grads = torch.cat(adv_grads, dim=0).t()
        gc.collect()
        torch.cuda.empty_cache()
        return adv_loss.item(), hr.item(), adv_grads
