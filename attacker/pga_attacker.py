from attacker.basic_attacker import BasicAttacker
import scipy.sparse as sp
from torch.optim import SGD
import numpy as np
from torch.optim.lr_scheduler import StepLR
from attacker.wrmf_sgd_attacker import WRMFSGD, SurrogateWRMF
from torch.utils.data import TensorDataset, DataLoader
import torch


class PGA(BasicAttacker):
    def __init__(self, attacker_config):
        super(PGA, self).__init__(attacker_config)
        WRMFSGD.init_fake_tensor(attacker_config)
        del self.unroll_steps

    def init_fake_tensor(self):
        return WRMFSGD.init_fake_tensor(self)

    def project_fake_tensor(self):
        WRMFSGD.project_fake_tensor(self)

    def generate_fake_users(self, verbose=True, writer=None):
        WRMFSGD.generate_fake_users(self, verbose, writer)