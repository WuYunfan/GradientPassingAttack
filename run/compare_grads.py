from dataset import get_dataset
from attacker import get_attacker
from utils import init_run, set_seed
import torch
from config import get_ml1m_attacker_config
import numpy as np


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2021)

    device = torch.device('cuda')
    dataset_config = {'name': 'SyntheticDataset', 'n_users': 1000, 'n_items': 300, 'binary_threshold': 10.,
                      'split_ratio': [0.8, 0.2], 'device': device}
    dataset = get_dataset(dataset_config)
    igcn_config = {'name': 'IGCN', 'n_layers': 3, 'dropout': 0.3, 'feature_ratio': 1.,
                   'embedding_size': 64, 'device': device, 'lr': 1.e-2, 'l2_reg': 1.e-5}
    attacker_config = {'name': 'GBFUG', 'lr': 10., 'momentum': 0.9, 'batch_size': 2048,
                       'device': device, 'n_fakes': 10, 'unroll_steps': 5,
                       'n_inters': 5, 'test_batch_size': 2048, 'target_item': 0,
                       'adv_epochs': 100, 'igcn_config': igcn_config, 'train_epochs': 50,
                       'topk': 20, 'weight': 20.}
    attacker = get_attacker(attacker_config, dataset)

    coss = []
    for i in range(10):
        set_seed(2021 + i)
        attacker.fake_indices, attacker.fake_tensor = attacker.init_fake_data()
        attacker.unroll_steps = 0
        set_seed(2021)
        _, _, p_grads = attacker.train_adv()
        attacker.unroll_steps = 50
        set_seed(2021)
        _, _, grads = attacker.train_adv()
        cos = (p_grads * grads).sum()
        cos = cos / (torch.norm(p_grads.flatten(), p=2, dim=0) * torch.norm(grads.flatten(), p=2, dim=0))
        print('Cos: ', cos.item())
        coss.append(cos.item())
    print('Mean cos: ', np.mean(coss))


if __name__ == '__main__':
    main()