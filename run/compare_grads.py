from dataset import get_dataset
from attacker import get_attacker
from utils import init_run, set_seed
import torch
from config import get_ml1m_attacker_config
import numpy as np


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2021)

    device = torch.device('cpu')
    dataset_config = {'name': 'SyntheticDataset', 'n_users': 1000, 'n_items': 300, 'binary_threshold': 10.,
                      'split_ratio': [0.8, 0.2], 'device': device}
    dataset = get_dataset(dataset_config)
    attacker_config = get_ml1m_attacker_config(device)[3]
    attacker_config['n_fakes'] = 10
    attacker_config['n_inters'] = 5
    attacker_config['target_item'] = 0
    attacker = get_attacker(attacker_config, dataset)

    coss = []
    for i in range(10):
        set_seed(2021 + i)
        attacker.fake_tensor.data = attacker.init_fake_data()

        attacker.unroll_steps = 49
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