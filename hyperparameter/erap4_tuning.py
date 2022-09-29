from sklearn.model_selection import ParameterGrid
import torch
import numpy as np
from dataset import get_dataset
from attacker import get_attacker
from utils import set_seed, init_run, get_target_items
from config import get_gowalla_config


def fitness(re_lr, aux_reg, s_l2, propagation_order, lr, momentum):
    set_seed(2021)
    device = torch.device('cuda')
    dataset_config, model_config, trainer_config = get_gowalla_config(device)[0]
    surrogate_model_config = {'name': 'IMF', 'n_layers': 0, 'embedding_size': 64}
    surrogate_trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-3,
                                'l2_reg': s_l2, 'aux_reg': aux_reg, 'neg_ratio': 4,
                                'n_epochs': 1000, 'batch_size': 2 ** 14, 'dataloader_num_workers': 16,
                                'test_batch_size': 2048, 'topks': [20]}
    attacker_config = {'name': 'ERAP4', 'device': device, 'n_fakes': 131, 're_lr': re_lr, 'topk': 20,
                       'n_inters': 41, 'lr': lr, 'momentum': momentum, 'adv_epochs': 30,
                       'propagation_order': propagation_order,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config}
    dataset = get_dataset(dataset_config)
    target_item = get_target_items(dataset)[0]
    attacker_config['target_item'] = target_item
    attacker = get_attacker(attacker_config, dataset)
    attacker.generate_fake_users()
    return attacker.eval(model_config, trainer_config)


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2021)
    param_grid = {'re_lr': [0.01, 0.1, 1.], 'aux_reg': [1.e-2, 1.e-3], 's_l2': [1.e-4, 0.],
                  'propagation_order':  [1, 2], 'lr': [0.1, 1., 10.], 'momentum': [0.9, 0.99]}
    grid = ParameterGrid(param_grid)
    max_hr = -np.inf
    best_params = None
    for params in grid:
        hr = fitness(params['re_lr'], params['aux_reg'], params['s_l2'],
                     params['propagation_order'], params['lr'], params['momentum'])
        print('Hit ratio: {:.3f}, Parameters: {:s}'.format(hr, str(params)))
        if hr > max_hr:
            max_hr = hr
            best_params = params
            print('Maximum hit ratio!')
    print('Maximum hit ratio: {:.3f}, Best Parameters: {:s}'.format(max_hr, str(best_params)))


if __name__ == '__main__':
    main()
