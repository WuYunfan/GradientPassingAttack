from sklearn.model_selection import ParameterGrid
import torch
import numpy as np
from dataset import get_dataset
from attacker import get_attacker
from utils import set_seed, init_run, get_target_items
from config import get_gowalla_config


def fitness(lr, s_lr, s_l2, momentum):
    set_seed(2021)
    device = torch.device('cuda')
    dataset_config, model_config, trainer_config = get_gowalla_config(device)[0]
    surrogate_config = {'layer_sizes': [64, 32], 'lr': s_lr, 'l2_reg': s_l2}
    attacker_config = {'name': 'ItemAESGD', 'lr': lr, 'momentum': momentum, 'batch_size': 2048,
                       'device': device, 'n_fakes': 131, 'unroll_steps': 5, 'train_epochs': 50,
                       'n_inters': 41, 'topk': 20, 'weight': 20., 'adv_epochs': 30,
                       'surrogate_config': surrogate_config}
    dataset = get_dataset(dataset_config)
    target_item = get_target_items(dataset)[0]
    attacker_config['target_item'] = target_item
    attacker = get_attacker(attacker_config, dataset)
    attacker.generate_fake_users()
    return attacker.eval(model_config, trainer_config)


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2021)
    param_grid = {'lr': [0.1, 1., 10.], 's_lr': [1.e-2, 1.e-3],
                  's_l2':  [1.e-4, 1.e-5, 0.], 'momentum': [0.9, 0.99]}
    grid = ParameterGrid(param_grid)
    max_hr = -np.inf
    best_params = None
    for params in grid:
        hr = fitness(params['lr'], params['s_lr'], params['s_l2'], params['momentum'])
        print('Hit ratio: {:.3f}, Parameters: {:s}'.format(hr, str(params)))
        if hr > max_hr:
            max_hr = hr
            best_params = params
            print('Maximum hit ratio!')
    print('Maximum hit ratio: {:.3f}, Best Parameters: {:s}'.format(max_hr, str(best_params)))


if __name__ == '__main__':
    main()
