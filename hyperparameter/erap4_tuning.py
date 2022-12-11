from sklearn.model_selection import ParameterGrid
import torch
import numpy as np
from dataset import get_dataset
from attacker import get_attacker
from utils import set_seed, init_run, get_target_items
from config import get_gowalla_config


def fitness(lr, momentum, s_lr, s_l2, propagation_order, alpha, n_pretrain_epochs):
    set_seed(2021)
    device = torch.device('cuda')
    dataset_config, model_config, trainer_config = get_gowalla_config(device)[0]
    surrogate_model_config = {'name': 'SurrogateERAP4MF', 'embedding_size': 64, 'propagation_order': propagation_order}
    surrogate_trainer_config = {'name': 'EPAR4Trainer', 'optimizer': 'Adam', 'lr': 1.e-2, 'weight': alpha,
                                'l2_reg': s_l2,  'n_epochs': n_pretrain_epochs, 'test_batch_size': 2048}
    attacker_config = {'name': 'ERAP4', 'lr': lr, 'momentum': momentum, 's_lr': s_lr,
                       'device': device, 'n_fakes': 131, 'unroll_steps': 3,
                       'n_inters': 41, 'topk': 20, 'kappa': 1., 'adv_epochs': 30,
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
    param_grid = {'lr': [0.1, 1., 10.], 's_lr': [1.e-2, 1.e-3], 's_l2':  [1.e-1, 1.e-2, 1.e-3],
                  'momentum': [0., 0.9, 0.99], 'propagation_order': [2], 'alpha': [20.],
                  'n_pretrain_epochs': [200]}
    grid = ParameterGrid(param_grid)
    max_hr = -np.inf
    best_params = None
    for params in grid:
        hr = fitness(params['lr'], params['momentum'], params['s_lr'], params['s_l2'],
                     params['propagation_order'], params['alpha'], params['n_pretrain_epochs'])
        print('Hit ratio: {:.3f}, Parameters: {:s}'.format(hr, str(params)))
        if hr > max_hr:
            max_hr = hr
            best_params = params
            print('Maximum hit ratio!')
    print('Maximum hit ratio: {:.3f}, Best Parameters: {:s}'.format(max_hr, str(best_params)))


if __name__ == '__main__':
    main()
