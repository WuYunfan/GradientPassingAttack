from sklearn.model_selection import ParameterGrid
import torch
import numpy as np
from dataset import get_dataset
from attacker import get_attacker
from utils import set_seed, init_run, get_target_items
from config import get_gowalla_config


def fitness(l2_reg, aux_reg, propagation_order):
    set_seed(2021)
    device = torch.device('cuda')
    dataset_config, model_config, trainer_config = get_gowalla_config(device)[0]
    surrogate_model_config = {'name': 'IMF', 'n_layers': 0, 'embedding_size': 64}
    surrogate_trainer_config = {'name': 'IGCNTrainer', 'optimizer': 'Adam', 'lr': 1.e-2,
                                'l2_reg': l2_reg, 'aux_reg': aux_reg,
                                'n_epochs': 100, 'batch_size': 2048, 'dataloader_num_workers': 6,
                                'test_batch_size': 512, 'topks': [20]}
    attacker_config = {'name': 'ERAP4', 'device': device, 'n_fakes': 131,
                       'n_inters': 41, 'propagation_order': propagation_order,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config}
    dataset = get_dataset(dataset_config)
    target_item = get_target_items(dataset)[0]
    attacker_config['target_item'] = target_item
    attacker = get_attacker(attacker_config, dataset)
    attacker.generate_fake_users()
    return attacker.eval(dataset_config, model_config, trainer_config)


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2021)
    param_grid = {'l2_reg': [1.e-4, 1.e-5, 0.], 'aux_reg': [0.1, 1.e-2, 1.e-3],
                  'propagation_order':  [1, 2, 3]}
    grid = ParameterGrid(param_grid)
    max_hr = -np.inf
    best_params = None
    for params in grid:
        hr = fitness(params['l2_reg'], params['aux_reg'], params['propagation_order'])
        print('Hit ratio: {:.3f}, Parameters: {:s}'.format(hr, str(params)))
        if hr > max_hr:
            max_hr = hr
            best_params = params
            print('Maximum hit ratio!')
    print('Maximum hit ratio: {:.3f}, Best Parameters: {:s}'.format(max_hr, str(best_params)))


if __name__ == '__main__':
    main()
