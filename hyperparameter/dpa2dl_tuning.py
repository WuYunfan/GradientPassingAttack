from sklearn.model_selection import ParameterGrid
import torch
import numpy as np
from dataset import get_dataset
from attacker import get_attacker
from utils import set_seed, init_run, get_target_items
from config import get_gowalla_config


def fitness(reg_u, alpha, kapaa, prob, s_l2, n_rounds, n_pretrain_epochs, lr):
    set_seed(2021)
    device = torch.device('cuda')
    dataset_config, model_config, trainer_config = get_gowalla_config(device)[0]
    surrogate_model_config = {'name': 'NeuMF', 'embedding_size': 64, 'device': device, 'layer_sizes': [64, 64, 64]}
    surrogate_trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': lr, 'l2_reg': s_l2,
                                'n_epochs': n_pretrain_epochs, 'batch_size': 2 ** 12, 'dataloader_num_workers': 16,
                                'test_batch_size': 64, 'topks': [20], 'neg_ratio': 4,
                                'mf_pretrain_epochs': 0, 'mlp_pretrain_epochs': 0}
    attacker_config = {'name': 'DPA2DL', 'device': device, 'n_fakes': 131, 'topk': 20,
                       'n_inters': 41, 'reg_u': reg_u, 'prob': prob, 'kappa': kapaa,
                       'step': 4, 'alpha': alpha, 'n_rounds': n_rounds,
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
    param_grid = {'reg_u': [1e2, 1e3, 1e4], 'alpha': [1.e-4, 1.e-3, 1.e-2], 'kapaa': [1.],
                  'prob':  [0.9], 's_l2': [1.e-2, 1.e-3, 0.], 'n_rounds': [5],
                  'n_pretrain_epochs': [20], 'lr': [1.e-2, 1.e-3]}
    grid = ParameterGrid(param_grid)
    max_hr = -np.inf
    best_params = None
    for params in grid:
        hr = fitness(params['reg_u'], params['alpha'], params['kapaa'], params['prob'],
                     params['s_l2'], params['n_rounds'], params['n_pretrain_epochs'], params['lr'])
        print('Hit ratio: {:.3f}, Parameters: {:s}'.format(hr, str(params)))
        if hr > max_hr:
            max_hr = hr
            best_params = params
            print('Maximum hit ratio!')
    print('Maximum hit ratio: {:.3f}, Best Parameters: {:s}'.format(max_hr, str(best_params)))


if __name__ == '__main__':
    main()
