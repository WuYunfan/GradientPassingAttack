from sklearn.model_selection import ParameterGrid
import torch
import numpy as np
from dataset import get_dataset
from attacker import get_attacker
from utils import set_seed, init_run


def fitness(lr, s_lr, s_l2):
    set_seed(2021)
    device = torch.device('cuda')
    dataset_config = {'name': 'ML1MDataset', 'min_inter': 10, 'path': 'data/ML1M',
                      'split_ratio': [0.8, 0.2], 'device': device}
    model_config = {'name': 'MF', 'embedding_size': 64, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-2,
                      'device': device, 'n_epochs': 200, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20, 100]}
    surrogate_config = {'embedding_size': 64, 'lr': s_lr, 'l2_reg': s_l2, 'weight': 20.,
                        'n_epochs': 50, 'unroll_steps': 5}
    attacker_config = {'name': 'WRMF_SGD', 'lr': lr, 'momentum': 0.95, 'batch_size': 2048,
                       'dataloader_num_workers': 0, 'device': device, 'n_fakes': 59,
                       'n_inters': 96, 'target_item': 2021, 'topk': 20,
                       'adv_epochs': 100, 'surrogate_config': surrogate_config}
    dataset = get_dataset(dataset_config)
    attacker = get_attacker(attacker_config, dataset)
    attacker.generate_fake_users(verbose=True)
    return attacker.eval(model_config, trainer_config, verbose=True)


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2021)
    param_grid = {'lr': [0.1, 1., 10.], 's_lr': [1.e-1, 1.e-2, 1.e-3], 's_l2':  [1.e-4, 1.e-5, 1.e-6]}
    grid = ParameterGrid(param_grid)
    max_ndcg = -np.inf
    best_params = None
    for params in grid:
        ndcg = fitness(params['lr'], params['s_lr'], params['s_l2'])
        print('NDCG: {:.3f}, Parameters: {:s}'.format(ndcg, str(params)))
        if ndcg > max_ndcg:
            max_ndcg = ndcg
            best_params = params
            print('Maximum NDCG!')
    print('Maximum NDCG: {:.3f}, Best Parameters: {:s}'.format(max_ndcg, str(best_params)))


if __name__ == '__main__':
    main()
