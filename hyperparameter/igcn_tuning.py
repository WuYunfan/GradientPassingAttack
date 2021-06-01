from sklearn.model_selection import ParameterGrid
from dataset import get_dataset
from trainer import get_trainer
import torch
import numpy as np
from attacker.gbfug_attacker import IGCN
from utils import set_seed, init_run


def fitness(lr, l2_reg, n_layers, dropout):
    set_seed(2021)
    device = torch.device('cuda')
    dataset_config = {'name': 'ML1MDataset', 'min_inter': 10, 'path': 'data/ML1M',
                      'split_ratio': [0.8, 0.2], 'device': device}
    model_config = {'name': 'IGCN', 'n_layers': n_layers, 'dropout': dropout, 'feature_ratio': 1.,
                    'embedding_size': 64, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': lr, 'l2_reg': l2_reg,
                      'device': device, 'n_epochs': 200, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20, 100]}
    dataset = get_dataset(dataset_config)
    model_config['dataset'] = dataset
    model = IGCN(model_config)
    trainer = get_trainer(trainer_config, dataset, model)
    return trainer.train()


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2021)
    param_grid = {'lr': [1.e-2, 1.e-3], 'l2_reg':  [1.e-4, 1.e-5, 0.],
                  'n_layers': [2, 3, 4], 'dropout': [0.3, 0.5, 0.7]}
    grid = ParameterGrid(param_grid)
    max_ndcg = -np.inf
    best_params = None
    for params in grid:
        ndcg = fitness(params['lr'], params['l2_reg'], params['n_layers'], params['dropout'])
        print('NDCG: {:.3f}, Parameters: {:s}'.format(ndcg, str(params)))
        if ndcg > max_ndcg:
            max_ndcg = ndcg
            best_params = params
            print('Maximum NDCG!')
    print('Maximum NDCG: {:.3f}, Best Parameters: {:s}'.format(max_ndcg, str(best_params)))


if __name__ == '__main__':
    main()