from sklearn.model_selection import ParameterGrid
import torch
import numpy as np
from dataset import get_dataset
from utils import set_seed, init_run
from model import get_model
from trainer import get_trainer


def fitness(eps, adv_reg):
    set_seed(2021)
    device = torch.device('cuda')
    dataset_config = {'name': 'ProcessedDataset', 'path': 'data/Gowalla/time',
                      'device': device}
    model_config = {'name': 'MF', 'embedding_size': 64, 'device': device}
    trainer_config = {'name': 'APRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-3,
                      'eps': eps, 'adv_reg': adv_reg, 'ckpt_path': 'checkpoints/pretrain_mf.pth',
                      'device': device, 'n_epochs': 1000, 'batch_size': 2 ** 14, 'dataloader_num_workers': 16,
                      'test_batch_size': 2048, 'topks': [50]}
    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    trainer = get_trainer(trainer_config, dataset, model)
    return trainer.train(verbose=True)


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2021)
    param_grid = {'eps': [0.01, 0.1, 1., 10.], 'adv_reg':  [0.001, 0.01, 0.1, 1., 10.]}
    grid = ParameterGrid(param_grid)
    max_ndcg = -np.inf
    best_params = None
    for params in grid:
        ndcg = fitness(params['eps'], params['adv_reg'])
        print('NDCG: {:.3f}, Parameters: {:s}'.format(ndcg, str(params)))
        if ndcg > max_ndcg:
            max_ndcg = ndcg
            best_params = params
            print('Maximum NDCG!')
    print('Maximum NDCG: {:.3f}, Best Parameters: {:s}'.format(max_ndcg, str(best_params)))


if __name__ == '__main__':
    main()