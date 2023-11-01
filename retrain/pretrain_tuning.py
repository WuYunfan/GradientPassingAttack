import torch
from dataset import get_dataset
from utils import set_seed, init_run
from model import get_model
from trainer import get_trainer
import optuna
import logging
import sys
from optuna.trial import TrialState
from config import get_gowalla_config as get_config
import os
import shutil


def objective(trial, pretrain_fixed_dim, victim_model):
    lr = trial.suggest_float('lr', 1.e-4, 1.e-1, log=True)
    l2_reg = trial.suggest_float('l2_reg', 1.e-5, 1.e-1, log=True)

    set_seed(2023)
    device = torch.device('cuda')
    config = get_config(device)
    dataset_config, model_config, trainer_config = config[victim_model]

    dataset_config['path'] = dataset_config['path'][:-4] + 'retrain'
    model_config['embedding_size'] = pretrain_fixed_dim
    trainer_config['lr'] = lr
    trainer_config['l2_reg'] = l2_reg
    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    trainer = get_trainer(trainer_config, model)
    return trainer.train(verbose=False)


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2023)

    pretrain_fixed_dim = 4
    victim_model = 0

    search_space = {'lr': [1.e-4, 1.e-3, 1.e-2, 1.e-1], 'l2_reg': [1.e-5, 1.e-4, 1.e-3, 1.e-2, 1.e-1]}
    optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
    study_name = 'pretrain_model_' + str(pretrain_fixed_dim) + '_' + str(victim_model)
    storage_name = 'sqlite:///../{}.db'.format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction='maximize',
                                sampler=optuna.samplers.GridSampler(search_space))

    study.optimize(lambda trial: objective(trial, pretrain_fixed_dim, victim_model))
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print('Study statistics: ')
    print('  Number of finished trials: ', len(study.trials))
    print('  Number of pruned trials: ', len(pruned_trials))
    print('  Number of complete trials: ', len(complete_trials))

    print('Best trial:')
    trial = study.best_trial
    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))


if __name__ == '__main__':
    main()