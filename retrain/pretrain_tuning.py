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


def objective(trial, n_epochs, pp, victim_model):
    lr = trial.suggest_float('lr', 1.e-5, 1.e-1, log=True)
    l2_reg = trial.suggest_float('l2_reg', 1.e-5, 1.e-1, log=True)
    pp_threshold = trial.suggest_float('pp_threshold', 0., 1., ) if pp else None

    set_seed(2023)
    device = torch.device('cuda')
    config = get_config(device)
    dataset_config, model_config, trainer_config = config[victim_model]

    dataset_config['path'] = dataset_config['path'][:-4] + 'retrain'
    trainer_config['n_epochs'] = n_epochs
    trainer_config['max_patience'] = n_epochs
    trainer_config['lr'] = lr
    trainer_config['l2_reg'] = l2_reg
    trainer_config['pp_threshold'] = pp_threshold
    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    trainer = get_trainer(trainer_config, model)
    return trainer.train(verbose=False)


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2023)

    n_epochs = 100
    pp = True
    victim_model = 0

    search_space = {'lr': [1.e-4, 1.e-3, 1.e-2, 1.e-1], 'l2_reg': [1.e-5, 1.e-4, 1.e-3, 1.e-2, 1.e-1]}
    if pp:
        search_space['pp_threshold'] = [0., 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
    study_name = 'pretrain_model_' + str(n_epochs) + '_' + str(victim_model) + ('_pp' if pp else '')
    storage_name = 'sqlite:///../{}.db'.format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction='maximize',
                                sampler=optuna.samplers.GridSampler(search_space))

    study.optimize(lambda trial: objective(trial, n_epochs, pp, victim_model))
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

    config = get_config('cpu')
    dataset_config, model_config, trainer_config = config[victim_model]
    save_path = '{:s}_{:s}_{:s}_{:.3f}.pth'.format(model_config['name'], trainer_config['name'],
                                                   dataset_config['name'], trial.value * 100)
    save_path = os.path.join('checkpoints', save_path)
    new_path = 'retrain/pretrain_model_' + str(n_epochs) + '_' + str(victim_model) + ('_pp' if pp else '') + '.pth'
    new_path = os.path.join('checkpoints', new_path)
    os.rename(save_path, new_path)


if __name__ == '__main__':
    main()