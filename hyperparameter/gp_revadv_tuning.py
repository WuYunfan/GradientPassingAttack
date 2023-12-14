from attacker import get_attacker
from utils import set_seed, init_run, get_target_items
from config import get_gowalla_config as get_config
import torch
from dataset import get_dataset
import optuna
import logging
import sys
from optuna.trial import TrialState
from optuna.study import MaxTrialsCallback
import shutil
import numpy as np


def objective(trial):
    lr = None
    s_lr = None
    s_l2 = None
    m_momentum = 0.05

    gp_config = dict()
    gp_config['threshold_odd'] = trial.suggest_categorical('threshold_odd', [-np.inf, 0., np.inf])
    gp_config['threshold_even'] = trial.suggest_categorical('threshold_even', [-np.inf, 0., np.inf])
    gp_config['alpha_odd'] = trial.suggest_categorical('alpha_odd', [1., 10., 100.]) \
        if gp_config['threshold_odd'] != np.inf else 0.
    gp_config['alpha_even'] = trial.suggest_categorical('alpha_even', [1.]) \
        if gp_config['threshold_even'] != np.inf else 0.

    set_seed(2023)
    device = torch.device('cuda')
    dataset_config, model_config, trainer_config = get_config(device)[0]
    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'RevAdvBCETrainer', 'optimizer': 'Adam', 'lr': s_lr, 'l2_reg': s_l2,
                                'n_epochs': 9, 'batch_size': 2048,
                                'test_batch_size': 2048, 'topks': [50], 'verbose': False, 'val_interval': 100,
                                'gp_config': gp_config}
    attacker_config = {'name': 'RevAdv', 'lr': lr, 'momentum': 1. - m_momentum,
                       'n_fakes': 131, 'unroll_steps': 1, 'n_inters': 41, 'topk': 50, 'adv_epochs': 30,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config}

    dataset = get_dataset(dataset_config)
    target_items = get_target_items(dataset)
    attacker_config['target_items'] = target_items
    dataset = get_dataset(dataset_config)
    attacker = get_attacker(attacker_config, dataset)
    attacker.generate_fake_users(verbose=False)
    hr = attacker.eval(model_config, trainer_config)
    shutil.rmtree('checkpoints')
    return hr


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2023)

    optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
    study_name = 'gp_revadv-tuning'
    storage = optuna.storages.RDBStorage(url='sqlite:///../{}.db'.format(study_name))
    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True, direction='maximize',
                                sampler=optuna.samplers.BruteForceSampler())

    study.optimize(objective)
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