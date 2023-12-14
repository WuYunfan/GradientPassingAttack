from utils import init_run
import optuna
import logging
import sys
from optuna.trial import TrialState
from optuna.study import MaxTrialsCallback
from retrain.run import run_new_items_recall
import shutil
import os
import numpy as np


def objective(trial, name, n_epochs, run_method, victim_model):
    log_path = __file__[:-3]
    if os.path.exists(os.path.join(log_path, name)):
        shutil.rmtree(os.path.join(log_path, name))

    lr = trial.suggest_categorical('lr', [1.e-4, 1.e-3, 1.e-2, 1.e-1])
    l2_reg = trial.suggest_categorical('l2_reg', [1.e-5, 1.e-4, 1.e-3, 1.e-2, 1.e-1])
    gp_config = None
    if run_method == 1:
        gp_config = dict()
        gp_config['threshold_odd'] = trial.suggest_categorical('threshold_odd', [-np.inf, 0., np.inf])
        gp_config['threshold_even'] = trial.suggest_categorical('threshold_even', [-np.inf, 0., np.inf])
        gp_config['alpha_odd'] = trial.suggest_categorical('alpha_odd', [1., 10., 100., 1000.]) \
            if gp_config['threshold_odd'] != np.inf else 0.
        gp_config['alpha_even'] = trial.suggest_categorical('alpha_even', [0.1, 1., 10.]) \
            if gp_config['threshold_even'] != np.inf else 0.

    jaccard_sim = run_new_items_recall(log_path, 2023, lr, l2_reg, gp_config,
                                       n_epochs, run_method, victim_model)
    return jaccard_sim


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2023)

    n_epochs = 100
    run_method = 0
    victim_model = 0
    names = {0: 'full_retrain', 1: 'full_retrain_wh_gp'}
    name = names[run_method]

    optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
    study_name = name + '_' + str(n_epochs) + '_' + str(victim_model)
    storage = optuna.storages.RDBStorage(url='sqlite:///../{}.db'.format(study_name))
    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True, direction='maximize',
                                sampler=optuna.samplers.BruteForceSampler())

    study.optimize(lambda trial: objective(trial, name, n_epochs, run_method, victim_model))
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
