from utils import init_run
import optuna
import logging
import sys
from optuna.trial import TrialState
from optuna.study import MaxTrialsCallback
from retrain.run import run_new_items_recall
import shutil
import os


def objective(trial, name, pretrain_fixed_dim, n_epochs, run_method, victim_model):
    log_path = __file__[:-3]
    if os.path.exists(os.path.join(log_path, name)):
        shutil.rmtree(os.path.join(log_path, name))

    lr = trial.suggest_float('lr', 1.e-5, 1.e-1, log=True)
    l2_reg = trial.suggest_float('l2_reg', 1.e-5, 1.e-1, log=True)
    pp_threshold = trial.suggest_float('pp_threshold', 0., 1.,) if run_method >= 2 else None

    jaccard_sim = run_new_items_recall(log_path, 2023, lr, l2_reg, pp_threshold, pretrain_fixed_dim,
                                       n_epochs, run_method, victim_model)
    return jaccard_sim


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2023)

    pretrain_fixed_dim = 4
    n_epochs = 100
    run_method = 2
    victim_model = 0
    names = {0: 'full_retrain', 1: 'pre_retrain', 2: 'full_retrain_wh_pp', 3: 'pre_retrain_wh_pp'}
    name = names[run_method]

    search_space = {'lr': [1.e-4, 1.e-3, 1.e-2, 1.e-1], 'l2_reg': [1.e-5, 1.e-4, 1.e-3, 1.e-2, 1.e-1]}
    if run_method >= 2:
        search_space['pp_threshold'] = [0., 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
    study_name = name + '_' + str(n_epochs) + '_' + str(victim_model)
    storage_name = 'sqlite:///../{}.db'.format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction='maximize',
                                sampler=optuna.samplers.GridSampler(search_space))

    study.optimize(lambda trial: objective(trial, name, pretrain_fixed_dim, n_epochs, run_method, victim_model))
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
