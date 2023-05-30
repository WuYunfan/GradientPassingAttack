from utils import init_run
import optuna
import logging
import sys
from optuna.trial import TrialState
from optuna.study import MaxTrialsCallback
from retrain.run import run_new_items_recall
import shutil
import os


def objective(trial):
    n_epochs = 100
    run_method = 2
    names = {0: 'full_retrain', 1: 'part_retrain', 2: 'pp_retrain'}
    name = names[run_method]

    log_path = __file__[:-3]
    if os.path.exists(os.path.join(log_path, name)):
        shutil.rmtree(os.path.join(log_path, name))

    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-1, log=True)
    pp_alpha = None if run_method != 2 else trial.suggest_float('pp_alpha', 1.e-3, 1., log=True)
    pp_threshold = None if run_method != 2 else trial.suggest_float('pp_threshold', 0., 1.,)

    jaccard_sim = run_new_items_recall(log_path, 2023, lr, l2_reg,
                                       pp_threshold, pp_alpha, n_epochs, run_method, trial=trial)
    return jaccard_sim


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2023)

    optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
    study_name = 'retrain-tuning'
    storage_name = 'sqlite:///../{}.db'.format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction='maximize')

    call_back = MaxTrialsCallback(100, states=(TrialState.RUNNING, TrialState.COMPLETE, TrialState.PRUNED))
    study.optimize(objective, callbacks=[call_back])
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
