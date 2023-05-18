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
    log_path = __file__[:-3]
    if os.path.exists(os.path.join(log_path, 'pp_retrain')):
        shutil.rmtree(os.path.join(log_path, 'pp_retrain'))
    pp_step = trial.suggest_int('pp_step', 1, 3)
    pp_alpha = trial.suggest_float('pp_alpha', 1.e-3, 1., log=True)
    bernoulli_p = trial.suggest_float('bernoulli_p', 0., 1.)
    kl_divergence = run_new_items_recall(pp_step, pp_alpha, bernoulli_p, log_path, 2023, trial=trial)
    return kl_divergence


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2023)

    optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
    study_name = 'retrain-tuning'
    storage_name = 'sqlite:///../{}.db'.format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction='minimize')

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
