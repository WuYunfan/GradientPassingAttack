from utils import init_run
import optuna
import logging
import sys
from optuna.trial import TrialState
from optuna.study import MaxTrialsCallback
from retrain.run import run_new_items_recall


def objective(trial):
    pp_step = trial.suggest_int('pp_step', 1, 3)
    m_pp_threshold = trial.suggest_float('m_pp_threshold', 1e-4, 1., log=True)
    bernoulli_p = trial.suggest_float('bernoulli_p', 1e-3, 1., log=True)
    return run_new_items_recall(pp_step, m_pp_threshold, bernoulli_p)


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2023)

    optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
    study_name = 'retrain-tuning'
    storage_name = 'sqlite:///../{}.db'.format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction='maximize')

    call_back = MaxTrialsCallback(50, states=(TrialState.RUNNING, TrialState.COMPLETE, TrialState.PRUNED))
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