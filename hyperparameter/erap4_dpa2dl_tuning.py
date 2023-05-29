from attacker import get_attacker
from utils import set_seed, init_run, get_target_items
from config import get_gowalla_config
import torch
from dataset import get_dataset
import optuna
import logging
import sys
from optuna.trial import TrialState
from optuna.study import MaxTrialsCallback


def objective(trial):
    reg_u = trial.suggest_float('reg_u', 1, 1e3, log=True)
    alpha = trial.suggest_float('alpha', 1e-6, 1e-4, log=True)
    s_l2 = trial.suggest_float('s_l2', 1e-4, 1e-1, log=True)
    s_lr = trial.suggest_float('s_lr', 1.e-4, 1.e-1, log=True)

    pp_step = trial.suggest_int('pp_step', 1, 3)
    pp_alpha = trial.suggest_float('pp_alpha', 1.e-3, 1., log=True)
    bernoulli_p = trial.suggest_float('bernoulli_p', 0., 1.)
    set_seed(2023)
    device = torch.device('cuda')
    dataset_config, model_config, trainer_config = get_gowalla_config(device)[0]
    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': s_lr, 'l2_reg': s_l2,
                                'n_epochs': 20, 'batch_size': 2 ** 12, 'dataloader_num_workers': 2,
                                'test_batch_size': 2048, 'topks': [50], 'neg_ratio': 4, 'verbose': False,
                                'val_interval': 100, 'pp_alpha': pp_alpha, 'pp_step': pp_step}
    attacker_config = {'name': 'DPA2DL', 'n_fakes': 131, 'topk': 50, 'bernoulli_p': bernoulli_p,
                       'n_inters': 41, 'reg_u': reg_u, 'prob': 0.9, 'kappa': 1.,
                       'step': 4, 'alpha': alpha, 'n_rounds': 5,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config}
    dataset = get_dataset(dataset_config)
    target_item = get_target_items(dataset, 0.1)[0]
    attacker_config['target_item'] = target_item
    attacker = get_attacker(attacker_config, dataset)
    attacker.generate_fake_users()
    return attacker.eval(model_config, trainer_config)


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2023)

    optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
    study_name = 'erap4_dpa2dl-tuning'
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