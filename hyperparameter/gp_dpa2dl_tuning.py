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
    reg_u = None
    alpha = None
    s_l2 = None
    s_lr = None

    gp_proportion = trial.suggest_float('gp_proportion', 0., 1.,)
    set_seed(2023)
    device = torch.device('cuda')
    dataset_config, model_config, trainer_config = get_config(device)[-4]
    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': s_lr, 'l2_reg': s_l2,
                                'n_epochs': 5, 'batch_size': 2 ** 12, 'dataloader_num_workers': 6,
                                'test_batch_size': 2048, 'topks': [50], 'neg_ratio': 4, 'verbose': False,
                                'val_interval': 100, 'gp_proportion': gp_proportion}
    attacker_config = {'name': 'DPA2DL', 'n_fakes': 131, 'topk': 50,
                       'n_inters': 41, 'reg_u': reg_u, 'prob': 0.9, 'kappa': 1.,
                       'step': 4, 'alpha': alpha, 'n_rounds': 1,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config}

    trainer_config['n_epochs'] = trainer_config['n_epochs'] // 10
    dataset = get_dataset(dataset_config)
    target_items = get_target_items(dataset)  # [:4] for yelp, [:2] for tenrec
    hits = []
    for target_item in target_items:
        attacker_config['target_item'] = target_item
        dataset = get_dataset(dataset_config)
        attacker = get_attacker(attacker_config, dataset)
        attacker.generate_fake_users(verbose=False)
        hits.append(attacker.eval(model_config, trainer_config))
        shutil.rmtree('checkpoints')
    return np.mean(hits)


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2023)

    search_space = {'gp_threshold': [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]}
    optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
    study_name = 'gp_dpa2dl-tuning'
    storage_name = 'sqlite:///../{}.db'.format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction='maximize',
                                sampler=optuna.samplers.GridSampler(search_space))

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