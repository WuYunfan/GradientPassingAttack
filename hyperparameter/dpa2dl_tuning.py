from attacker import get_attacker
from utils import set_seed, init_run, get_target_items
from config import get_gowalla_config
import torch
from dataset import get_dataset
import optuna
import logging
import sys
from optuna.trial import TrialState


def objective(trial):
    reg_u = trial.suggest_float('reg_u', 1, 1e3, log=True)
    alpha = trial.suggest_float('alpha', 1e-5, 1e-3, log=True)
    s_l2 = trial.suggest_float('s_l2', 1e-5, 1e-2, log=True)
    lr = trial.suggest_float('lr', 1.e-4, 1.e-1, log=True)
    set_seed(2023)
    device = torch.device('cuda')
    dataset_config, model_config, trainer_config = get_gowalla_config(device)[0]
    surrogate_model_config = {'name': 'NeuMF', 'embedding_size': 64, 'device': device, 'layer_sizes': [64, 64, 64]}
    surrogate_trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': lr, 'l2_reg': s_l2,
                                'n_epochs': 20, 'batch_size': 2 ** 12, 'dataloader_num_workers': 16,
                                'test_batch_size': 64, 'topks': [50], 'neg_ratio': 4,
                                'mf_pretrain_epochs': 0, 'mlp_pretrain_epochs': 0}
    attacker_config = {'name': 'DPA2DL', 'device': device, 'n_fakes': 131, 'topk': 50,
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
    study_name = 'mf-tuning'
    storage_name = 'sqlite:///../{}.db'.format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction='maximize')

    study.optimize(objective, n_trials=50)
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