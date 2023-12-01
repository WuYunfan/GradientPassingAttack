import torch
from dataset import get_dataset
from utils import set_seed, init_run
from model import get_model
from trainer import get_trainer
import optuna
import logging
import sys
from optuna.trial import TrialState
from optuna.study import MaxTrialsCallback


def objective(trial):
    lr = trial.suggest_float('lr', 1.e-4, 1.e-1, log=True)
    l2_reg = trial.suggest_float('l2_reg', 1.e-5, 1.e-1, log=True)
    set_seed(2023)
    device = torch.device('cuda')
    dataset_config = {'name': 'ProcessedDataset', 'path': 'data/Gowalla/time',
                      'device': device}
    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': lr, 'l2_reg': l2_reg,
                      'n_epochs': 1000, 'batch_size': 2 ** 14, 'dataloader_num_workers': 6,
                      'test_batch_size': 2048, 'topks': [50]}
    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    trainer = get_trainer(trainer_config, model)
    return trainer.train(verbose=True)


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2023)

    search_space = {'lr': [1.e-4, 1.e-3, 1.e-2, 1.e-1], 'l2_reg': [1.e-5, 1.e-4, 1.e-3, 1.e-2, 1.e-1]}
    optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
    study_name = 'lightgcn-tuning'
    storage = optuna.storages.RDBStorage(url='sqlite:///../{}.db'.format(study_name),
                                         failed_trial_callback=optuna.storages.RetryFailedTrialCallback())
    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True, direction='maximize',
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