from dataset import get_dataset
from model import get_model
from trainer import get_trainer
import torch
from utils import init_run, set_seed, AverageMeter
from tensorboardX import SummaryWriter
from config import get_gowalla_config
import numpy as np
import os
from tensorboard.backend.event_processing import event_accumulator


def eval_rec_on_new_users(trainer, n_old_users, writer, verbose):
    val_data = trainer.dataset.val_data.copy()

    for user in range(n_old_users):
        trainer.dataset.val_data[user] = []
    results, metrics = trainer.eval('val')
    if verbose:
        print('New users and all items result. {:s}'.format(results))
    trainer.dataset.val_data = val_data.copy()
    trainer.record(writer, 'new_user', metrics)

    for user in range(n_old_users, trainer.model.n_users):
        trainer.dataset.val_data[user] = []
    results, metrics = trainer.eval('val')
    if verbose:
        print('Old users and all items result. {:s}'.format(results))
    trainer.dataset.val_data = val_data.copy()
    trainer.record(writer, 'old_user', metrics)


def initial_parameter(new_model, pre_train_model):
    dataset = pre_train_model.dataset
    with torch.no_grad():
        new_model.embedding.weight.data[:dataset.n_users, :] = pre_train_model.embedding.weight[:dataset.n_users, :]
        new_model.embedding.weight.data[-dataset.n_items:, :] = pre_train_model.embedding.weight[-dataset.n_items:, :]


def jaccard_similarity(list1, list2):
    intersection = len(np.intersect1d(list1, list2))
    union = len(np.union1d(list1, list2))
    return 1. * intersection / union


def eval_rec_and_surrogate(trainer, n_old_users, full_rec_items, writer, verbose):
    eval_rec_on_new_users(trainer, n_old_users, writer, verbose)
    jaccard_sim = 0
    n = full_rec_items.shape[0]
    rec_items = trainer.get_rec_items('test', None)[:n, :]
    for i in range(n):
        jaccard_sim += jaccard_similarity(rec_items[i], full_rec_items[i])
    jaccard_sim /= n
    if verbose:
        print('Jaccard similarity {:.4f}'.format(jaccard_sim))
    writer.add_scalar('{:s}_{:s}/Jaccard_similarity'.format(trainer.model.name, trainer.name), jaccard_sim, trainer.epoch)
    return jaccard_sim


def run_new_items_recall(log_path, seed, lr, l2_reg, pp_alpha, n_epochs, run_method, trial=None, verbose=False):
    device = torch.device('cuda')
    config = get_gowalla_config(device)
    dataset_config, model_config, trainer_config = config[0]
    dataset_config['path'] = dataset_config['path'][:-4] + 'retrain'
    trainer_config['max_patience'] = 1000

    sub_dataset = get_dataset(dataset_config)
    pre_train_model = get_model(model_config, sub_dataset)
    if os.path.exists('retrain/pre_train_model.pth'):
        pre_train_model.load('retrain/pre_train_model.pth')
    else:
        trainer = get_trainer(trainer_config, pre_train_model)
        trainer.train(verbose=False)
        pre_train_model.save('retrain/pre_train_model.pth')

    dataset_config['path'] = dataset_config['path'][:-7] + 'time'
    full_dataset = get_dataset(dataset_config)
    full_train_model = get_model(model_config, full_dataset)
    trainer = get_trainer(trainer_config, full_train_model)
    if os.path.exists('retrain/full_train_model.pth'):
        full_train_model.load('retrain/full_train_model.pth')
    else:
        trainer.train(verbose=False)
        full_train_model.save('retrain/full_train_model.pth')
    full_rec_items = trainer.get_rec_items('test', None)[:sub_dataset.n_users, :]

    trainer_config['n_epochs'] = n_epochs if n_epochs is not None else trainer_config['n_epochs']
    trainer_config['lr'] = lr if lr is not None else trainer_config['lr']
    trainer_config['l2_reg'] = l2_reg if l2_reg is not None else trainer_config['l2_reg']
    names = {0: 'full_retrain', 1: 'part_retrain', 2: 'pp_retrain'}
    if run_method == 0:
        writer = SummaryWriter(os.path.join(log_path, names[run_method]))
        set_seed(seed)
        new_model = get_model(model_config, full_dataset)
        new_trainer = get_trainer(trainer_config, new_model)
        extra_eval = (eval_rec_and_surrogate, (sub_dataset.n_users, full_rec_items))
        new_trainer.train(verbose=verbose, writer=writer, extra_eval=extra_eval, trial=trial)
        writer.close()
        print('Limited full Retrain!')

    if run_method == 1:
        writer = SummaryWriter(os.path.join(log_path, names[run_method]))
        set_seed(seed)
        new_model = get_model(model_config, full_dataset)
        new_trainer = get_trainer(trainer_config, new_model)
        initial_parameter(new_model, pre_train_model)
        extra_eval = (eval_rec_and_surrogate, (sub_dataset.n_users, full_rec_items))
        new_trainer.train(verbose=verbose, writer=writer, extra_eval=extra_eval, trial=trial)
        writer.close()
        print('Part Retrain!')

    if run_method == 2:
        trainer_config['pp_alpha'] = pp_alpha
        writer = SummaryWriter(os.path.join(log_path, names[run_method]))
        set_seed(seed)
        new_model = get_model(model_config, full_dataset)
        new_trainer = get_trainer(trainer_config, new_model)
        initial_parameter(new_model, pre_train_model)
        extra_eval = (eval_rec_and_surrogate, (sub_dataset.n_users, full_rec_items))
        new_trainer.train(verbose=verbose, writer=writer, extra_eval=extra_eval, trial=trial)
        writer.close()
        print('Retrain with parameter propagation!')

    ea = event_accumulator.EventAccumulator(os.path.join(log_path, names[run_method]))
    ea.Reload()
    kl_divergences = ea.Scalars('{:s}_{:s}/Jaccard_similarity'.format(new_trainer.model.name, new_trainer.name))
    kl_divergences = [x.value for x in kl_divergences]
    return kl_divergences[-1]


def main():
    seed_list = [2023, 42, 0, 131, 1024]
    seed = seed_list[0]
    log_path = __file__[:-3]
    init_run(log_path, seed)

    lr = None
    l2_reg = None
    pp_alpha = None
    n_epochs = 1000
    run_method = 0
    jaccard_sim = run_new_items_recall(log_path, seed, lr, l2_reg, pp_alpha, n_epochs, run_method)
    print('Jaccard similarity', jaccard_sim)


if __name__ == '__main__':
    main()
