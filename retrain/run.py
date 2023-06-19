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
import time


def eval_rec_on_new_users(trainer, n_old_users, writer):
    val_data = trainer.dataset.val_data.copy()

    for user in range(n_old_users):
        trainer.dataset.val_data[user] = []
    results, metrics = trainer.eval('val')
    print('New users and all items result. {:s}'.format(results))
    trainer.dataset.val_data = val_data.copy()
    if writer:
        trainer.record(writer, 'new_user', metrics)

    for user in range(n_old_users, trainer.model.n_users):
        trainer.dataset.val_data[user] = []
    results, metrics = trainer.eval('val')
    print('Old users and all items result. {:s}'.format(results))
    trainer.dataset.val_data = val_data.copy()
    if writer:
        trainer.record(writer, 'old_user', metrics)


def initial_parameter(new_model, pre_train_model):
    dataset = pre_train_model.dataset
    with torch.no_grad():
        new_model.embedding.weight.data[:dataset.n_users, :] = pre_train_model.embedding.weight[:dataset.n_users, :]
        new_model.embedding.weight.data[-dataset.n_items:, :] = pre_train_model.embedding.weight[-dataset.n_items:, :]


def calculate_jaccard_similarity(rec_items, full_rec_items):
    n = rec_items.shape[0]
    jaccard_sims = np.zeros((n, ), dtype=np.float32)
    for i in range(n):
        intersection = np.intersect1d(rec_items[i, :], full_rec_items[i, :]).shape[0]
        union = np.union1d(rec_items[i, :], full_rec_items[i, :]).shape[0]
        jaccard_sims[i] = 1. * intersection / union
    return jaccard_sims


def calculate_ndcg(rec_items, full_rec_items, denominator):
    n, k = rec_items.shape[0], rec_items.shape[1]
    hit_matrix = np.zeros_like(rec_items, dtype=np.float32)
    for user in range(n):
        for item_idx in range(k):
            if rec_items[user, item_idx] in full_rec_items[user]:
                hit_matrix[user, item_idx] = 1.

    dcgs = np.sum(hit_matrix / denominator[:, :k], axis=1)
    idcgs = np.sum(1. / denominator[:, :k], axis=1)
    ndcgs = dcgs / idcgs
    return ndcgs


def eval_rec_and_surrogate(trainer, n_old_users, full_rec_items, topks, writer, verbose):
    if not verbose:
        return
    start_time = time.time()
    eval_rec_on_new_users(trainer, n_old_users, writer)
    n = full_rec_items.shape[0]
    rec_items = trainer.get_rec_items('test', k=max(topks))[:n, :]
    metrics = {'Jaccard': {}, 'NDCG': {}}

    denominator = np.log2(np.arange(2, max(topks) + 2, dtype=np.float32))[None, :]
    for k in topks:
        metrics['Jaccard'][k] = np.mean(calculate_jaccard_similarity(rec_items[:, :k], full_rec_items[:, :k]))
        metrics['NDCG'][k] = np.mean(calculate_ndcg(rec_items[:, :k], full_rec_items[:, :k], denominator))

    jaccard = ''
    ndcg = ''
    for k in topks:
        jaccard += '{:.3f}%@{:d}, '.format(metrics['Jaccard'][k] * 100., k)
        ndcg += '{:.3f}%@{:d}, '.format(metrics['NDCG'][k] * 100., k)
    results = 'Jaccard similarity: {:s}NDCG: {:s}'.format(jaccard, ndcg)
    consumed_time = time.time() - start_time
    print(results, 'Time: {:.3f}s'.format(consumed_time))
    if writer is not None:
        trainer.record(writer, 'surrogate', metrics, topks)
    return metrics['Jaccard'][topks[0]]


def run_new_items_recall(log_path, seed, lr, l2_reg, pp_proportion, n_epochs, run_method,
                         verbose=False, topks=(50, 200)):
    device = torch.device('cuda')
    config = get_gowalla_config(device)
    dataset_config, model_config, trainer_config = config[0]
    trainer_config['max_patience'] = trainer_config['n_epochs']

    full_dataset = get_dataset(dataset_config)
    full_train_model = get_model(model_config, full_dataset)
    trainer = get_trainer(trainer_config, full_train_model)
    if os.path.exists('retrain/full_train_model.pth'):
        full_train_model.load('retrain/full_train_model.pth')
    else:
        trainer.train(verbose=False)
        full_train_model.save('retrain/full_train_model.pth')

    dataset_config['path'] = dataset_config['path'][:-4] + 'retrain'
    sub_dataset = get_dataset(dataset_config)
    full_rec_items = trainer.get_rec_items('test', k=max(topks))[:sub_dataset.n_users, :]

    if not verbose:
        trainer_config['val_interval'] = 1000
    trainer_config['n_epochs'] = n_epochs if n_epochs is not None else trainer_config['n_epochs']
    trainer_config['lr'] = lr if lr is not None else trainer_config['lr']
    trainer_config['l2_reg'] = l2_reg if l2_reg is not None else trainer_config['l2_reg']
    if run_method != 0:
        pre_train_model = get_model(model_config, sub_dataset)
        trainer = get_trainer(trainer_config, pre_train_model)
        trainer.train(verbose=False)

    extra_eval = (eval_rec_and_surrogate, (sub_dataset.n_users, full_rec_items, topks))
    names = {0: 'full_retrain', 1: 'part_retrain', 2: 'pp_retrain'}
    writer = SummaryWriter(os.path.join(log_path, names[run_method]))
    new_model = get_model(model_config, full_dataset)
    set_seed(seed)
    if run_method == 0:
        new_trainer = get_trainer(trainer_config, new_model)
        new_trainer.train(verbose=verbose, writer=writer, extra_eval=extra_eval)
        print('Limited full Retrain!')

    if run_method == 1:
        new_trainer = get_trainer(trainer_config, new_model)
        initial_parameter(new_model, pre_train_model)
        new_trainer.train(verbose=verbose, writer=writer, extra_eval=extra_eval)
        print('Part Retrain!')

    if run_method == 2:
        trainer_config['pp_proportion'] = pp_proportion
        new_trainer = get_trainer(trainer_config, new_model)
        initial_parameter(new_model, pre_train_model)
        new_trainer.train(verbose=verbose, writer=writer, extra_eval=extra_eval)
        print('Retrain with parameter propagation!')

    writer.close()
    jaccard_sim = eval_rec_and_surrogate(new_trainer, sub_dataset.n_users, full_rec_items, topks, None, True)
    return jaccard_sim


def main():
    seed_list = [2023, 42, 0, 131, 1024]
    seed = seed_list[0]
    log_path = __file__[:-3]
    init_run(log_path, seed)

    lr = None
    l2_reg = None
    pp_proportion = None
    n_epochs = None
    run_method = None
    jaccard_sim = run_new_items_recall(log_path, seed, lr, l2_reg, pp_proportion, n_epochs, run_method)
    print('Jaccard similarity', jaccard_sim)


if __name__ == '__main__':
    main()
