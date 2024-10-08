from dataset import get_dataset
from model import get_model
from trainer import get_trainer
import torch
from utils import init_run, set_seed
from tensorboardX import SummaryWriter
from config import get_gowalla_config as get_config
import numpy as np
import os
import time


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


def eval_surrogate(trainer, full_rec_items, writer, verbose):
    if not verbose:
        return
    start_time = time.time()
    rec_items = trainer.get_rec_items('val')

    metrics = {'Jaccard': {}, 'NDCG': {}}
    denominator = np.log2(np.arange(2, max(trainer.topks) + 2, dtype=np.float32))[None, :]
    for k in trainer.topks:
        metrics['Jaccard'][k] = np.mean(calculate_jaccard_similarity(rec_items[:, :k], full_rec_items[:, :k]))
        metrics['NDCG'][k] = np.mean(calculate_ndcg(rec_items[:, :k], full_rec_items[:, :k], denominator))

    jaccard = ''
    ndcg = ''
    for k in trainer.topks:
        jaccard += '{:.3f}%@{:d}, '.format(metrics['Jaccard'][k] * 100., k)
        ndcg += '{:.3f}%@{:d}, '.format(metrics['NDCG'][k] * 100., k)
    results = 'Jaccard similarity: {:s}NDCG: {:s}'.format(jaccard, ndcg)

    consumed_time = time.time() - start_time
    print(results, 'Time: {:.3f}s'.format(consumed_time))
    if writer is not None:
        trainer.record(writer, 'surrogate', metrics)
    return metrics['Jaccard'][trainer.topks[0]]


def cal_jaccard_sim_recall(log_path, seed, lr, l2_reg, gp_config,
                         n_epochs, run_method, victim_model, verbose=False):
    device = torch.device('cuda')
    config = get_config(device)
    dataset_config, model_config, trainer_config = config[victim_model]
    trainer_config['max_patience'] = trainer_config['n_epochs']

    dataset = get_dataset(dataset_config)
    full_train_model = get_model(model_config, dataset)
    full_train_trainer = get_trainer(trainer_config, full_train_model)
    if os.path.exists('retrain/full_train_model.pth'):
        full_train_model.load('retrain/full_train_model.pth')
    else:
        full_train_trainer.train(verbose=False)
        full_train_model.save('retrain/full_train_model.pth')
    full_rec_items = full_train_trainer.get_rec_items('val')
    del full_train_trainer, full_train_model

    trainer_config['n_epochs'] = n_epochs
    trainer_config['lr'] = lr
    trainer_config['l2_reg'] = l2_reg

    extra_eval = (eval_surrogate, full_rec_items)
    names = {0: 'full_retrain', 1: 'full_retrain_wh_gp'}
    writer = SummaryWriter(os.path.join(log_path, names[run_method]))
    if gp_config is not None:
        assert run_method == 1
        trainer_config['gp_config'] = gp_config

    set_seed(seed)
    new_model = get_model(model_config, dataset)
    new_trainer = get_trainer(trainer_config, new_model)
    new_trainer.train(verbose=verbose, writer=writer, extra_eval=extra_eval)
    writer.close()
    print('--------------------------------Finish Training!--------------------------------')

    results, _ = new_trainer.eval('val')
    print(results)
    jaccard_sim = eval_surrogate(new_trainer, full_rec_items, None, True)
    return jaccard_sim


def main():
    seed_list = [2023, 42, 0, 131, 1024]
    seed = seed_list[0]
    log_path = __file__[:-3]
    init_run(log_path, seed)

    lr = None
    l2_reg = None
    gp_config = None
    n_epochs = None
    run_method = None
    victim_model = None
    jaccard_sim = cal_jaccard_sim_recall(log_path, seed, lr, l2_reg, gp_config, n_epochs, run_method, victim_model)
    print('Jaccard similarity', jaccard_sim)


if __name__ == '__main__':
    main()
