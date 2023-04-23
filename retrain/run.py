from dataset import get_dataset
from model import get_model
from trainer import get_trainer
import torch
from utils import init_run, set_seed
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
    trainer.dataset.val_data = val_data
    trainer.record(writer, 'new_user', metrics)


def cal_recall_set(rec_items_a, rec_items_b):
    rates = []
    for user in range(len(rec_items_a)):
        set_a = rec_items_a[user]
        set_b = rec_items_b[user]
        if len(set_b) > 0:
            rate = 1. * len(set_a & set_b) / len(set_b)
        else:
            rate = 1.
        rates.append(rate)
    return np.mean(rates)


def get_new_rec_items(rec_items_a, rec_items_b):
    new_rec_items = []
    for user in range(rec_items_a.shape[0]):
        set_a = set(rec_items_a[user, :].tolist())
        set_b = set(rec_items_b[user, :].tolist())
        new_rec_items.append(set_a - set_b)
    return new_rec_items


def initial_parameter(new_model, model):
    dataset = model.config['dataset']
    with torch.no_grad():
        new_model.embedding.weight.data[:dataset.n_users, :] = model.embedding.weight[:dataset.n_users, :]
        new_model.embedding.weight.data[-dataset.n_items:, :] = model.embedding.weight[-dataset.n_items:, :]


def eval_rec_and_surrogate(trainer, old_rec_items, full_retrain_new_rec_items, writer, verbose):
    n_old_users = old_rec_items.shape[0]
    eval_rec_on_new_users(trainer, n_old_users, writer, verbose)
    rec_items = trainer.get_rec_items('train', None)[:n_old_users, :]
    new_rec_items = get_new_rec_items(rec_items, old_rec_items)
    recall = cal_recall_set(new_rec_items, full_retrain_new_rec_items)
    if verbose:
        print('Recall of new recommended items: {:.3f}'.format(recall * 100))
    writer.add_scalar('{:s}_{:s}/new_items_recall'.format(trainer.model.name, trainer.name), recall, trainer.epoch)
    return recall


def run_new_items_recall(pp_step, m_pp_threshold, bernoulli_p, log_path, seed, trial=None, run_base_line=False):
    device = torch.device('cuda')
    config = get_gowalla_config(device)
    dataset_config, model_config, trainer_config = config[0]
    dataset_config['path'] = dataset_config['path'][:-4] + 'retrain'
    trainer_config['max_patience'] = 1000

    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    trainer = get_trainer(trainer_config, model)
    if os.path.exists('retrain/pre_train_model.pth'):
        model.load('retrain/pre_train_model.pth')
    else:
        writer = SummaryWriter(os.path.join(log_path, 'pre_train'))
        trainer.train(verbose=False, writer=writer)
        model.save('retrain/pre_train_model.pth')
        writer.close()
    old_rec_items = trainer.get_rec_items('train', None)

    dataset_config['path'] = dataset_config['path'][:-7] + 'time'
    new_dataset = get_dataset(dataset_config)

    if os.path.exists('retrain/new_rec_items.npy'):
        full_retrain_new_rec_items = np.load('retrain/new_rec_items.npy', allow_pickle=True).tolist()
    else:
        n_full_retrain = 5
        full_retrain_new_rec_items = None
        for i in range(n_full_retrain):
            writer = SummaryWriter(os.path.join(log_path, 'full_retrain_' + str(i)))
            new_model = get_model(model_config, new_dataset)
            new_trainer = get_trainer(trainer_config, new_model)
            new_trainer.train(verbose=False, writer=writer)
            writer.close()
            print('Full Retrain ' + str(i) + ' !')
            rec_items = new_trainer.get_rec_items('train', None)[:dataset.n_users, :]
            new_rec_items = get_new_rec_items(rec_items, old_rec_items)
            if full_retrain_new_rec_items is None:
                full_retrain_new_rec_items = new_rec_items
            else:
                for user in range(len(full_retrain_new_rec_items)):
                    full_retrain_new_rec_items[user] &= new_rec_items[user]
        np.save('retrain/new_rec_items.npy', full_retrain_new_rec_items)

    trainer_config['n_epochs'] = 200
    if run_base_line:
        writer = SummaryWriter(os.path.join(log_path, 'limited_full_retrain'))
        set_seed(seed)
        new_model = get_model(model_config, new_dataset)
        new_trainer = get_trainer(trainer_config, new_model)
        extra_eval = (eval_rec_and_surrogate, (old_rec_items, full_retrain_new_rec_items, writer))
        new_trainer.train(verbose=False, writer=writer, extra_eval=extra_eval, trial=trial)
        writer.close()
        print('Limited full Retrain!')

        writer = SummaryWriter(os.path.join(log_path, 'part_retrain'))
        set_seed(seed)
        new_model = get_model(model_config, new_dataset)
        new_trainer = get_trainer(trainer_config, new_model)
        initial_parameter(new_model, model)
        extra_eval = (eval_rec_and_surrogate, (old_rec_items, full_retrain_new_rec_items, writer))
        new_trainer.train(verbose=False, writer=writer, extra_eval=extra_eval, trial=trial)
        writer.close()
        print('Part Retrain!')

    trainer_config['pp_step'] = pp_step
    trainer_config['pp_threshold'] = 1. - m_pp_threshold
    writer = SummaryWriter(os.path.join(log_path, 'pp_retrain'))
    set_seed(seed)
    new_model = get_model(model_config, new_dataset)
    init_embedding = torch.clone(new_model.embedding.weight.detach())
    new_trainer = get_trainer(trainer_config, new_model)
    initial_parameter(new_model, model)
    with torch.no_grad():
        prob = torch.full(new_model.embedding.weight.shape, bernoulli_p, device=new_model.device)
        mask = torch.bernoulli(prob)
        new_model.embedding.weight.data = new_model.embedding.weight * mask + init_embedding * (1 - mask)
    extra_eval = (eval_rec_and_surrogate, (old_rec_items, full_retrain_new_rec_items, writer))
    new_trainer.train(verbose=False, writer=writer, extra_eval=extra_eval, trial=trial)
    writer.close()
    print('Retrain with parameter propagation!')

    ea = event_accumulator.EventAccumulator(os.path.join(log_path, 'pp_retrain'))
    ea.Reload()
    new_items_recall = ea.Scalars('{:s}_{:s}/new_items_recall'.format(trainer.model.name, trainer.name))
    maximum_recall = np.max([x.value for x in new_items_recall])
    return maximum_recall


def main():
    seed_list = [0, 42, 2022, 131, 1024]
    seed = seed_list[0]
    log_path = __file__[:-3]
    init_run(log_path, seed)

    pp_step = 2
    m_pp_threshold = 0.01
    bernoulli_p = 0.1
    maximum_recall = run_new_items_recall(pp_step, m_pp_threshold, bernoulli_p, log_path, seed, run_base_line=True)
    print('Maximum recall', maximum_recall)


if __name__ == '__main__':
    main()
