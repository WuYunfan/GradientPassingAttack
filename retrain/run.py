from dataset import get_dataset
from model import get_model
from trainer import get_trainer
import torch
from utils import init_run, generate_adj_mat, set_seed
from tensorboardX import SummaryWriter
from config import get_gowalla_config
import numpy as np
import os


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
        new_model.embedding.weight.zero_()
        new_model.embedding.weight.data[:dataset.n_users, :] = model.embedding.weight[:dataset.n_users, :]
        new_model.embedding.weight.data[-dataset.n_items:, :] = model.embedding.weight[-dataset.n_items:, :]


def main():
    seed_list = [0, 42, 2022, 131, 1024]

    log_path = __file__[:-3]
    init_run(log_path, seed_list[0])

    device = torch.device('cuda')
    config = get_gowalla_config(device)
    dataset_config, model_config, trainer_config = config[0]
    dataset_config['path'] = dataset_config['path'][:-4] + 'retrain'

    writer = SummaryWriter(os.path.join(log_path, 'pre_train'))
    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    trainer = get_trainer(trainer_config, dataset, model)
    if os.path.exists('retrain/pre_train_model.pth'):
        model.load('retrain/pre_train_model.pth')
    else:
        trainer.train(verbose=False, writer=writer)
        model.save('retrain/pre_train_model.pth')
    old_rec_items = trainer.get_rec_items('train', None)
    writer.close()

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
            new_trainer = get_trainer(trainer_config, new_dataset, new_model)
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

    for n_epochs in [10, 20, 50, 200]:
        trainer_config['n_epochs'] = n_epochs

        writer = SummaryWriter(os.path.join(log_path, 'full_retrain_' + str(n_epochs)))
        new_model = get_model(model_config, new_dataset)
        new_trainer = get_trainer(trainer_config, new_dataset, new_model)
        new_trainer.train(verbose=True, writer=writer)
        writer.close()
        print('Limited full Retrain!')
        new_trainer.retrain_eval(dataset.n_users, dataset.n_items)
        rec_items = new_trainer.get_rec_items('train', None)[:dataset.n_users, :]
        new_rec_items = get_new_rec_items(rec_items, old_rec_items)
        recall = cal_recall_set(new_rec_items, full_retrain_new_rec_items)
        print('Recall of limited full retrain: {:.3f}, n_epochs {:d}\n'.format(recall * 100, n_epochs))

        writer = SummaryWriter(os.path.join(log_path, 'part_retrain' + str(n_epochs)))
        new_model = get_model(model_config, new_dataset)
        new_trainer = get_trainer(trainer_config, new_dataset, new_model)
        initial_parameter(new_model, model)
        new_trainer.train(verbose=True, writer=writer)
        writer.close()
        print('Part Retrain!')
        new_trainer.retrain_eval(dataset.n_users, dataset.n_items)
        rec_items = new_trainer.get_rec_items('train', None)[:dataset.n_users, :]
        new_rec_items = get_new_rec_items(rec_items, old_rec_items)
        recall = cal_recall_set(new_rec_items, full_retrain_new_rec_items)
        print('Recall of part retrain: {:.3f}, n_epochs {:d}\n'.format(recall * 100, n_epochs))

        trainer_config['parameter_propagation'] = 2
        writer = SummaryWriter(os.path.join(log_path, 'pp_retrain' + str(n_epochs)))
        new_model = get_model(model_config, new_dataset)
        new_trainer = get_trainer(trainer_config, new_dataset, new_model)
        new_trainer.generate_pp_mat(dataset.n_users)
        initial_parameter(new_model, model)
        new_trainer.train(verbose=True, writer=writer)
        writer.close()
        print('Retrain with parameter propagation!')
        new_trainer.retrain_eval(dataset.n_users, dataset.n_items)
        rec_items = new_trainer.get_rec_items('train', None)[:dataset.n_users, :]
        new_rec_items = get_new_rec_items(rec_items, old_rec_items)
        recall = cal_recall_set(new_rec_items, full_retrain_new_rec_items)
        print('Recall of parameter propagation:{:.3f}, n_epochs {:d}\n'.format(recall * 100, n_epochs))


if __name__ == '__main__':
    main()