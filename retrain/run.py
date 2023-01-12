from dataset import get_dataset
from model import get_model
from trainer import get_trainer
import torch
from utils import init_run, generate_adj_mat, set_seed
from tensorboardX import SummaryWriter
from config import get_gowalla_config
import numpy as np
import os


def cal_overlap_rate_array(rec_items_a, rec_items_b):
    rates = []
    for user in range(rec_items_a.shape[0]):
        set_a = set(rec_items_a[user, :].tolist())
        set_b = set(rec_items_b[user, :].tolist())
        rate = 1. * len(set_a & set_b) / rec_items_a.shape[1]
        rates.append(rate)
    return np.mean(rates)


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


def initial_parameter(new_trainer, model):
    new_model = new_trainer.model
    dataset = model.config['dataset']
    new_dataset = new_model.config['dataset']
    with torch.no_grad():
        new_model.embedding.weight.zero_()
        new_model.embedding.weight.data[:dataset.n_users, :] = model.embedding.weight[:dataset.n_users, :]
        new_model.embedding.weight.data[new_dataset.n_users:new_dataset.n_users + dataset.n_items, :] = \
            model.embedding.weight[dataset.n_users:, :]

        adj_mat = generate_adj_mat(new_dataset, new_dataset.device)
        aggregated_embedding = adj_mat.spmm(new_model.embedding.weight, norm='both')
        new_model.embedding.weight.data[dataset.n_users:new_dataset.n_users, :] = \
            aggregated_embedding[dataset.n_users:new_dataset.n_users, :]
        new_model.embedding.weight.data[new_dataset.n_users + dataset.n_items:, :] = \
            aggregated_embedding[new_dataset.n_users + dataset.n_items:, :]


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2021)

    device = torch.device('cuda')
    config = get_gowalla_config(device)
    dataset_config, model_config, trainer_config = config[0]
    dataset_config['path'] = dataset_config['path'][:-4] + 'retrain'

    writer = SummaryWriter(os.path.join(log_path, 'pre_train'))
    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    initial_embedding = torch.clone(model.embedding.weight.data)
    trainer = get_trainer(trainer_config, dataset, model)
    trainer.train(verbose=True, writer=writer)
    old_rec_items = trainer.get_rec_items('train', None, k=100)
    writer.close()

    dataset_config['path'] = dataset_config['path'][:-7] + 'time'
    new_dataset = get_dataset(dataset_config)

    writer = SummaryWriter(os.path.join(log_path, 'full_retrain'))
    new_model = get_model(model_config, new_dataset)
    with torch.no_grad():
        new_model.embedding.weight.data[:dataset.n_users, :] = initial_embedding[:dataset.n_users, :]
        new_model.embedding.weight.data[new_dataset.n_users:new_dataset.n_users + dataset.n_items, :] = \
            initial_embedding[dataset.n_users:, :]
    new_trainer = get_trainer(trainer_config, new_dataset, new_model)
    new_trainer.train(verbose=True, writer=writer)
    writer.close()
    print('Full Retrain!')
    new_trainer.retrain_eval(dataset.n_users, dataset.n_items)
    full_rec_items = new_trainer.get_rec_items('train', None, k=100)[:dataset.n_users, :]
    overlap_rate = cal_overlap_rate_array(old_rec_items, full_rec_items)
    print('Overlap rate between full retrain and w/o retrain: {:.3f}\n'.format(overlap_rate * 100))
    full_retrain_new_rec_items = get_new_rec_items(full_rec_items, old_rec_items)

    writer = SummaryWriter(os.path.join(log_path, 'part_retrain'))
    new_model = get_model(model_config, new_dataset)
    new_trainer = get_trainer(trainer_config, new_dataset, new_model)
    initial_parameter(new_trainer, model)
    new_trainer.train(verbose=True, writer=writer)
    writer.close()
    print('Part Retrain!')
    new_trainer.retrain_eval(dataset.n_users, dataset.n_items)
    rec_items = new_trainer.get_rec_items('train', None, k=100)[:dataset.n_users, :]
    overlap_rate = cal_overlap_rate_array(rec_items, full_rec_items)
    print('Overlap rate between full and part retrain: {:.3f}\n'.format(overlap_rate * 100))
    new_rec_items = get_new_rec_items(rec_items, old_rec_items)
    recall = cal_recall_set(new_rec_items, full_retrain_new_rec_items)
    print('Recall between full and part retrain: {:.3f}\n'.format(recall * 100))

    writer = SummaryWriter(os.path.join(log_path, 'pp_retrain_1'))
    train_data = new_dataset.train_data
    additional_train_data = [[item for item in train_data[u] if item > dataset.n_items] for u in range(dataset.n_users)]
    additional_train_data += [train_data[u] for u in range(dataset.n_users, new_dataset.n_users)]
    new_dataset.train_data = additional_train_data
    new_model = get_model(model_config, new_dataset)
    trainer_config['parameter_propagation'] = 2
    new_trainer = get_trainer(trainer_config, new_dataset, new_model)
    initial_parameter(new_trainer, model)
    new_trainer.train(verbose=True, writer=writer)
    writer.close()

    writer = SummaryWriter(os.path.join(log_path, 'pp_retrain_2'))
    new_dataset.train_data = train_data
    new_trainer = get_trainer(trainer_config, new_dataset, new_model)
    new_trainer.train(verbose=True, writer=writer)
    writer.close()
    print('Retrain with parameter propagation!')
    new_trainer.retrain_eval(dataset.n_users, dataset.n_items)
    rec_items = new_trainer.get_rec_items('train', None, k=100)[:dataset.n_users, :]
    overlap_rate = cal_overlap_rate_array(rec_items, full_rec_items)
    print('Overlap rate between full retrain and parameter propagation:{:.3f}\n'.format(overlap_rate * 100))
    new_rec_items = get_new_rec_items(rec_items, old_rec_items)
    recall = cal_recall_set(new_rec_items, full_retrain_new_rec_items)
    print('Recall between full retrain and parameter propagation:{:.3f}\n'.format(recall * 100))


if __name__ == '__main__':
    main()