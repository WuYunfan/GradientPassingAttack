from dataset import get_dataset
from attacker.gbfug_attacker import IGCN
from trainer import get_trainer
from attacker import get_attacker
import torch
from utils import init_run, set_seed
import numpy as np
from torch.nn.init import kaiming_uniform_, calculate_gain, normal_, zeros_, ones_


def calculate_diff(rec_items_0, rec_items_1):
    diffs = []
    for user in range(rec_items_0.shape[0]):
        a = set(rec_items_0[user].tolist())
        b = set(rec_items_1[user].tolist())
        inter = a & b
        union = a | b
        diff = 1. - 1. * len(inter) / len(union)
        diffs.append(diff)
    return np.mean(diffs)


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2021)

    device = torch.device('cuda')
    dataset_config = {'name': 'ML1MDataset', 'min_inter': 10, 'path': 'data/ML1M',
                      'split_ratio': [0.8, 0.2], 'device': device}
    dataset = get_dataset(dataset_config)
    model_config = {'name': 'IGCN', 'n_layers': 3, 'dropout': 0.3, 'feature_ratio': 1., 'embedding_size': 64,
                    'dataset': dataset, 'device': device}
    model = IGCN(model_config)
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5,
                      'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20, 100], 'device': device}
    trainer = get_trainer(trainer_config, dataset, model)

    trainer.train()
    rec_items_0 = trainer.get_rec_items('val')

    attacker_config = {'name': 'RandomAttacker', 'n_fakes': 595, 'device': device, 'n_inters': 96}
    attacker = get_attacker(attacker_config, dataset)
    fake_users = attacker.generate_fake_users()
    for fake_u in range(fake_users.shape[0]):
        items = np.nonzero(fake_users[fake_u, :])[0].tolist()
        dataset.train_data.append(items)
        dataset.val_data.append([])
        dataset.train_array.extend([[fake_u + dataset.n_users, item] for item in items])
    dataset.n_users += fake_users.shape[0]
    model.n_users += fake_users.shape[0]
    model.adj = model.generate_graph(dataset)
    model.feat, _, _ = model.generate_feat(dataset, is_updating=True)
    rec_items_1 = trainer.get_rec_items('val')

    set_seed(2021)
    normal_(model.dense_layer.weight, std=0.1)
    zeros_(model.dense_layer.bias)
    trainer = get_trainer(trainer_config, dataset, model)
    trainer.train()
    rec_items_2 = trainer.get_rec_items('val')

    diff_0_1 = calculate_diff(rec_items_0, rec_items_1)
    diff_0_2 = calculate_diff(rec_items_0, rec_items_2)
    diff_1_2 = calculate_diff(rec_items_1, rec_items_2)
    print('diff 0-1: {:.5f}, iff 0-2: {:.5f}, iff 1-2: {:.5f}'.format(diff_0_1, diff_0_2, diff_1_2))


if __name__ == '__main__':
    main()
