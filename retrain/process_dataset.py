import torch
from dataset import get_dataset


def resize_dataset(dataset, ratio):
    n_users = int(dataset.n_users * ratio)
    dataset.n_users = n_users
    dataset.train_data = dataset.train_data[:n_users]
    dataset.val_data = dataset.val_data[:n_users]


def main():
    name = 'Gowalla'
    device = torch.device('cpu')
    dataset_config = {'name': 'ProcessedDataset', 'path': 'data/' + name + '/time',
                      'device': device}
    dataset = get_dataset(dataset_config)
    resize_dataset(dataset, 0.99)
    dataset.output_dataset('data/' +name + '/retrain')


if __name__ == '__main__':
    main()