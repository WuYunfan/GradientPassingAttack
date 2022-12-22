from dataset import get_dataset
from model import get_model
from trainer import get_trainer
import torch
from utils import init_run
from tensorboardX import SummaryWriter
from config import get_gowalla_config


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2021)

    device = torch.device('cuda')
    config = get_gowalla_config(device)
    dataset_config, model_config, trainer_config = config[0]
    dataset_config['path'] = dataset_config['path'][:-4] + 'retrain'

    writer = SummaryWriter(log_path)
    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    trainer = get_trainer(trainer_config, dataset, model)
    trainer.train(verbose=True, writer=writer)
    writer.close()

    writer = SummaryWriter(log_path)
    dataset_config['path'] = dataset_config['path'][:-7] + 'time'
    new_dataset = get_dataset(dataset_config)
    new_model = get_model(model_config, new_dataset)
    with torch.no_grad():
        new_model.embedding.data[:dataset.n_users, :] = model.embedding[:dataset.n_users, :]
        new_model.embedding.data[new_dataset.n_users:new_dataset.n_users + dataset.n_items, :] = \
            model.embedding[dataset.n_users:, :]
    new_trainer = get_trainer(trainer_config, new_dataset, new_model)
    new_trainer.train(verbose=True, writer=writer)
    writer.close()
    print('\nRetrain 1000 epochs!')
    new_trainer.retrain_eval(dataset.n_users, dataset.n_items)

    writer = SummaryWriter(log_path)
    new_model = get_model(model_config, new_dataset)
    with torch.no_grad():
        new_model.embedding.data[:dataset.n_users, :] = model.embedding[:dataset.n_users, :]
        new_model.embedding.data[new_dataset.n_users:new_dataset.n_users + dataset.n_items, :] = \
            model.embedding[dataset.n_users:, :]
    trainer_config['n_epochs'] = 200
    new_trainer = get_trainer(trainer_config, new_dataset, new_model)
    new_trainer.train(verbose=True, writer=writer)
    writer.close()
    print('\nRetrain 200 epochs!')
    new_trainer.retrain_eval(dataset.n_users, dataset.n_items)

    writer = SummaryWriter(log_path)
    new_model = get_model(model_config, new_dataset)
    with torch.no_grad():
        new_model.embedding.data[:dataset.n_users, :] = model.embedding[:dataset.n_users, :]
        new_model.embedding.data[new_dataset.n_users:new_dataset.n_users + dataset.n_items, :] = \
            model.embedding[dataset.n_users:, :]
    trainer_config['n_epochs'] = 200
    trainer_config['parameter_propagation'] = 2
    new_trainer = get_trainer(trainer_config, new_dataset, new_model)
    new_trainer.train(verbose=True, writer=writer)
    writer.close()
    print('\nRetrain 200 epochs with parameter propagation!')
    new_trainer.retrain_eval(dataset.n_users, dataset.n_items)


if __name__ == '__main__':
    main()