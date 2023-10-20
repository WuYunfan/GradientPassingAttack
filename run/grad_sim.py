import torch
from utils import init_run
from config import get_gowalla_config as get_config
from dataset import get_dataset
from model import get_model
from trainer import get_trainer
import numpy as np


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2023)

    device = torch.device('cuda')
    dataset_config, model_config, trainer_config = get_config(device)[0]
    trainer_config['name'] = 'BPRTrainerRecord'

    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    trainer = get_trainer(trainer_config, model)
    trainer.train(verbose=True)
    np.save('grad_sim.npy', np.array(trainer.records))


if __name__ == '__main__':
    main()