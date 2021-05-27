import torch
from dataset import get_dataset
from attacker import get_attacker
from tensorboardX import SummaryWriter
from utils import init_run, get_target_items
from config import get_ml1m_config


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2021)

    device = torch.device('cuda')
    dataset_config = {'name': 'ML1MDataset', 'min_inter': 10, 'path': 'data/ML1M',
                      'split_ratio': [0.8, 0.2], 'device': device}
    attacker_config = {'name': 'BasicAttacker', 'n_fakes': 0, 'device': device,
                       'n_inters': 0, 'target_item': 0}
    dataset = get_dataset(dataset_config)
    target_items = get_target_items(dataset)
    attacker = get_attacker(attacker_config, dataset)

    configs = get_ml1m_config(device)
    writer = SummaryWriter(log_path)
    for model_config, trainer_config in configs:
        if model_config['name'] == 'NeuMF':
            dataset.negative_sample_ratio = 4
        else:
            dataset.negative_sample_ratio = 1
        if model_config['name'] == 'MF' and trainer_config['name'] == 'APRTrainer':
            trainer_config['ckpt_path'] = apr_ckpt_path
        for target_item in target_items:
            dataset.attack_data = None
            attacker.target_item = target_item
            attacker.eval(model_config, trainer_config, writer=writer, retrain=False)
        if model_config['name'] == 'MF' and trainer_config['name'] == 'BPRTrainer':
            apr_ckpt_path = attacker.trainer.save_path
        attacker.model = None
    writer.close()


if __name__ == '__main__':
    main()
