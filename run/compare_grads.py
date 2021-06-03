from dataset import get_dataset
from attacker import get_attacker
from utils import init_run
import torch
from config import get_ml1m_attacker_config


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2021)

    device = torch.device('cuda')
    dataset_config = {'name': 'SyntheticDataset', 'n_users': 1000, 'n_items': 300, 'binary_threshold': 10.,
                      'split_ratio': [0.8, 0.2], 'device': device}
    dataset = get_dataset(dataset_config)
    attacker_config = get_ml1m_attacker_config(device)[3]
    attacker_config['target_item'] = 0
    attacker_config['train_epochs'] = 10
    attacker = get_attacker(attacker_config, dataset)
    attacker.train_igcn_model(True, None)

    weight, bias = attacker.igcn_model.dense_layer.weight, attacker.igcn_model.dense_layer.bias
    for _ in range(10):
        attacker.igcn_model.dense_layer.weight, attacker.igcn_model.dense_layer.bias = weight, bias
        attacker.fake_indices, attacker.fake_tensor = attacker.init_fake_data()
        p_grads, grads = attacker.get_two_gradients()
        print(p_grads, grads)
        cos = (p_grads * grads).sum()
        cos = cos / (torch.norm(p_grads.flatten(), p=2, dim=0) * torch.norm(grads.flatten(), p=2, dim=0))
        print('Cos: ', cos.item())


if __name__ == '__main__':
    main()