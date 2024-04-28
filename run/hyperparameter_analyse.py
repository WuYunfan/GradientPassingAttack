import torch
from dataset import get_dataset
from attacker import get_attacker
from tensorboardX import SummaryWriter
from utils import init_run, get_target_items, set_seed
from config import get_gowalla_config as get_config
from config import get_gowalla_attacker_config as get_attacker_config
import shutil
import numpy as np
target_items_lists = [[13163, 9306, 11375, 4780, 9990], [275, 7673, 7741, 10376, 7942],
                      [5851, 11920, 12563, 1254, 9246], [1692, 8460, 8293, 2438, 4490],
                      [12094, 12757, 3592, 4019, 2534]]
target_items_lists = np.array(target_items_lists)


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2023)
    seed_list = [2024, 42, 0, 131, 1024]

    device = torch.device('cuda')
    dataset_config = get_config(device)[0][0]

    hyperparameters = {'threshold_odd': [-np.inf, 0., np.inf], 'threshold_even': [-np.inf, 0., np.inf],
                       'alpha_odd': [0.1, 1., 10., 100., 1000.], 'alpha_even': [0.1, 1., 10., 100., 1000.]}
    for key in hyperparameters.keys():
        for value in hyperparameters[key]:
            attacker_config = get_attacker_config()[-1]
            attacker_config['surrogate_trainer_config']['gp_config']['threshold_odd'] = 0.
            attacker_config['surrogate_trainer_config']['gp_config']['alpha_odd'] = 1.
            attacker_config['surrogate_trainer_config']['gp_config']['threshold_even'] = -np.inf
            attacker_config['surrogate_trainer_config']['gp_config']['alpha_even'] = 10.
            attacker_config['surrogate_trainer_config']['gp_config'][key] = value

            recalls = []
            for i in range(5):
                set_seed(seed_list[i])
                dataset = get_dataset(dataset_config)
                target_items = target_items_lists[i]
                print('Target items of {:d}th run: {:s}'.format(i, str(target_items)))
                attacker_config['target_items'] = target_items

                attacker = get_attacker(attacker_config, dataset)
                attacker.generate_fake_users()
                _, model_config, trainer_config = get_config(device)[-2]
                recall = attacker.eval(model_config, trainer_config) * 100
                recalls.append(recall)
                shutil.rmtree('checkpoints')
            print('Hyperparameter {:s} with value {:.1f}, mean {:.3f}%, std {:.3f}%'.
                  format(key, value, np.mean(recalls), np.std(recalls, ddof=1)))


if __name__ == '__main__':
    main()
