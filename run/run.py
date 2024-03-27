import os.path
import torch
from dataset import get_dataset
from attacker import get_attacker
from tensorboardX import SummaryWriter
from utils import init_run, get_target_items, set_seed
from config import get_gowalla_config as get_config
from config import get_gowalla_attacker_config as get_attacker_config
import shutil
import numpy as np
""""
target_items_lists = [[13163, 9306, 11375, 4780, 9990], [275, 7673, 7741, 10376, 7942],
                      [5851, 11920, 12563, 1254, 9246], [1692, 8460, 8293, 2438, 4490],
                      [12094, 12757, 3592, 4019, 2534]]
                      
target_items_lists = [[13971, 24204, 10290, 24038, 9836], [19230, 5616, 557, 19986, 17702],
                      [21356, 5766, 2076, 4267, 18261], [8893, 20936, 19034, 16248, 178],
                      [21077, 10796, 4749, 19918, 5106]]     

target_items_lists = [[12684, 63675, 21879, 94301, 39464], [34308, 46874, 30667, 81047, 38239],
                      [88149, 14117, 13876, 32133, 66545], [36114, 86438, 87948, 77826, 89003],
                      [46344,  7322,  8368, 86818, 73708]]                                   
"""


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2023)
    seed_list = [2024, 42, 0, 131, 1024]

    device = torch.device('cuda')
    dataset_config = get_config(device)[0][0]
    attacker_config = get_attacker_config()[0]

    for i in range(5):
        set_seed(seed_list[i])
        dataset = get_dataset(dataset_config)
        target_items = get_target_items(dataset)
        print('Target items of {:d}th run: {:s}'.format(i, str(target_items)))
        attacker_config['target_items'] = target_items

        attacker = get_attacker(attacker_config, dataset)
        if os.path.exists(log_path + '-' + str(target_items)):
            shutil.rmtree(log_path + '-' + str(target_items))
        writer = SummaryWriter(log_path + '-' + str(target_items))
        attacker.generate_fake_users(writer=writer)
        configs = get_config(device)[:-1]
        for idx, (_, model_config, trainer_config) in enumerate(configs):
            attacker.eval(model_config, trainer_config, writer=writer)
            if idx == 0:
                configs[idx + 1][2]['ckpt_path'] = attacker.trainer.save_path
        writer.close()
        shutil.rmtree('checkpoints')


if __name__ == '__main__':
    main()
