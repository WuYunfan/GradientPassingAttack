import torch
from dataset import get_dataset
from attacker import get_attacker
from tensorboardX import SummaryWriter
from utils import init_run, get_target_items, set_seed
from config import get_gowalla_config as get_config
from config import get_gowalla_attacker_config as get_attacker_config
import shutil


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
