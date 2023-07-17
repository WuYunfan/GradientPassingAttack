import torch
from dataset import get_dataset
from attacker import get_attacker
from tensorboardX import SummaryWriter
from utils import init_run, get_target_items
from config import get_gowalla_config as get_config
from config import get_gowalla_attacker_config as get_attacker_config
import shutil


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2023)

    device = torch.device('cuda')
    dataset_config = get_config(device)[0][0]
    dataset = get_dataset(dataset_config)
    target_items = get_target_items(dataset)
    print('Target items: ', target_items)
    attacker_config = get_attacker_config()[0]

    for target_item in target_items:
        attacker_config['target_item'] = target_item
        dataset = get_dataset(dataset_config)
        attacker = get_attacker(attacker_config, dataset)
        writer = SummaryWriter(log_path + str(target_items))
        attacker.generate_fake_users(writer=writer)
        configs = get_config(device)
        for idx, (_, model_config, trainer_config) in enumerate(configs):
            attacker.eval(model_config, trainer_config, writer=writer)
            if idx == 0:
                configs[idx + 1][2]['ckpt_path'] = attacker.trainer.save_path
        writer.close()
        shutil.rmtree('checkpoints')


if __name__ == '__main__':
    main()
