import sys
from attacker.revadv_attacker import RevAdv
from attacker.heuristic import RandomAttacker
from attacker.heuristic import BandwagonAttacker
from attacker.basic_attacker import BasicAttacker
from attacker.dpa2dl_attacker import DPA2DL


def get_attacker(config, dataset):
    config = config.copy()
    config['dataset'] = dataset
    config['device'] = dataset.device
    attacker = getattr(sys.modules['attacker'], config['name'])
    attacker = attacker(config)
    return attacker


__all__ = ['get_attacker']
