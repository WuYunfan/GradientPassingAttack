import sys
from attacker.wrmf_sgd_attacker import WRMFSGD
from attacker.heuristic import RandomAttacker
from attacker.heuristic import BandwagonAttacker
from attacker.basic_attacker import BasicAttacker
from attacker.itemae_sgd_attacker import ItemAESGD
from attacker.erap4_attacker import ERAP4


def get_attacker(config, dataset):
    config = config.copy()
    config['dataset'] = dataset
    attacker = getattr(sys.modules['attacker'], config['name'])
    attacker = attacker(config)
    return attacker


__all__ = ['get_attacker']
