import numpy as np
from attacker.basic_attacker import BasicAttacker
import scipy.sparse as sp


class RandomAttacker(BasicAttacker):
    def __init__(self, attacker_config):
        super(RandomAttacker, self).__init__(attacker_config)
        self.candidates = set(range(self.n_items)) - set(self.target_items)
        self.candidates = np.array(list(self.candidates))

    def generate_fake_users(self, verbose=True, writer=None):
        self.fake_users = np.zeros([self.n_fakes, self.n_items], dtype=np.float32)
        for f_u in range(self.n_fakes):
            filler_items = np.random.choice(self.candidates, size=self.n_inters - self.target_items.shape[0], replace=False)
            self.fake_users[f_u, filler_items] = 1.
            self.fake_users[f_u, self.target_items] = 1.


class BandwagonAttacker(BasicAttacker):
    def __init__(self, attacker_config):
        super(BandwagonAttacker, self).__init__(attacker_config)
        self.n_top_items = int(self.n_items * attacker_config['top_rate'])
        self.n_popular_inters = int((self.n_inters - self.target_items.shape[0]) * attacker_config['popular_inter_rate'])
        data_mat = sp.coo_matrix((np.ones((len(self.dataset.train_array),)), np.array(self.dataset.train_array).T),
                                 shape=(self.n_users, self.n_items), dtype=np.float32).tocsr()
        item_popularity = np.array(np.sum(data_mat, axis=0)).squeeze()
        popularity_rank = np.argsort(item_popularity)[::-1].copy()
        popular_items = popularity_rank[:self.n_top_items]
        unpopular_items = popularity_rank[self.n_top_items:]
        self.popular_candidates = np.array(list(set(popular_items) - set(self.target_items)))
        self.unpopular_candidates = np.array(list(set(unpopular_items) - set(self.target_items)))

    def generate_fake_users(self, verbose=True, writer=None):
        self.fake_users = np.zeros([self.n_fakes, self.n_items], dtype=np.float32)
        for f_u in range(self.n_fakes):
            selected_items = np.random.choice(self.popular_candidates, size=self.n_popular_inters, replace=False)
            filler_items = np.random.choice(self.unpopular_candidates,
                                            size=self.n_inters - self.n_popular_inters - self.target_items.shape[0],
                                            replace=False)
            self.fake_users[f_u, selected_items] = 1.
            self.fake_users[f_u, filler_items] = 1.
            self.fake_users[f_u, self.target_items] = 1.
