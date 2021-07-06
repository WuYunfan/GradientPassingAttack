def get_ml1m_config(device):
    recommender_config = []
    model_config = {'name': 'Popular', 'device': device}
    trainer_config = {'name': 'BasicTrainer',  'n_epochs': 0, 'test_batch_size': 512,
                      'topks': [20, 100], 'device': device}
    recommender_config.append((model_config, trainer_config))

    model_config = {'name': 'MF', 'embedding_size': 64, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-2,
                      'n_epochs': 200, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20, 100], 'device': device}
    recommender_config.append((model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 2, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-2, 'l2_reg': 1.e-3,
                      'n_epochs': 200, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20, 100], 'device': device}
    recommender_config.append((model_config, trainer_config))

    model_config = {'name': 'ItemKNN', 'k': 1000, 'device': device}
    trainer_config = {'name': 'BasicTrainer',  'n_epochs': 0, 'test_batch_size': 512,
                      'topks': [20, 100], 'device': device}
    recommender_config.append((model_config, trainer_config))

    model_config = {'name': 'NeuMF', 'embedding_size': 64, 'device': device, 'layer_sizes': [64, 64, 64]}
    trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-3,
                      'device': device, 'n_epochs': 400, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 64, 'topks': [20, 100], 'mf_pretrain_epochs': 100,
                      'mlp_pretrain_epochs': 100, 'max_patience': 100}
    recommender_config.append((model_config, trainer_config))

    model_config = {'name': 'MF', 'embedding_size': 64, 'device': device}
    trainer_config = {'name': 'APRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-2,
                      'adv_reg': 1., 'eps': 0.1,
                      'device': device, 'n_epochs': 200, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20, 100]}
    recommender_config.append((model_config, trainer_config))

    model_config = {'name': 'MultiVAE', 'layer_sizes': [64, 32],
                    'device': device, 'dropout': 0.3}
    trainer_config = {'name': 'MLTrainer', 'optimizer': 'Adam', 'lr': 0.01, 'l2_reg': 0., 'kl_reg': 0.2,
                      'device': device, 'n_epochs': 200, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20, 100]}
    recommender_config.append((model_config, trainer_config))
    return recommender_config


def get_reduced_config(device):
    recommender_config = []
    model_config = {'name': 'Popular', 'device': device}
    trainer_config = {'name': 'BasicTrainer',  'n_epochs': 0, 'test_batch_size': 512,
                      'topks': [20, 100], 'device': device}
    recommender_config.append((model_config, trainer_config))

    model_config = {'name': 'MF', 'embedding_size': 64, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-2,
                      'n_epochs': 200, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20, 100], 'device': device}
    recommender_config.append((model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 2, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-2, 'l2_reg': 1.e-3,
                      'n_epochs': 200, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20, 100], 'device': device}
    recommender_config.append((model_config, trainer_config))

    model_config = {'name': 'ItemKNN', 'k': 50, 'device': device}
    trainer_config = {'name': 'BasicTrainer',  'n_epochs': 0, 'test_batch_size': 512,
                      'topks': [20, 100], 'device': device}
    recommender_config.append((model_config, trainer_config))

    model_config = {'name': 'NeuMF', 'embedding_size': 64, 'device': device, 'layer_sizes': [64, 64, 64]}
    trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': 1.e-2, 'l2_reg': 1.e-2,
                      'device': device, 'n_epochs': 400, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 64, 'topks': [20, 100], 'mf_pretrain_epochs': 100,
                      'mlp_pretrain_epochs': 100, 'max_patience': 100}
    recommender_config.append((model_config, trainer_config))

    model_config = {'name': 'MF', 'embedding_size': 64, 'device': device}
    trainer_config = {'name': 'APRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-2,
                      'adv_reg': 0.1, 'eps': 1.,
                      'device': device, 'n_epochs': 200, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20, 100]}
    recommender_config.append((model_config, trainer_config))

    model_config = {'name': 'MultiVAE', 'layer_sizes': [64, 32],
                    'device': device, 'dropout': 0.5}
    trainer_config = {'name': 'MLTrainer', 'optimizer': 'Adam', 'lr': 0.01, 'l2_reg': 1.e-4, 'kl_reg': 0.2,
                      'device': device, 'n_epochs': 200, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20, 100]}
    recommender_config.append((model_config, trainer_config))
    return recommender_config


def get_ml1m_attacker_config(device):
    attacker_configs = []
    attacker_config = {'name': 'RandomAttacker', 'n_fakes': 59, 'device': device,
                       'n_inters': 96}
    attacker_configs.append(attacker_config)

    attacker_config = {'name': 'BandwagonAttacker', 'n_fakes': 59, 'device': device,
                       'n_inters': 96, 'top_rate': 0.1, 'popular_inter_rate': 0.5}
    attacker_configs.append(attacker_config)

    surrogate_config = {'embedding_size': 64, 'lr': 0.1, 'l2_reg': 1.e-4}
    attacker_config = {'name': 'WRMFSGD', 'lr': 1., 'momentum': 0.95, 'batch_size': 2048,
                       'device': device, 'n_fakes': 59, 'unroll_steps': 5, 'train_epochs': 50,
                       'n_inters': 96, 'topk': 20, 'test_batch_size': 512,
                       'weight': 20., 'adv_epochs': 100, 'surrogate_config': surrogate_config}
    attacker_configs.append(attacker_config)

    surrogate_config = {'layer_sizes': [64, 32], 'lr': 0.01, 'l2_reg': 1.e-4}
    attacker_config = {'name': 'ItemAESGD', 'lr': 10., 'momentum': 0.95, 'batch_size': 2048,
                       'device': device, 'n_fakes': 59, 'unroll_steps': 5, 'train_epochs': 50,
                       'n_inters': 96, 'topk': 20, 'test_batch_size': 512,
                       'weight': 20., 'adv_epochs': 100, 'surrogate_config': surrogate_config}
    attacker_configs.append(attacker_config)

    surrogate_config = {'name': 'IGCN', 'n_layers': 3, 'dropout': 0.3, 'feature_ratio': 1.,
                        'embedding_size': 64, 'device': device, 'lr': 0.1, 'l2_reg': 1.e-4}
    attacker_config = {'name': 'GBFUG', 'lr': 0.1, 'momentum': 0.95, 'batch_size': 2048,
                       'device': device, 'n_fakes': 59, 'unroll_steps': 0,
                       'n_inters': 96, 'test_batch_size': 512,
                       'adv_epochs': 100, 'surrogate_config': surrogate_config, 'train_epochs': 200,
                       'topk': 20, 'weight': 20., 'dataloader_num_workers': 6}
    attacker_configs.append(attacker_config)
    return attacker_configs


def get_reduced_attacker_config(device):
    attacker_configs = []
    attacker_config = {'name': 'RandomAttacker', 'n_fakes': 43, 'device': device,
                       'n_inters': 25}
    attacker_configs.append(attacker_config)

    attacker_config = {'name': 'BandwagonAttacker', 'n_fakes': 43, 'device': device,
                       'n_inters': 25, 'top_rate': 0.1, 'popular_inter_rate': 0.5}
    attacker_configs.append(attacker_config)

    surrogate_config = {'embedding_size': 64, 'lr': 0.001, 'l2_reg': 1.e-5}
    attacker_config = {'name': 'WRMFSGD', 'lr': 10., 'momentum': 0.95, 'batch_size': 2048,
                       'device': device, 'n_fakes': 59, 'unroll_steps': 5, 'train_epochs': 50,
                       'n_inters': 96, 'topk': 20, 'test_batch_size': 512,
                       'weight': 20., 'adv_epochs': 100, 'surrogate_config': surrogate_config}
    attacker_configs.append(attacker_config)

    surrogate_config = {'layer_sizes': [64, 32], 'lr': 0.001, 'l2_reg': 1.e-5}
    attacker_config = {'name': 'ItemAESGD', 'lr': 10., 'momentum': 0.95, 'batch_size': 2048,
                       'device': device, 'n_fakes': 59, 'unroll_steps': 5, 'train_epochs': 50,
                       'n_inters': 96, 'topk': 20, 'test_batch_size': 512,
                       'weight': 20., 'adv_epochs': 100, 'surrogate_config': surrogate_config}
    attacker_configs.append(attacker_config)

    surrogate_config = {'name': 'IGCN', 'n_layers': 3, 'dropout': 0.3, 'feature_ratio': 1.,
                        'embedding_size': 64, 'device': device, 'lr': 0.1, 'l2_reg': 1.e-5}
    attacker_config = {'name': 'GBFUG', 'lr': 1., 'momentum': 0.95, 'batch_size': 2048,
                       'device': device, 'n_fakes': 59, 'unroll_steps': 0,
                       'n_inters': 96, 'test_batch_size': 512,
                       'adv_epochs': 100, 'surrogate_config': surrogate_config, 'train_epochs': 200,
                       'topk': 20, 'weight': 20., 'dataloader_num_workers': 6}
    attacker_configs.append(attacker_config)
    return attacker_configs



