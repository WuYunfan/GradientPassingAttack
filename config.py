def get_gowalla_config(device):
    dataset_config = {'name': 'ProcessedDataset', 'path': 'data/Gowalla/time',
                      'device': device}
    gowalla_config = []

    model_config = {'name': 'MF', 'embedding_size': 64}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam',
                      'lr': 0.001, 'l2_reg': 0.001,
                      'n_epochs': 1000, 'batch_size': 2 ** 14, 'dataloader_num_workers': 6,
                      'test_batch_size': 2048, 'topks': [50]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MF', 'embedding_size': 64}
    trainer_config = {'name': 'APRTrainer', 'optimizer': 'Adam',
                      'lr': 0.001, 'l2_reg': 0.001,
                      'eps': 1.0, 'adv_reg': 0.01, 'ckpt_path': 'checkpoints/pretrain_mf.pth',
                      'n_epochs': 1000, 'batch_size': 2 ** 14, 'dataloader_num_workers': 6,
                      'test_batch_size': 2048, 'topks': [50]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam',
                      'lr': 0.001, 'l2_reg': 1.e-05,
                      'n_epochs': 1000, 'batch_size': 2 ** 14, 'dataloader_num_workers': 6,
                      'test_batch_size': 2048, 'topks': [50]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'ItemKNN', 'k': 1000}
    trainer_config = {'name': 'BasicTrainer', 'n_epochs': 0,
                      'test_batch_size': 2048, 'topks': [50]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MultiVAE', 'layer_sizes': [64, 32],
                    'dropout': 0.8}
    trainer_config = {'name': 'MLTrainer', 'optimizer': 'Adam',
                      'lr': 0.001, 'l2_reg': 0.0001, 'kl_reg': 0.2,
                      'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 2,
                      'test_batch_size': 2048, 'topks': [50]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'NeuMF', 'embedding_size': 64, 'layer_sizes': [64, 64, 64]}
    trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam',
                      'lr': 0.01, 'l2_reg': 0.01,
                      'n_epochs': 1000, 'batch_size': 2 ** 12, 'dataloader_num_workers': 6,
                      'test_batch_size': 64, 'topks': [50], 'mf_pretrain_epochs': 100,
                      'mlp_pretrain_epochs': 100, 'max_patience': 100, 'neg_ratio': 4}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MF', 'embedding_size': 64}
    trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam',
                      'lr': 0.001, 'l2_reg': 0.01,
                      'n_epochs': 1000, 'batch_size': 2 ** 12, 'dataloader_num_workers': 6,
                      'test_batch_size': 2048, 'topks': [50], 'neg_ratio': 4}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3}
    trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam',
                      'lr': 0.001, 'l2_reg': 0.001,
                      'n_epochs': 1000, 'batch_size': 2 ** 12, 'dataloader_num_workers': 6,
                      'test_batch_size': 2048, 'topks': [50], 'neg_ratio': 4}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MF', 'embedding_size': 64}
    trainer_config = {'name': 'MSETrainer', 'optimizer': 'Adam',
                      'lr': 0.001, 'l2_reg': 0.0001,
                      'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 2,
                      'test_batch_size': 2048, 'topks': [50]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3}
    trainer_config = {'name': 'MSETrainer', 'optimizer': 'Adam',
                      'lr': 0.01, 'l2_reg': 1e-05,
                      'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 2,
                      'test_batch_size': 2048, 'topks': [50]}
    gowalla_config.append((dataset_config, model_config, trainer_config))
    return gowalla_config


def get_gowalla_attacker_config():
    gowalla_attacker_config = []

    attacker_config = {'name': 'RandomAttacker',  'n_fakes': 0, 'n_inters': 41, 'topk': 50}
    gowalla_attacker_config.append(attacker_config)

    attacker_config = {'name': 'RandomAttacker',  'n_fakes': 131, 'n_inters': 41, 'topk': 50}
    gowalla_attacker_config.append(attacker_config)

    attacker_config = {'name': 'BandwagonAttacker', 'top_rate': 0.1, 'popular_inter_rate': 0.5,
                       'n_fakes': 131, 'n_inters': 41, 'topk': 50}
    gowalla_attacker_config.append(attacker_config)

    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'MSETrainer', 'optimizer': 'Adam', 'lr': 0.01, 'l2_reg': 0.0,
                                'n_epochs': 47, 'batch_size': 2048, 'dataloader_num_workers': 2,
                                'test_batch_size': 2048, 'topks': [50], 'verbose': False, 'val_interval': 100}
    attacker_config = {'name': 'WRMFSGD', 'lr': 10.0, 'momentum': 0.95,
                       'n_fakes': 131, 'unroll_steps': 3, 'n_inters': 41, 'topk': 50, 'adv_epochs': 30,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config}
    gowalla_attacker_config.append(attacker_config)

    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'MSETrainer', 'optimizer': 'Adam', 'lr': 0.01, 'l2_reg': 0.0001,
                                'n_epochs': 50, 'batch_size': 2048, 'dataloader_num_workers': 2,
                                'test_batch_size': 2048, 'topks': [50], 'verbose': False, 'val_interval': 100}
    attacker_config = {'name': 'PGA', 'lr': 0.1, 'momentum': 0.95,
                       'n_fakes': 131, 'n_inters': 41, 'topk': 50, 'adv_epochs': 30,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config}
    gowalla_attacker_config.append(attacker_config)

    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': 0.001, 'l2_reg': 0.01,
                                'n_epochs': 20, 'batch_size': 2 ** 12, 'dataloader_num_workers': 6,
                                'test_batch_size': 2048, 'topks': [50], 'neg_ratio': 4, 'verbose': False,
                                'val_interval': 100}
    attacker_config = {'name': 'DPA2DL', 'n_fakes': 131, 'topk': 50,
                       'n_inters': 41, 'reg_u': 1000.0, 'prob': 0.9, 'kappa': 1.,
                       'step': 4, 'alpha': 1e-05, 'n_rounds': 5,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config}
    gowalla_attacker_config.append(attacker_config)

    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'MSETrainer', 'optimizer': 'Adam', 'lr': 0.01, 'l2_reg': 0.0,
                                'n_epochs': 47, 'batch_size': 2048, 'dataloader_num_workers': 2,
                                'test_batch_size': 2048, 'topks': [50], 'verbose': False, 'val_interval': 100,
                                'pp_threshold': 0.5}
    attacker_config = {'name': 'WRMFSGD', 'lr': 10.0, 'momentum': 0.95,
                       'n_fakes': 131, 'unroll_steps': 3, 'n_inters': 41, 'topk': 50, 'adv_epochs': 30,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config}
    gowalla_attacker_config.append(attacker_config)

    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'MSETrainer', 'optimizer': 'Adam', 'lr': 0.01, 'l2_reg': 0.0001,
                                'n_epochs': 50, 'batch_size': 2048, 'dataloader_num_workers': 2,
                                'test_batch_size': 2048, 'topks': [50], 'verbose': False,
                                'val_interval': 100, 'pp_threshold': 0.7}
    attacker_config = {'name': 'PGA', 'lr': 0.1, 'momentum': 0.95,
                       'n_fakes': 131, 'n_inters': 41, 'topk': 50, 'adv_epochs': 30,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config}
    gowalla_attacker_config.append(attacker_config)

    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': 0.001, 'l2_reg': 0.01,
                                'n_epochs': 20, 'batch_size': 2 ** 12, 'dataloader_num_workers': 6,
                                'test_batch_size': 2048, 'topks': [50], 'neg_ratio': 4, 'verbose': False,
                                'val_interval': 100, 'pp_threshold': 0.4}
    attacker_config = {'name': 'DPA2DL', 'n_fakes': 131, 'topk': 50,
                       'n_inters': 41, 'reg_u': 1000.0, 'prob': 0.9, 'kappa': 1.,
                       'step': 4, 'alpha': 1e-05, 'n_rounds': 5,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config}
    gowalla_attacker_config.append(attacker_config)
    return gowalla_attacker_config


def get_yelp_config(device):
    dataset_config = {'name': 'ProcessedDataset', 'path': 'data/Yelp/time',
                      'device': device}
    yelp_config = []

    model_config = {'name': 'MF', 'embedding_size': 64}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam',
                      'lr': 0.001, 'l2_reg': 0.001,
                      'n_epochs': 1000, 'batch_size': 2 ** 14, 'dataloader_num_workers': 6,
                      'test_batch_size': 2048, 'topks': [50]}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MF', 'embedding_size': 64}
    trainer_config = {'name': 'APRTrainer', 'optimizer': 'Adam',
                      'lr': 0.001, 'l2_reg': 0.001,
                      'eps': 1.0, 'adv_reg': 0.01, 'ckpt_path': 'checkpoints/pretrain_mf.pth',
                      'n_epochs': 1000, 'batch_size': 2 ** 14, 'dataloader_num_workers': 6,
                      'test_batch_size': 2048, 'topks': [50]}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam',
                      'lr': 0.001, 'l2_reg': 1e-05,
                      'n_epochs': 1000, 'batch_size': 2 ** 14, 'dataloader_num_workers': 6,
                      'test_batch_size': 2048, 'topks': [50]}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'ItemKNN', 'k': 100}
    trainer_config = {'name': 'BasicTrainer', 'n_epochs': 0,
                      'test_batch_size': 2048, 'topks': [50]}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MultiVAE', 'layer_sizes': [64, 32],
                    'dropout': 0.8}
    trainer_config = {'name': 'MLTrainer', 'optimizer': 'Adam',
                      'lr': 0.001, 'l2_reg': 1e-05, 'kl_reg': 0.2,
                      'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 2,
                      'test_batch_size': 2048, 'topks': [50]}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'NeuMF', 'embedding_size': 64, 'layer_sizes': [64, 64, 64]}
    trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam',
                      'lr': 0.01, 'l2_reg': 0.01,
                      'n_epochs': 1000, 'batch_size': 2 ** 12, 'dataloader_num_workers': 6,
                      'test_batch_size': 64, 'topks': [50], 'mf_pretrain_epochs': 100,
                      'mlp_pretrain_epochs': 100, 'max_patience': 100, 'neg_ratio': 4}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MF', 'embedding_size': 64}
    trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam',
                      'lr': 0.001, 'l2_reg': 0.01,
                      'n_epochs': 1000, 'batch_size': 2 ** 12, 'dataloader_num_workers': 6,
                      'test_batch_size': 2048, 'topks': [50], 'neg_ratio': 4}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3}
    trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam',
                      'lr': 0.001, 'l2_reg': 0.001,
                      'n_epochs': 1000, 'batch_size': 2 ** 12, 'dataloader_num_workers': 6,
                      'test_batch_size': 2048, 'topks': [50], 'neg_ratio': 4}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MF', 'embedding_size': 64}
    trainer_config = {'name': 'MSETrainer', 'optimizer': 'Adam',
                      'lr': 0.01, 'l2_reg': 1.e-5,
                      'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 2,
                      'test_batch_size': 2048, 'topks': [50]}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3}
    trainer_config = {'name': 'MSETrainer', 'optimizer': 'Adam',
                      'lr': 0.01, 'l2_reg': 1.e-5,
                      'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 2,
                      'test_batch_size': 2048, 'topks': [50]}
    yelp_config.append((dataset_config, model_config, trainer_config))
    return yelp_config


def get_yelp_attacker_config():
    yelp_attacker_config = []

    attacker_config = {'name': 'RandomAttacker',  'n_fakes': 0, 'n_inters': 36, 'topk': 50}
    yelp_attacker_config.append(attacker_config)

    attacker_config = {'name': 'RandomAttacker',  'n_fakes': 355, 'n_inters': 36, 'topk': 50}
    yelp_attacker_config.append(attacker_config)

    attacker_config = {'name': 'BandwagonAttacker', 'top_rate': 0.1, 'popular_inter_rate': 0.5,
                       'n_fakes': 355, 'n_inters': 36, 'topk': 50}
    yelp_attacker_config.append(attacker_config)

    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'MSETrainer', 'optimizer': 'Adam', 'lr': 0.01, 'l2_reg': 0.01,
                                'n_epochs': 50, 'batch_size': 2048, 'dataloader_num_workers': 2,
                                'test_batch_size': 2048, 'topks': [50], 'verbose': False,
                                'val_interval': 100}
    attacker_config = {'name': 'PGA', 'lr': 0.1, 'momentum': 0.95,
                       'n_fakes': 355, 'n_inters': 36, 'topk': 50, 'adv_epochs': 30,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config}
    yelp_attacker_config.append(attacker_config)

    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': 0.001, 'l2_reg': 0.01,
                                'n_epochs': 20, 'batch_size': 2 ** 12, 'dataloader_num_workers': 6,
                                'test_batch_size': 2048, 'topks': [50], 'neg_ratio': 4, 'verbose': False,
                                'val_interval': 100}
    attacker_config = {'name': 'DPA2DL', 'n_fakes': 355, 'topk': 50,
                       'n_inters': 36, 'reg_u': 10000.0, 'prob': 0.9, 'kappa': 1.,
                       'step': 20, 'alpha': 1e-05, 'n_rounds': 5,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config}
    yelp_attacker_config.append(attacker_config)

    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'MSETrainer', 'optimizer': 'Adam', 'lr': 0.01, 'l2_reg': 0.01,
                                'n_epochs': 50, 'batch_size': 2048, 'dataloader_num_workers': 2,
                                'test_batch_size': 2048, 'topks': [50], 'verbose': False,
                                'val_interval': 100, 'pp_threshold': 0.6}
    attacker_config = {'name': 'PGA', 'lr': 0.1, 'momentum': 0.95,
                       'n_fakes': 355, 'n_inters': 36, 'topk': 50, 'adv_epochs': 30,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config}
    yelp_attacker_config.append(attacker_config)

    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': 0.001, 'l2_reg': 0.01,
                                'n_epochs': 20, 'batch_size': 2 ** 12, 'dataloader_num_workers': 6,
                                'test_batch_size': 2048, 'topks': [50], 'neg_ratio': 4, 'verbose': False,
                                'val_interval': 100, 'pp_threshold': 0.}
    attacker_config = {'name': 'DPA2DL', 'n_fakes': 355, 'topk': 50,
                       'n_inters': 36, 'reg_u': 10000.0, 'prob': 0.9, 'kappa': 1.,
                       'step': 20, 'alpha': 1e-05, 'n_rounds': 5,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config}
    yelp_attacker_config.append(attacker_config)
    return yelp_attacker_config


def get_tenrec_config(device):
    dataset_config = {'name': 'ProcessedDataset', 'path': 'data/Tenrec/time',
                      'device': device}
    tenrec_config = []

    model_config = {'name': 'MF', 'embedding_size': 64}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam',
                      'lr': 0.001, 'l2_reg': 0.001,
                      'n_epochs': 200, 'batch_size': 2 ** 18, 'dataloader_num_workers': 6,
                      'test_batch_size': 4096, 'topks': [50], 'max_patience': 20}
    tenrec_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MF', 'embedding_size': 64}
    trainer_config = {'name': 'APRTrainer', 'optimizer': 'Adam',
                      'lr': 0.001, 'l2_reg': 0.001,
                      'eps': 0.1, 'adv_reg': 10.0, 'ckpt_path': 'checkpoints/pretrain_mf.pth',
                      'n_epochs': 200, 'batch_size': 2 ** 18, 'dataloader_num_workers': 6,
                      'test_batch_size': 4096, 'topks': [50], 'max_patience': 20}
    tenrec_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam',
                      'lr': 0.01, 'l2_reg': 0.0001,
                      'n_epochs': 200, 'batch_size': 2 ** 18, 'dataloader_num_workers': 6,
                      'test_batch_size': 4096, 'topks': [50], 'max_patience': 20}
    tenrec_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'ItemKNN', 'k': 1000}
    trainer_config = {'name': 'BasicTrainer', 'n_epochs': 0,
                      'test_batch_size': 4096, 'topks': [50]}
    tenrec_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MultiVAE', 'layer_sizes': [64, 32],
                    'dropout': 0.8}
    trainer_config = {'name': 'MLTrainer', 'optimizer': 'Adam',
                      'lr': 0.001, 'l2_reg': 1e-5, 'kl_reg': 0.2,
                      'n_epochs': 200, 'batch_size': 4096, 'dataloader_num_workers': 2,
                      'test_batch_size': 4096, 'topks': [50], 'max_patience': 20}
    tenrec_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MF', 'embedding_size': 64}
    trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam',
                      'lr': 0.001, 'l2_reg': 0.001,
                      'n_epochs': 200, 'batch_size': 2 ** 16, 'dataloader_num_workers': 6,
                      'test_batch_size': 4096, 'topks': [50], 'neg_ratio': 4, 'max_patience': 20}
    tenrec_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3}
    trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam',
                      'lr': 0.01, 'l2_reg': 0.01,
                      'n_epochs': 200, 'batch_size': 2 ** 16, 'dataloader_num_workers': 6,
                      'test_batch_size': 4096, 'topks': [50], 'neg_ratio': 4, 'max_patience': 20}
    tenrec_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MF', 'embedding_size': 64}
    trainer_config = {'name': 'MSETrainer', 'optimizer': 'Adam',
                      'lr': 0.01, 'l2_reg': 1.e-5,
                      'n_epochs': 200, 'batch_size': 4096, 'dataloader_num_workers': 2,
                      'test_batch_size': 4096, 'topks': [50], 'max_patience': 20}
    tenrec_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3}
    trainer_config = {'name': 'MSETrainer', 'optimizer': 'Adam',
                      'lr': 0.01, 'l2_reg': 1e-05,
                      'n_epochs': 200, 'batch_size': 4096, 'dataloader_num_workers': 2,
                      'test_batch_size': 4096, 'topks': [50], 'max_patience': 20}
    tenrec_config.append((dataset_config, model_config, trainer_config))
    return tenrec_config


def get_tenrec_attacker_config():
    tenrec_attacker_config = []

    attacker_config = {'name': 'RandomAttacker',  'n_fakes': 0, 'n_inters': 34, 'topk': 50}
    tenrec_attacker_config.append(attacker_config)

    attacker_config = {'name': 'RandomAttacker',  'n_fakes': 11952, 'n_inters': 34, 'topk': 50}
    tenrec_attacker_config.append(attacker_config)

    attacker_config = {'name': 'BandwagonAttacker', 'top_rate': 0.1, 'popular_inter_rate': 0.5,
                       'n_fakes': 11952, 'n_inters': 34, 'topk': 50}
    tenrec_attacker_config.append(attacker_config)

    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': 0.01, 'l2_reg': 0.001,
                                'n_epochs': 1, 'batch_size': 2 ** 16, 'dataloader_num_workers': 6,
                                'test_batch_size': 4096, 'topks': [50], 'neg_ratio': 4, 'verbose': False,
                                'val_interval': 100}
    attacker_config = {'name': 'DPA2DL', 'n_fakes': 11952, 'topk': 50,
                       'n_inters': 34, 'reg_u': 1000.0, 'prob': 0.9, 'kappa': 1.,
                       'step': 500, 'alpha': 1e-06, 'n_rounds': 1,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config}
    tenrec_attacker_config.append(attacker_config)

    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': 0.01, 'l2_reg': 0.001,
                                'n_epochs': 1, 'batch_size': 2 ** 16, 'dataloader_num_workers': 6,
                                'test_batch_size': 4096, 'topks': [50], 'neg_ratio': 4, 'verbose': False,
                                'val_interval': 100, 'pp_threshold': 0.9}
    attacker_config = {'name': 'DPA2DL', 'n_fakes': 11952, 'topk': 50,
                       'n_inters': 34, 'reg_u': 1000.0, 'prob': 0.9, 'kappa': 1.,
                       'step': 500, 'alpha': 1e-06, 'n_rounds': 1,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config}
    tenrec_attacker_config.append(attacker_config)
    return tenrec_attacker_config