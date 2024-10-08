import numpy as np


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

    model_config = {'name': 'MultiVAE', 'layer_sizes': [64, 32],
                    'dropout': 0.8}
    trainer_config = {'name': 'MLTrainer', 'optimizer': 'Adam',
                      'lr': 0.001, 'l2_reg': 0.0001, 'kl_reg': 0.2,
                      'n_epochs': 1000, 'batch_size': 2048,
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
    return gowalla_config


def get_gowalla_attacker_config():
    gowalla_attacker_config = []

    attacker_config = {'name': 'RandomAttacker', 'n_fakes': 0, 'n_inters': 41, 'topk': 50}
    gowalla_attacker_config.append(attacker_config)

    attacker_config = {'name': 'RandomAttacker', 'n_fakes': 131, 'n_inters': 41, 'topk': 50}
    gowalla_attacker_config.append(attacker_config)

    attacker_config = {'name': 'BandwagonAttacker', 'top_rate': 0.1, 'popular_inter_rate': 0.5,
                       'n_fakes': 131, 'n_inters': 41, 'topk': 50}
    gowalla_attacker_config.append(attacker_config)

    gp_config = {'threshold_odd': 0, 'threshold_even': -np.inf, 'alpha_odd': 1., 'alpha_even': 10.}
    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': 0.01, 'l2_reg': 0.01,
                                'n_epochs': 1, 'batch_size': 2 ** 12, 'dataloader_num_workers': 6,
                                'test_batch_size': 2048, 'topks': [50], 'neg_ratio': 4, 'verbose': False,
                                'gp_config': gp_config}
    attacker_config = {'name': 'RAPURAttacker', 'top_rate': 0.1, 'n_fakes': 131, 'n_inters': 41, 'topk': 50, 'step': 1,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config}
    gowalla_attacker_config.append(attacker_config)

    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'UserBatchTrainer', 'optimizer': 'Adam', 'lr': 0.01, 'l2_reg': 0.01,
                                'n_epochs': 50, 'batch_size': 2048, 'loss_function': 'mse_loss',
                                'test_batch_size': 2048, 'topks': [50], 'verbose': False}
    attacker_config = {'name': 'PGAAttacker', 'lr': 10., 'momentum': 0.95,
                       'n_fakes': 131, 'n_inters': 41, 'topk': 50, 'adv_epochs': 30,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config}
    gowalla_attacker_config.append(attacker_config)

    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'UserBatchTrainer', 'optimizer': 'Adam', 'lr': 0.1, 'l2_reg': 0.1,
                                'n_epochs': 45, 'batch_size': 2048, 'loss_function': 'mse_loss', 'weight': 20.,
                                'test_batch_size': 2048, 'topks': [50], 'verbose': False}
    attacker_config = {'name': 'RevAdvAttacker', 'lr': 1., 'momentum': 0.95,
                       'n_fakes': 131, 'unroll_steps': 5, 'n_inters': 41, 'topk': 50, 'adv_epochs': 30,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config}
    gowalla_attacker_config.append(attacker_config)

    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': 0.01, 'l2_reg': 0.01,
                                'n_epochs': 1, 'batch_size': 2 ** 12, 'dataloader_num_workers': 6,
                                'test_batch_size': 2048, 'topks': [50], 'neg_ratio': 4, 'verbose': False}
    attacker_config = {'name': 'DPA2DLAttacker', 'n_fakes': 131, 'topk': 50,
                       'n_inters': 41, 'reg_u': 10., 'prob': 0.9, 'kappa': 1.,
                       'step': 1, 'alpha': 1.e-7, 'n_rounds': 1,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config}
    gowalla_attacker_config.append(attacker_config)

    gp_config = {'threshold_odd': 0, 'threshold_even': -np.inf, 'alpha_odd': 1., 'alpha_even': 100.}
    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': 0.01, 'l2_reg': 0.01,
                                'n_epochs': 1, 'batch_size': 2 ** 12, 'dataloader_num_workers': 6,
                                'test_batch_size': 2048, 'topks': [50], 'neg_ratio': 4, 'verbose': False,
                                'gp_config': gp_config}
    pre_trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam',
                          'lr': 0.001, 'l2_reg': 0.01,
                          'n_epochs': 1000, 'batch_size': 2 ** 12, 'dataloader_num_workers': 6,
                          'test_batch_size': 2048, 'topks': [50], 'neg_ratio': 4, 'gp_config': gp_config}
    attacker_config = {'name': 'DPA2DLAttacker', 'n_fakes': 131, 'topk': 50,
                       'n_inters': 41, 'reg_u': 10., 'prob': 0.9, 'kappa': 1.,
                       'step': 1, 'alpha': 1.e-7, 'n_rounds': 1,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config,
                       'pre_trainer_config': pre_trainer_config,
                       'pre_user_sample_p': 0.1}
    gowalla_attacker_config.append(attacker_config)

    gp_config = {'threshold_odd': 0, 'threshold_even': -np.inf, 'alpha_odd': 1., 'alpha_even': 10.}
    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': 0.01, 'l2_reg': 0.01,
                                'n_epochs': 1, 'batch_size': 2 ** 12, 'dataloader_num_workers': 6,
                                'test_batch_size': 2048, 'topks': [50], 'neg_ratio': 4, 'verbose': False,
                                'gp_config': gp_config}
    attacker_config = {'name': 'DPA2DLAttacker', 'n_fakes': 131, 'topk': 50,
                       'n_inters': 41, 'reg_u': 10., 'prob': 0.9, 'kappa': 1.,
                       'step': 1, 'alpha': 1.e-7, 'n_rounds': 1,
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

    model_config = {'name': 'MultiVAE', 'layer_sizes': [64, 32],
                    'dropout': 0.8}
    trainer_config = {'name': 'MLTrainer', 'optimizer': 'Adam',
                      'lr': 0.001, 'l2_reg': 1e-05, 'kl_reg': 0.2,
                      'n_epochs': 1000, 'batch_size': 2048,
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
    return yelp_config


def get_yelp_attacker_config():
    yelp_attacker_config = []

    attacker_config = {'name': 'RandomAttacker', 'n_fakes': 0, 'n_inters': 36, 'topk': 50}
    yelp_attacker_config.append(attacker_config)

    attacker_config = {'name': 'RandomAttacker', 'n_fakes': 355, 'n_inters': 36, 'topk': 50}
    yelp_attacker_config.append(attacker_config)

    attacker_config = {'name': 'BandwagonAttacker', 'top_rate': 0.1, 'popular_inter_rate': 0.5,
                       'n_fakes': 355, 'n_inters': 36, 'topk': 50}
    yelp_attacker_config.append(attacker_config)

    gp_config = {'threshold_odd': 0., 'threshold_even': -np.inf, 'alpha_odd': 1., 'alpha_even': 10., 'sample_p': 0.5}
    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': 0.001, 'l2_reg': 0.01,
                                'n_epochs': 1, 'batch_size': 2 ** 12, 'dataloader_num_workers': 6,
                                'test_batch_size': 2048, 'topks': [50], 'neg_ratio': 4, 'verbose': False,
                                'gp_config': gp_config}
    attacker_config = {'name': 'RAPURAttacker', 'top_rate': 0.1, 'n_fakes': 355, 'n_inters': 36, 'topk': 50, 'step': 2,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config}
    yelp_attacker_config.append(attacker_config)

    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'UserBatchTrainer', 'optimizer': 'Adam', 'lr': 0.01, 'l2_reg': 1.,
                                'n_epochs': 50, 'batch_size': 2048, 'loss_function': 'mse_loss',
                                'test_batch_size': 2048, 'topks': [50], 'verbose': False}
    attacker_config = {'name': 'PGAAttacker', 'lr': 10., 'momentum': 0.95,
                       'n_fakes': 355, 'n_inters': 36, 'topk': 50, 'adv_epochs': 30,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config}
    yelp_attacker_config.append(attacker_config)

    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': 0.01, 'l2_reg': 0.01,
                                'n_epochs': 1, 'batch_size': 2 ** 12, 'dataloader_num_workers': 6,
                                'test_batch_size': 2048, 'topks': [50], 'neg_ratio': 4, 'verbose': False}
    attacker_config = {'name': 'DPA2DLAttacker', 'n_fakes': 355, 'topk': 50,
                       'n_inters': 36, 'reg_u': 10., 'prob': 0.9, 'kappa': 1.,
                       'step': 2, 'alpha': 1.e-7, 'n_rounds': 1,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config}
    yelp_attacker_config.append(attacker_config)

    gp_config = {'threshold_odd': 0., 'threshold_even': -np.inf, 'alpha_odd': 1., 'alpha_even': 100., 'sample_p': 0.5}
    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': 0.01, 'l2_reg': 0.01,
                                'n_epochs': 1, 'batch_size': 2 ** 12, 'dataloader_num_workers': 6,
                                'test_batch_size': 2048, 'topks': [50], 'neg_ratio': 4, 'verbose': False,
                                'gp_config': gp_config}
    pre_trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam',
                          'lr': 0.001, 'l2_reg': 0.01,
                          'n_epochs': 1000, 'batch_size': 2 ** 12, 'dataloader_num_workers': 6,
                          'test_batch_size': 2048, 'topks': [50], 'neg_ratio': 4, 'gp_config': gp_config}
    attacker_config = {'name': 'DPA2DLAttacker', 'n_fakes': 355, 'topk': 50,
                       'n_inters': 36, 'reg_u': 10., 'prob': 0.9, 'kappa': 1.,
                       'step': 2, 'alpha': 1.e-7, 'n_rounds': 1,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config,
                       'pre_trainer_config': pre_trainer_config,
                       'pre_user_sample_p': 0.1}
    yelp_attacker_config.append(attacker_config)

    gp_config = {'threshold_odd': 0., 'threshold_even': -np.inf, 'alpha_odd': 1., 'alpha_even': 10., 'sample_p': 0.5}
    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': 0.01, 'l2_reg': 0.01,
                                'n_epochs': 1, 'batch_size': 2 ** 12, 'dataloader_num_workers': 6,
                                'test_batch_size': 2048, 'topks': [50], 'neg_ratio': 4, 'verbose': False,
                                'gp_config': gp_config}
    attacker_config = {'name': 'DPA2DLAttacker', 'n_fakes': 355, 'topk': 50,
                       'n_inters': 36, 'reg_u': 10., 'prob': 0.9, 'kappa': 1.,
                       'step': 2, 'alpha': 1.e-7, 'n_rounds': 1,
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
                      'n_epochs': 200, 'batch_size': 2 ** 18, 'dataloader_num_workers': 10,
                      'test_batch_size': 4096, 'topks': [50], 'max_patience': 20}
    tenrec_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MF', 'embedding_size': 64}
    trainer_config = {'name': 'APRTrainer', 'optimizer': 'Adam',
                      'lr': 0.001, 'l2_reg': 0.001,
                      'eps': 0.1, 'adv_reg': 10.0, 'ckpt_path': 'checkpoints/pretrain_mf.pth',
                      'n_epochs': 200, 'batch_size': 2 ** 18, 'dataloader_num_workers': 10,
                      'test_batch_size': 4096, 'topks': [50], 'max_patience': 20}
    tenrec_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam',
                      'lr': 0.01, 'l2_reg': 0.0001,
                      'n_epochs': 200, 'batch_size': 2 ** 18, 'dataloader_num_workers': 10,
                      'test_batch_size': 4096, 'topks': [50], 'max_patience': 20}
    tenrec_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MultiVAE', 'layer_sizes': [64, 32],
                    'dropout': 0.8}
    trainer_config = {'name': 'MLTrainer', 'optimizer': 'Adam',
                      'lr': 0.001, 'l2_reg': 1e-5, 'kl_reg': 0.2,
                      'n_epochs': 200, 'batch_size': 4096,
                      'test_batch_size': 4096, 'topks': [50], 'max_patience': 20}
    tenrec_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MF', 'embedding_size': 64}
    trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam',
                      'lr': 0.001, 'l2_reg': 0.001,
                      'n_epochs': 200, 'batch_size': 2 ** 16, 'dataloader_num_workers': 10,
                      'test_batch_size': 4096, 'topks': [50], 'neg_ratio': 4, 'max_patience': 20}
    tenrec_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3}
    trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam',
                      'lr': 0.01, 'l2_reg': 0.01,
                      'n_epochs': 200, 'batch_size': 2 ** 16, 'dataloader_num_workers': 10,
                      'test_batch_size': 4096, 'topks': [50], 'neg_ratio': 4, 'max_patience': 20}
    tenrec_config.append((dataset_config, model_config, trainer_config))
    return tenrec_config


def get_tenrec_attacker_config():
    tenrec_attacker_config = []

    attacker_config = {'name': 'RandomAttacker', 'n_fakes': 0, 'n_inters': 34, 'topk': 50}
    tenrec_attacker_config.append(attacker_config)

    attacker_config = {'name': 'RandomAttacker', 'n_fakes': 11952, 'n_inters': 34, 'topk': 50}
    tenrec_attacker_config.append(attacker_config)

    attacker_config = {'name': 'BandwagonAttacker', 'top_rate': 0.1, 'popular_inter_rate': 0.5,
                       'n_fakes': 11952, 'n_inters': 34, 'topk': 50}
    tenrec_attacker_config.append(attacker_config)

    gp_config = {'threshold_odd': 0., 'threshold_even': -np.inf, 'alpha_odd': 1., 'alpha_even': 10., 'sample_p': 0.25}
    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': 0.001, 'l2_reg': 0.001,
                                'n_epochs': 1, 'batch_size': 2 ** 16, 'dataloader_num_workers': 10,
                                'test_batch_size': 4096, 'topks': [50], 'neg_ratio': 4, 'verbose': False,
                                'gp_config': gp_config}
    attacker_config = {'name': 'RAPURAttacker', 'top_rate': 0.1, 'n_fakes': 11952, 'n_inters': 34, 'topk': 50, 'step': 100,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config}
    tenrec_attacker_config.append(attacker_config)

    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': 0.01, 'l2_reg': 0.001,
                                'n_epochs': 2, 'batch_size': 2 ** 16, 'dataloader_num_workers': 10,
                                'test_batch_size': 4096, 'topks': [50], 'neg_ratio': 4, 'verbose': False}
    attacker_config = {'name': 'DPA2DLAttacker', 'n_fakes': 11952, 'topk': 50,
                       'n_inters': 34, 'reg_u': 1.e7, 'prob': 0.99, 'kappa': 100.,
                       'step': 100, 'alpha': 1.e-8, 'n_rounds': 2,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config}
    tenrec_attacker_config.append(attacker_config)

    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': 0.01, 'l2_reg': 0.001,
                                'n_epochs': 1, 'batch_size': 2 ** 16, 'dataloader_num_workers': 10,
                                'test_batch_size': 4096, 'topks': [50], 'neg_ratio': 4, 'verbose': False}
    attacker_config = {'name': 'DPA2DLAttacker', 'n_fakes': 11952, 'topk': 50,
                       'n_inters': 34, 'reg_u': 1.e8, 'prob': 0.99, 'kappa': 1.5,
                       'step': 100, 'alpha': 1.e-8, 'n_rounds': 1,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config}
    tenrec_attacker_config.append(attacker_config)

    gp_config = {'threshold_odd': 0., 'threshold_even': -np.inf, 'alpha_odd': 1., 'alpha_even': 10., 'sample_p': 0.25}
    surrogate_model_config = {'name': 'MF', 'embedding_size': 64, 'verbose': False}
    surrogate_trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': 0.01, 'l2_reg': 0.001,
                                'n_epochs': 1, 'batch_size': 2 ** 16, 'dataloader_num_workers': 10,
                                'test_batch_size': 4096, 'topks': [50], 'neg_ratio': 4, 'verbose': False,
                                'gp_config': gp_config}
    attacker_config = {'name': 'DPA2DLAttacker', 'n_fakes': 11952, 'topk': 50,
                       'n_inters': 34, 'reg_u': 1.e8, 'prob': 0.99, 'kappa': 2.,
                       'step': 100, 'alpha': 1.e-8, 'n_rounds': 1,
                       'surrogate_model_config': surrogate_model_config,
                       'surrogate_trainer_config': surrogate_trainer_config}
    tenrec_attacker_config.append(attacker_config)
    return tenrec_attacker_config
