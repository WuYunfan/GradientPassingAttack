def get_gowalla_config(device):
    dataset_config = {'name': 'ProcessedDataset', 'path': 'data/Gowalla/time',
                      'device': device}
    gowalla_config = []

    model_config = {'name': 'MF', 'embedding_size': 64}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam',
                      'lr': 0.002387878274820975, 'l2_reg': 0.011155230293498226,
                      'n_epochs': 1000, 'batch_size': 2 ** 14, 'dataloader_num_workers': 16,
                      'test_batch_size': 2048, 'topks': [50]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MF', 'embedding_size': 64}
    trainer_config = {'name': 'APRTrainer', 'optimizer': 'Adam',
                      'lr': 0.002387878274820975, 'l2_reg': 0.011155230293498226,
                      'eps': 0.09174597008562989, 'adv_reg': 2.830520632169947,
                      'n_epochs': 1000, 'batch_size': 2 ** 14, 'dataloader_num_workers': 16,
                      'test_batch_size': 2048, 'topks': [50]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam',
                      'lr': 0.0024245542821549034, 'l2_reg': 1.2605220577001407e-05,
                      'n_epochs': 1000, 'batch_size': 2 ** 14, 'dataloader_num_workers': 16,
                      'test_batch_size': 2048, 'topks': [50]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'ItemKNN', 'k': 992}
    trainer_config = {'name': 'BasicTrainer', 'n_epochs': 0,
                      'test_batch_size': 2048, 'topks': [50]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MultiVAE', 'layer_sizes': [64, 32],
                    'dropout': 0.8}
    trainer_config = {'name': 'MLTrainer', 'optimizer': 'Adam',
                      'lr': 0.0029114029303403777, 'l2_reg': 0.0005858536081166332, 'kl_reg': 0.2,
                      'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 16,
                      'test_batch_size': 2048, 'topks': [50]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'NeuMF', 'embedding_size': 64, 'layer_sizes': [64, 64, 64]}
    trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam',
                      'lr': 0.012373037729617153, 'l2_reg': 0.013288322307691246,
                      'n_epochs': 1000, 'batch_size': 2 ** 12, 'dataloader_num_workers': 16,
                      'test_batch_size': 64, 'topks': [50], 'mf_pretrain_epochs': 100,
                      'mlp_pretrain_epochs': 100, 'max_patience': 100, 'neg_ratio': 4}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MF', 'embedding_size': 64}
    trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam',
                      'lr': 0.0034028958774352377, 'l2_reg': 0.0044493794757059215,
                      'n_epochs': 1000, 'batch_size': 2 ** 12, 'dataloader_num_workers': 16,
                      'test_batch_size': 64, 'topks': [50], 'neg_ratio': 4}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3}
    trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam',
                      'lr': 0.003866939759804029, 'l2_reg': 0.0002709568074610985,
                      'n_epochs': 1000, 'batch_size': 2 ** 12, 'dataloader_num_workers': 16,
                      'test_batch_size': 64, 'topks': [50], 'neg_ratio': 4}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MF', 'embedding_size': 64}
    trainer_config = {'name': 'MSETrainer', 'optimizer': 'Adam',
                      'lr': 0.020052064501354657, 'l2_reg': 1.0452068949751401e-05,
                      'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 16, 'weight': 20.,
                      'test_batch_size': 2048, 'topks': [50]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3}
    trainer_config = {'name': 'MSETrainer', 'optimizer': 'Adam',
                      'lr': 0.05097082287366014, 'l2_reg': 1.1536540558141095e-05,
                      'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 16, 'weight': 20.,
                      'test_batch_size': 2048, 'topks': [50]}
    gowalla_config.append((dataset_config, model_config, trainer_config))
    return gowalla_config


def get_yelp_config(device):
    dataset_config = {'name': 'ProcessedDataset', 'path': 'data/Yelp/time',
                      'device': device}
    yelp_config = []

    model_config = {'name': 'MF', 'embedding_size': 64}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam',
                      'lr': 0.013318639991483658, 'l2_reg': 0.002375395961651879,
                      'n_epochs': 1000, 'batch_size': 2 ** 14, 'dataloader_num_workers': 16,
                      'test_batch_size': 2048, 'topks': [50]}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MF', 'embedding_size': 64}
    trainer_config = {'name': 'APRTrainer', 'optimizer': 'Adam',
                      'lr': 0.013318639991483658, 'l2_reg': 0.002375395961651879,
                      'eps': 1.0990263391828763, 'adv_reg': 0.012768267623407449,
                      'n_epochs': 1000, 'batch_size': 2 ** 14, 'dataloader_num_workers': 16,
                      'test_batch_size': 2048, 'topks': [50]}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam',
                      'lr': 0.0017579703298454693, 'l2_reg': 0.00013774376304248568,
                      'n_epochs': 1000, 'batch_size': 2 ** 14, 'dataloader_num_workers': 16,
                      'test_batch_size': 2048, 'topks': [50]}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'ItemKNN', 'k': 978}
    trainer_config = {'name': 'BasicTrainer', 'n_epochs': 0,
                      'test_batch_size': 2048, 'topks': [50]}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MultiVAE', 'layer_sizes': [64, 32],
                    'dropout': 0.8}
    trainer_config = {'name': 'MLTrainer', 'optimizer': 'Adam',
                      'lr': 0.008888739586820031, 'l2_reg': 0.00040210062922974547, 'kl_reg': 0.2,
                      'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 16,
                      'test_batch_size': 2048, 'topks': [50]}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'NeuMF', 'embedding_size': 64, 'layer_sizes': [64, 64, 64]}
    trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam',
                      'lr': 0.003143057859093012, 'l2_reg': 0.025892068648844482,
                      'n_epochs': 1000, 'batch_size': 2 ** 12, 'dataloader_num_workers': 16,
                      'test_batch_size': 64, 'topks': [50], 'mf_pretrain_epochs': 100,
                      'mlp_pretrain_epochs': 100, 'max_patience': 100, 'neg_ratio': 4}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MF', 'embedding_size': 64}
    trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam',
                      'lr': 0.004373142178129419, 'l2_reg': 0.005797745872659756,
                      'n_epochs': 1000, 'batch_size': 2 ** 12, 'dataloader_num_workers': 16,
                      'test_batch_size': 64, 'topks': [50], 'neg_ratio': 4}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3}
    trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam',
                      'lr': 0.0012257268783622802, 'l2_reg': 0.00028410025524581024,
                      'n_epochs': 1000, 'batch_size': 2 ** 12, 'dataloader_num_workers': 16,
                      'test_batch_size': 64, 'topks': [50], 'neg_ratio': 4}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MF', 'embedding_size': 64}
    trainer_config = {'name': 'MSETrainer', 'optimizer': 'Adam',
                      'lr': 0.02518732573468356, 'l2_reg': 0.0031657868721002135,
                      'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 16, 'weight': 20.,
                      'test_batch_size': 2048, 'topks': [50]}
    yelp_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3}
    trainer_config = {'name': 'MSETrainer', 'optimizer': 'Adam',
                      'lr': 0.023431729702286325, 'l2_reg': 1.495941158635992e-05,
                      'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 16, 'weight': 20.,
                      'test_batch_size': 2048, 'topks': [50]}
    yelp_config.append((dataset_config, model_config, trainer_config))
    return yelp_config
