def get_gowalla_config(device):
    dataset_config = {'name': 'ProcessedDataset', 'path': 'data/Gowalla/time',
                      'device': device}
    gowalla_config = []

    model_config = {'name': 'MF', 'embedding_size': 64}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam',
                      'lr': 0.002387878274820975, 'l2_reg': 0.011155230293498226,
                      'n_epochs': 1000, 'batch_size': 2 ** 14, 'dataloader_num_workers': 16,
                      'test_batch_size': 2048, 'topks': [20]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MF', 'embedding_size': 64}
    trainer_config = {'name': 'APRTrainer', 'optimizer': 'Adam',
                      'lr': 0.002387878274820975, 'l2_reg': 0.011155230293498226,
                      'eps': 0.09174597008562989, 'adv_reg': 2.830520632169947,
                      'n_epochs': 1000, 'batch_size': 2 ** 14, 'dataloader_num_workers': 16,
                      'test_batch_size': 2048, 'topks': [20]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam',
                      'lr': 0.0024245542821549034, 'l2_reg': 1.2605220577001407e-05,
                      'n_epochs': 1000, 'batch_size': 2 ** 14, 'dataloader_num_workers': 16,
                      'test_batch_size': 2048, 'topks': [20]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'ItemKNN', 'k': 992}
    trainer_config = {'name': 'BasicTrainer', 'n_epochs': 0,
                      'test_batch_size': 2048, 'topks': [20]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MultiVAE', 'layer_sizes': [64, 32],
                    'dropout': 0.8}
    trainer_config = {'name': 'MLTrainer', 'optimizer': 'Adam',
                      'lr': 0.0029114029303403777, 'l2_reg': 0.0005858536081166332, 'kl_reg': 0.2,
                      'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 16,
                      'test_batch_size': 2048, 'topks': [20]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'NeuMF', 'embedding_size': 64, 'layer_sizes': [64, 64, 64]}
    trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam',
                      'lr': 0.012373037729617153, 'l2_reg': 0.013288322307691246,
                      'n_epochs': 1000, 'batch_size': 2 ** 12, 'dataloader_num_workers': 16,
                      'test_batch_size': 64, 'topks': [20], 'mf_pretrain_epochs': 100,
                      'mlp_pretrain_epochs': 100, 'max_patience': 100, 'neg_ratio': 4}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MF', 'embedding_size': 64}
    trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam',
                      'lr': 0.0034028958774352377, 'l2_reg': 0.0044493794757059215,
                      'n_epochs': 1000, 'batch_size': 2 ** 12, 'dataloader_num_workers': 16,
                      'test_batch_size': 64, 'topks': [20], 'neg_ratio': 4}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3}
    trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam',
                      'lr': 0.003866939759804029, 'l2_reg': 0.0002709568074610985,
                      'n_epochs': 1000, 'batch_size': 2 ** 12, 'dataloader_num_workers': 16,
                      'test_batch_size': 64, 'topks': [20], 'neg_ratio': 4}
    gowalla_config.append((dataset_config, model_config, trainer_config))
    return gowalla_config


def get_gowalla_attacker_config(device):
    return None