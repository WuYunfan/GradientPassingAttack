def get_gowalla_config(device):
    dataset_config = {'name': 'ProcessedDataset', 'path': 'data/Gowalla/time',
                      'device': device}
    gowalla_config = []

    model_config = {'name': 'MF', 'embedding_size': 64, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-4, 'l2_reg': 1.e-3,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'LightGCN', 'embedding_size': 64, 'n_layers': 3, 'device': device}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'ItemKNN', 'k': 1000, 'device': device}
    trainer_config = {'name': 'BasicTrainer',  'device': device, 'n_epochs': 0,
                      'test_batch_size': 512, 'topks': [20]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'NGCF', 'embedding_size': 64, 'layer_sizes': [64, 64, 64],
                    'device': device, 'dropout': 0.1}
    trainer_config = {'name': 'BPRTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-3,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    model_config = {'name': 'MultiVAE', 'layer_sizes': [64, 32],
                    'device': device, 'dropout': 0.7}
    trainer_config = {'name': 'MLTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-4, 'kl_reg': 0.2,
                      'device': device, 'n_epochs': 1000, 'batch_size': 512, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [20]}
    gowalla_config.append((dataset_config, model_config, trainer_config))

    dataset_config = dataset_config.copy()
    dataset_config['neg_ratio'] = 4
    model_config = {'name': 'NeuMF', 'embedding_size': 64, 'device': device, 'layer_sizes': [64, 64, 64]}
    trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-3,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 64, 'topks': [20], 'mf_pretrain_epochs': 100,
                      'mlp_pretrain_epochs': 100, 'max_patience': 100}
    gowalla_config.append((dataset_config, model_config, trainer_config))
    return gowalla_config