import torch
from dataset import get_dataset
from model import get_model
from trainer import get_trainer
import matplotlib.pyplot as plt
from utils import set_seed
import imageio.v2 as imageio

plt.rc('font', family='Times New Roman')
plt.rcParams['pdf.fonttype'] = 42


def main():
    set_seed(2023)
    device = torch.device('cpu')
    dataset_config = {'name': 'ProcessedDataset', 'path': 'data/Demo',
                      'device': device}
    model_config = {'name': 'MF', 'embedding_size': 2}
    trainer_config = {'name': 'BCETrainer', 'optimizer': 'Adam',
                      'lr': 0.01, 'l2_reg': 0.2,
                      'n_epochs': 0, 'batch_size': 1, 'dataloader_num_workers': 2,
                      'test_batch_size': 1, 'topks': [50], 'neg_ratio': 1, 'gp_threshold': 0.9}
    old_dataset = get_dataset(dataset_config)
    old_model = get_model(model_config, old_dataset)
    old_trainer = get_trainer(trainer_config, old_model)
    old_model.train()

    frames = []
    for _ in range(100):
        old_trainer.train_one_epoch()
        fig, ax = plt.subplots(constrained_layout=True, figsize=(9, 9))
        embeddings = old_model.embedding.weight.detach().cpu().numpy()
        user_embeddings = embeddings[:old_model.n_users, :]
        item_embeddings = embeddings[old_model.n_users:, :]
        ax.scatter(user_embeddings[:, 0], user_embeddings[:, 1], color='red', marker='^', s=200)
        ax.scatter(item_embeddings[:, 0], item_embeddings[:, 1], color='blue', marker='*', s=200)
        for i in range(user_embeddings.shape[0]):
            ax.annotate(f'user_{i}', (user_embeddings[i, 0], user_embeddings[i, 1]), fontsize=18)
        for i in range(item_embeddings.shape[0]):
            ax.annotate(f'item_{i}', (item_embeddings[i, 0], item_embeddings[i, 1]), fontsize=18)
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.tick_params(labelsize=14)

        plt.savefig('frame.png', format='png', dpi=300)
        plt.close(fig)
        frame = imageio.imread("frame.png")
        frames.append(frame)
    imageio.mimsave('training_gp.gif', frames, 'GIF', duration=0.1)


if __name__ == '__main__':
    main()