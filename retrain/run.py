from dataset import get_dataset
from model import get_model
from trainer import get_trainer
import torch
from utils import init_run, set_seed, AverageMeter
from tensorboardX import SummaryWriter
from config import get_gowalla_config
import numpy as np
import torch.nn.functional as F
import os
from tensorboard.backend.event_processing import event_accumulator


def eval_rec_on_new_users(trainer, n_old_users, writer, verbose):
    val_data = trainer.dataset.val_data.copy()
    for user in range(n_old_users):
        trainer.dataset.val_data[user] = []
    results, metrics = trainer.eval('val')
    if verbose:
        print('New users and all items result. {:s}'.format(results))
    trainer.dataset.val_data = val_data
    trainer.record(writer, 'new_user', metrics)


def initial_parameter(new_model, pre_train_model):
    dataset = pre_train_model.dataset
    with torch.no_grad():
        new_model.embedding.weight.data[:dataset.n_users, :] = pre_train_model.embedding.weight[:dataset.n_users, :]
        new_model.embedding.weight.data[-dataset.n_items:, :] = pre_train_model.embedding.weight[-dataset.n_items:, :]


def eval_rec_and_surrogate(trainer, n_old_users, full_train_model, writer, verbose):
    eval_rec_on_new_users(trainer, n_old_users, writer, verbose)
    kls = AverageMeter()
    with torch.no_grad():
        for users in trainer.test_user_loader:
            users = users[0]
            scores_input = trainer.model.predict(users)
            scores_target = full_train_model.predict(users)
            kl = F.kl_div(F.log_softmax(scores_input, dim=1), F.softmax(scores_target, dim=1), reduction='batchmean')
            kls.update(kl.item(), users.shape[0])
    if verbose:
        print('KL divergence of surrogate model: {:.3f}'.format(kls.avg))
    writer.add_scalar('{:s}_{:s}/kl_divergence'.format(trainer.model.name, trainer.name), kls.avg, trainer.epoch)
    return kls.avg


def run_new_items_recall(pp_step, pp_alpha, bernoulli_p, log_path, seed,
                         trial=None, run_base_line=False, n_epochs=100):
    device = torch.device('cuda')
    config = get_gowalla_config(device)
    dataset_config, model_config, trainer_config = config[0]
    dataset_config['path'] = dataset_config['path'][:-4] + 'retrain'
    trainer_config['max_patience'] = 1000

    sub_dataset = get_dataset(dataset_config)
    pre_train_model = get_model(model_config, sub_dataset)
    if os.path.exists('retrain/pre_train_model.pth'):
        pre_train_model.load('retrain/pre_train_model.pth')
    else:
        trainer = get_trainer(trainer_config, pre_train_model)
        trainer.train(verbose=False)
        pre_train_model.save('retrain/pre_train_model.pth')

    dataset_config['path'] = dataset_config['path'][:-7] + 'time'
    full_dataset = get_dataset(dataset_config)
    full_train_model = get_model(model_config, full_dataset)
    if os.path.exists('retrain/full_train_model.npy'):
        full_train_model.load('retrain/full_train_model.pth')
    else:
        trainer = get_trainer(trainer_config, full_train_model)
        trainer.train(verbose=False)
        full_train_model.save('retrain/full_train_model.pth')

    trainer_config['n_epochs'] = n_epochs
    if run_base_line:
        writer = SummaryWriter(os.path.join(log_path, 'full_retrain'))
        set_seed(seed)
        new_model = get_model(model_config, full_dataset)
        new_trainer = get_trainer(trainer_config, new_model)
        extra_eval = (eval_rec_and_surrogate, (sub_dataset.n_users, full_train_model))
        new_trainer.train(verbose=False, writer=writer, extra_eval=extra_eval, trial=trial)
        writer.close()
        print('Limited full Retrain!')

        writer = SummaryWriter(os.path.join(log_path, 'part_retrain'))
        set_seed(seed)
        new_model = get_model(model_config, full_dataset)
        new_trainer = get_trainer(trainer_config, new_model)
        initial_parameter(new_model, pre_train_model)
        extra_eval = (eval_rec_and_surrogate, (sub_dataset.n_users, full_train_model))
        new_trainer.train(verbose=False, writer=writer, extra_eval=extra_eval, trial=trial)
        writer.close()
        print('Part Retrain!')

    trainer_config['pp_step'] = pp_step
    trainer_config['pp_alpha'] = pp_alpha
    writer = SummaryWriter(os.path.join(log_path, 'pp_retrain'))
    set_seed(seed)
    new_model = get_model(model_config, full_dataset)
    init_embedding = torch.clone(new_model.embedding.weight.detach())
    new_trainer = get_trainer(trainer_config, new_model)
    initial_parameter(new_model, pre_train_model)
    with torch.no_grad():
        prob = torch.full(new_model.embedding.weight.shape, bernoulli_p, device=new_model.device)
        mask = torch.bernoulli(prob)
        new_model.embedding.weight.data = new_model.embedding.weight * mask + init_embedding * (1 - mask)
    extra_eval = (eval_rec_and_surrogate, (sub_dataset.n_users, full_train_model))
    new_trainer.train(verbose=False, writer=writer, extra_eval=extra_eval, trial=trial)
    writer.close()
    print('Retrain with parameter propagation!')

    ea = event_accumulator.EventAccumulator(os.path.join(log_path, 'pp_retrain'))
    ea.Reload()
    kl_divergences = ea.Scalars('{:s}_{:s}/kl_divergence'.format(new_trainer.model.name, new_trainer.name))
    kl_divergences = [x.value for x in kl_divergences]
    return kl_divergences[-1]


def main():
    seed_list = [2023, 42, 0, 131, 1024]
    seed = seed_list[0]
    log_path = __file__[:-3]
    init_run(log_path, seed)

    pp_step = 2
    pp_alpha = 0.1
    bernoulli_p = 0.1
    kl_divergence = run_new_items_recall(pp_step, pp_alpha, bernoulli_p, log_path, seed, run_base_line=True)
    print('KL divergence', kl_divergence)


if __name__ == '__main__':
    main()
