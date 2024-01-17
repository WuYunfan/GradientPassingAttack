from dataset import get_dataset
from model import get_model
import torch
from config import get_gowalla_config
import numpy as np
import matplotlib.pyplot as plt
import dgl
import scipy.sparse as sp
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

plt.rc('font', family='Times New Roman')
plt.rcParams['pdf.fonttype'] = 42


def main():
    """
    full_retrain = [0.072709367, 0.278802127, 0.320085526, 0.349425495, 0.382153869]
    full_retrain_wt_gp = [0.140626445, 0.303476125, 0.347834647, 0.380880475, 0.398396462]
    epochs = [1, 5, 10, 50, 100]
    pdf = PdfPages('retrain_gowalla_mf_bpr.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(5, 3))
    ax.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='Original training', color='#FF8C00')
    ax.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), 'x-', markersize=3, label='Training enhanced by GP', color='#008B8B')
    ax.set_xticks(np.arange(len(epochs)))
    ax.set_xticklabels(epochs, fontsize=17)
    ax.tick_params(axis='y', labelsize=17)
    ax.set_xlabel('Retraining Epoch', fontsize=17)
    ax.set_ylabel('Jaccard Similarity@50', fontsize=17)
    ax.set_title('MF-BPR', fontsize=17)
    ax.legend(fontsize=13, loc=4, frameon=False)
    ax.grid(True, which='major', linestyle=':', linewidth=0.8)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    full_retrain = [0.284897178, 0.314809233, 0.357316643, 0.610431731, 0.610431731]
    full_retrain_wt_gp = [0.334004045, 0.454284012, 0.426267385, 0.624437809, 0.624437809]
    epochs = [1, 5, 10, 50, 100]
    pdf = PdfPages('retrain_gowalla_lgcn_bpr.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(5, 3))
    ax.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='Original training', color='#FF8C00')
    ax.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), 'x-', markersize=3, label='Training enhanced by GP', color='#008B8B')
    ax.set_xticks(np.arange(len(epochs)))
    ax.set_xticklabels(epochs, fontsize=17)
    ax.tick_params(axis='y', labelsize=17)
    ax.set_xlabel('Retraining Epoch', fontsize=17)
    ax.set_ylabel('Jaccard Similarity@50', fontsize=17)
    ax.set_title('LightGCN-BPR', fontsize=17)
    ax.legend(fontsize=13, loc=4, frameon=False)
    ax.grid(True, which='major', linestyle=':', linewidth=0.8)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    full_retrain = [0.092861742, 0.45056802, 0.500385225, 0.583127856, 0.654973984]
    full_retrain_wt_gp = [0.393685371, 0.546095967, 0.59344399, 0.656317115, 0.719439507]
    epochs = [1, 5, 10, 50, 100]
    pdf = PdfPages('retrain_gowalla_mf_bce.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(5, 3))
    ax.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='Original training', color='#FF8C00')
    ax.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), 'x-', markersize=3, label='Training enhanced by GP', color='#008B8B')
    ax.set_xticks(np.arange(len(epochs)))
    ax.set_xticklabels(epochs, fontsize=17)
    ax.tick_params(axis='y', labelsize=17)
    ax.set_xlabel('Retraining Epoch', fontsize=17)
    ax.set_ylabel('Jaccard Similarity@50', fontsize=17)
    ax.set_title('MF-BCE', fontsize=17)
    ax.legend(fontsize=13, loc=4, frameon=False)
    ax.grid(True, which='major', linestyle=':', linewidth=0.8)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    full_retrain = [0.386606574, 0.489965975, 0.5626055, 0.617218792, 0.617218792]
    full_retrain_wt_gp = [0.431357354, 0.554632962, 0.581400335, 0.618135214, 0.61813283]
    epochs = [1, 5, 10, 50, 100]
    pdf = PdfPages('retrain_gowalla_lgcn_bce.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(5, 3))
    ax.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='Original training', color='#FF8C00')
    ax.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), 'x-', markersize=3, label='Training enhanced by GP', color='#008B8B')
    ax.set_xticks(np.arange(len(epochs)))
    ax.set_xticklabels(epochs, fontsize=17)
    ax.tick_params(axis='y', labelsize=17)
    ax.set_xlabel('Retraining Epoch', fontsize=17)
    ax.set_ylabel('Jaccard Similarity@50', fontsize=17)
    ax.set_title('LightGCN-BCE', fontsize=17)
    ax.legend(fontsize=13, loc=4, frameon=False)
    ax.grid(True, which='major', linestyle=':', linewidth=0.8)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    full_retrain = [0.135182157, 0.281044066, 0.341998488]
    full_retrain_wt_gp = [0.185392991, 0.347992837, 0.425070107]
    epochs = [1, 5, 10]
    pdf = PdfPages('retrain_tenrec_mf_bpr.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(5, 3))
    ax.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='Original training', color='#FF8C00')
    ax.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), 'x-', markersize=3, label='Training enhanced by GP', color='#008B8B')
    ax.set_xticks(np.arange(len(epochs)))
    ax.set_xticklabels(epochs, fontsize=17)
    ax.tick_params(axis='y', labelsize=17)
    ax.set_xlabel('Retraining Epoch', fontsize=17)
    ax.set_ylabel('Jaccard Similarity@50', fontsize=17)
    ax.set_title('MF-BPR', fontsize=17)
    ax.legend(fontsize=13, loc=4, frameon=False)
    ax.grid(True, which='major', linestyle=':', linewidth=0.8)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    full_retrain = [0.146110639, 0.225682482, 0.240437642]
    full_retrain_wt_gp = [0.212120742, 0.278439343, 0.2914094]
    epochs = [1, 5, 10]
    pdf = PdfPages('retrain_tenrec_lgcn_bpr.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(5, 3))
    ax.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='Original training', color='#FF8C00')
    ax.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), 'x-', markersize=3, label='Training enhanced by GP', color='#008B8B')
    ax.set_xticks(np.arange(len(epochs)))
    ax.set_xticklabels(epochs, fontsize=17)
    ax.tick_params(axis='y', labelsize=17)
    ax.set_xlabel('Retraining Epoch', fontsize=17)
    ax.set_ylabel('Jaccard Similarity@50', fontsize=17)
    ax.set_title('LightGCN-BPR', fontsize=17)
    ax.legend(fontsize=13, loc=4, frameon=False)
    ax.grid(True, which='major', linestyle=':', linewidth=0.8)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    full_retrain = [0.135210618, 0.318030983, 0.37006554]
    full_retrain_wt_gp = [0.351232648, 0.425643593, 0.425643533]
    epochs = [1, 5, 10]
    pdf = PdfPages('retrain_tenrec_mf_bce.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(5, 3))
    ax.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='Original training', color='#FF8C00')
    ax.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), 'x-', markersize=3, label='Training enhanced by GP', color='#008B8B')
    ax.set_xticks(np.arange(len(epochs)))
    ax.set_xticklabels(epochs, fontsize=17)
    ax.tick_params(axis='y', labelsize=17)
    ax.set_xlabel('Retraining Epoch', fontsize=17)
    ax.set_ylabel('Jaccard Similarity@50', fontsize=17)
    ax.set_title('MF-BCE', fontsize=17)
    ax.legend(fontsize=13, loc=4, frameon=False)
    ax.grid(True, which='major', linestyle=':', linewidth=0.8)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    full_retrain = [0.195164144, 0.296233118, 0.376415819]
    full_retrain_wt_gp = [0.36368373, 0.439675152, 0.470000029]
    epochs = [1, 5, 10]
    pdf = PdfPages('retrain_tenrec_lgcn_bce.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(5, 3))
    ax.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='Original training', color='#FF8C00')
    ax.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), 'x-', markersize=3, label='Training enhanced by GP', color='#008B8B')
    ax.set_xticks(np.arange(len(epochs)))
    ax.set_xticklabels(epochs, fontsize=17)
    ax.tick_params(axis='y', labelsize=17)
    ax.set_xlabel('Retraining Epoch', fontsize=17)
    ax.set_ylabel('Jaccard Similarity@50', fontsize=17)
    ax.set_title('LightGCN-BCE', fontsize=17)
    ax.legend(fontsize=13, loc=4, frameon=False)
    ax.grid(True, which='major', linestyle=':', linewidth=0.8)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    """

    mean_retraining = [783.2894, 698.841, 821.3908]
    mean_all = [1027.1948, 714.0858, 860.0182]
    std_retraining = [45.38373072, 52.85959821, 5.982027474]
    std_all = [49.62931231, 52.80338773, 5.878489491]
    methods = ['PGA', 'RevAdv', 'DPA2DL']
    idx = np.arange(len(methods))
    width = 0.2
    pdf = PdfPages('attack_time_gowalla.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(7, 4))
    ax.bar(idx - width, mean_all, width, label='All consumed time', yerr=std_all, color='#FF8C00')
    ax.bar(idx, mean_retraining, width, label='Retraining time', yerr=std_retraining, color='#008B8B')
    ax.set_xticks(idx)
    ax.set_xticklabels(methods, fontsize=17)
    ax.tick_params(axis='y', labelsize=17)
    ax.set_xlabel('Attack Method', fontsize=17)
    ax.set_ylabel('Consumed Time (s)', fontsize=17)
    ax.set_title('Gowalla', fontsize=17)
    ax.legend(fontsize=15, loc=9, frameon=False)
    ax.grid(True, which='major', linestyle=':', linewidth=0.8)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    mean_retraining = [2212.7436, 0., 2546.9886]
    mean_all = [2621.9806, 0., 2691.1378]
    std_retraining = [16.12011474, 0., 36.06443697]
    std_all = [18.37570253, 0., 36.12893548]
    methods = ['PGA', 'RevAdv', 'DPA2DL']
    idx = np.arange(len(methods))
    width = 0.2
    pdf = PdfPages('attack_time_yelp.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(7, 4))
    ax.bar(idx - width, mean_all, width, label='All consumed time', yerr=std_all, color='#FF8C00')
    ax.bar(idx, mean_retraining, width, label='Retraining time', yerr=std_retraining, color='#008B8B')
    ax.set_xticks(idx)
    ax.set_xticklabels(methods, fontsize=17)
    ax.tick_params(axis='y', labelsize=17)
    ax.set_xlabel('Attack Method', fontsize=17)
    # ax.set_ylabel('Consumed Time (s)', fontsize=17)
    ax.set_title('Yelp', fontsize=17)
    ax.legend(fontsize=15, loc=9, frameon=False)
    ax.grid(True, which='major', linestyle=':', linewidth=0.8)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    grad_sims = np.load('grad_sim_gowalla.npy')[:100, :]
    epochs = np.arange(grad_sims.shape[0])
    pdf = PdfPages('grad_sim_gowalla.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(7, 4))
    ax.plot(epochs, grad_sims[:, 0], '-', markersize=0, label='Interacted user-item pairs', color='#FF8C00')
    ax.fill_between(epochs, grad_sims[:, 0] - grad_sims[:, 1], grad_sims[:, 0] + grad_sims[:, 1], alpha=0.5, color='#FF8C00')
    ax.plot(epochs, grad_sims[:, 2], '-', markersize=0, label='Random user-item pairs', color='#008B8B')
    ax.fill_between(epochs, grad_sims[:, 2] - grad_sims[:, 3], grad_sims[:, 2] + grad_sims[:, 3], alpha=0.5, color='#008B8B')
    ax.set_xlabel('Training Epoch', fontsize=17)
    ax.set_ylabel('Cosine Similarity', fontsize=17)
    ax.set_ylim(-1, 1)
    ax.tick_params(axis='both', labelsize=17)
    ax.set_title('Gowalla', fontsize=17)
    ax.legend(fontsize=17, loc=0, frameon=False)
    ax.grid(True, which='major', linestyle=':', linewidth=0.8)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    grad_sims = np.load('grad_sim_yelp.npy')[:100, :]
    epochs = np.arange(grad_sims.shape[0])
    pdf = PdfPages('grad_sim_yelp.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(7, 4))
    ax.plot(epochs, grad_sims[:, 0], '-', markersize=0, label='Interacted user-item pairs', color='#FF8C00')
    ax.fill_between(epochs, grad_sims[:, 0] - grad_sims[:, 1], grad_sims[:, 0] + grad_sims[:, 1], alpha=0.5, color='#FF8C00')
    ax.plot(epochs, grad_sims[:, 2], '-', markersize=0, label='Random user-item pairs', color='#008B8B')
    ax.fill_between(epochs, grad_sims[:, 2] - grad_sims[:, 3], grad_sims[:, 2] + grad_sims[:, 3], alpha=0.5, color='#008B8B')
    ax.set_xlabel('Training Epoch', fontsize=17)
    # ax.set_ylabel('Cosine Similarity', fontsize=17)
    ax.set_ylim(-1, 1)
    ax.tick_params(axis='both', labelsize=17)
    ax.set_title('Yelp', fontsize=17)
    ax.legend(fontsize=17, loc=0, frameon=False)
    ax.grid(True, which='major', linestyle=':', linewidth=0.8)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)
    pdf.close()


if __name__ == '__main__':
    main()