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
from matplotlib.ticker import MultipleLocator
plt.rc('font', family='Times New Roman')
plt.rcParams['pdf.fonttype'] = 42


def main():
    """
    full_retrain = [0.072709367, 0.278802127, 0.320085526, 0.349425495, 0.382153869]
    full_retrain_wt_gp = [0.140626445, 0.303476125, 0.347834647, 0.380880475, 0.398396462]
    full_retrain_time = [4.05, 19.135, 38.178, 199.45, 382.09]
    full_retrain_wt_gp_time = [3.837, 20.095, 41.023, 202.052, 412.404]
    epochs = [1, 5, 10, 50, 100]
    pdf = PdfPages('retrain_gowalla.pdf')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, constrained_layout=True, figsize=(10, 6))
    ax1.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='Original training', color='#FF8C00')
    ax1.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), 'x-', markersize=3, label='Training enhanced by GP', color='#008B8B')
    for i, txt in enumerate(full_retrain):
        ax1.text(i + 0.1, full_retrain[i] - 0.01, '{:.1f}s'.format(full_retrain_time[i]),
                 fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    for i, txt in enumerate(full_retrain_wt_gp_time):
        ax1.text(i + 0.1, full_retrain_wt_gp[i], '{:.1f}s'.format(full_retrain_wt_gp_time[i]),
                 fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    ax1.set_xticks(np.arange(len(epochs)))
    ax1.set_xticklabels(epochs, fontsize=21)
    ax1.yaxis.set_major_locator(MultipleLocator(0.1))
    ax1.tick_params(axis='y', labelsize=21)
    ax1.set_ylabel('JS@50', fontsize=21)
    ax1.set_title('MF-BPR', fontsize=21)
    ax1.grid(True, which='major', linestyle=':', linewidth=0.8)
    ax1.minorticks_on()
    ax1.tick_params(which='both', direction='in')
    ax1.xaxis.set_ticks_position('both')
    ax1.yaxis.set_ticks_position('both')

    full_retrain = [0.284897178, 0.314809233, 0.357316643, 0.610431731, 0.610431731]
    full_retrain_wt_gp = [0.334004045, 0.454284012, 0.426267385, 0.624437809, 0.624437809]
    full_retrain_time = [3.965, 20.566, 39.341, 198.775, 391.665]
    full_retrain_wt_gp_time = [4.138, 21.172, 42.72, 204.173, 407.382]
    epochs = [1, 5, 10, 50, 100]
    ax2.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='Original training', color='#FF8C00')
    ax2.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), 'x-', markersize=3, label='Training enhanced by GP', color='#008B8B')
    for i, txt in enumerate(full_retrain):
        ax2.text(i + 0.1, full_retrain[i] - 0.01, '{:.1f}s'.format(full_retrain_time[i]),
                 fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    for i, txt in enumerate(full_retrain_wt_gp_time):
        ax2.text(i + 0.1, full_retrain_wt_gp[i], '{:.1f}s'.format(full_retrain_wt_gp_time[i]),
                 fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    ax2.set_xticks(np.arange(len(epochs)))
    ax2.set_xticklabels(epochs, fontsize=21)
    ax2.yaxis.set_major_locator(MultipleLocator(0.1))
    ax2.tick_params(axis='y', labelsize=21)
    ax2.set_title('LightGCN-BPR', fontsize=21)
    ax2.grid(True, which='major', linestyle=':', linewidth=0.8)
    ax2.minorticks_on()
    ax2.tick_params(which='both', direction='in')
    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')

    full_retrain = [0.092861742, 0.45056802, 0.500385225, 0.583127856, 0.654973984]
    full_retrain_wt_gp = [0.393685371, 0.546095967, 0.59344399, 0.656317115, 0.719439507]
    full_retrain_time = [4.626, 21.54, 47.484, 234.485, 455.21]
    full_retrain_wt_gp_time = [5.656, 24.673, 52.94, 250.384, 531.372]
    epochs = [1, 5, 10, 50, 100]
    ax3.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='Original training', color='#FF8C00')
    ax3.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), 'x-', markersize=3, label='Training enhanced by GP', color='#008B8B')
    for i, txt in enumerate(full_retrain):
        ax3.text(i + 0.1, full_retrain[i] - 0.01, '{:.1f}s'.format(full_retrain_time[i]),
                 fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    for i, txt in enumerate(full_retrain_wt_gp_time):
        ax3.text(i + 0.1, full_retrain_wt_gp[i], '{:.1f}s'.format(full_retrain_wt_gp_time[i]),
                 fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    ax3.set_xticks(np.arange(len(epochs)))
    ax3.set_xticklabels(epochs, fontsize=21)
    ax3.yaxis.set_major_locator(MultipleLocator(0.1))
    ax3.tick_params(axis='y', labelsize=21)
    ax3.set_xlabel('Retraining Epoch', fontsize=21)
    ax3.set_ylabel('JS@50', fontsize=21)
    ax3.set_title('MF-BCE', fontsize=21)
    ax3.grid(True, which='major', linestyle=':', linewidth=0.8)
    ax3.minorticks_on()
    ax3.tick_params(which='both', direction='in')
    ax3.xaxis.set_ticks_position('both')
    ax3.yaxis.set_ticks_position('both')

    full_retrain = [0.386606574, 0.489965975, 0.5626055, 0.617218792, 0.617218792]
    full_retrain_wt_gp = [0.431357354, 0.554632962, 0.581400335, 0.618135214, 0.61813283]
    full_retrain_time = [5.397, 23.875, 49.476, 240.912, 494.612]
    full_retrain_wt_gp_time = [5.395, 27.987, 52.184, 272.318, 520.435]
    epochs = [1, 5, 10, 50, 100]
    ax4.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='Original training', color='#FF8C00')
    ax4.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), 'x-', markersize=3, label='Training enhanced by GP', color='#008B8B')
    for i, txt in enumerate(full_retrain):
        ax4.text(i + 0.1, full_retrain[i] - 0.01, '{:.1f}s'.format(full_retrain_time[i]),
                 fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    for i, txt in enumerate(full_retrain_wt_gp_time):
        ax4.text(i + 0.1, full_retrain_wt_gp[i], '{:.1f}s'.format(full_retrain_wt_gp_time[i]),
                 fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    ax4.set_xticks(np.arange(len(epochs)))
    ax4.set_xticklabels(epochs, fontsize=21)
    ax4.yaxis.set_major_locator(MultipleLocator(0.1))
    ax4.tick_params(axis='y', labelsize=21)
    ax4.set_xlabel('Retraining Epoch', fontsize=21)
    ax4.set_title('LightGCN-BCE', fontsize=21)
    ax4.grid(True, which='major', linestyle=':', linewidth=0.8)
    ax4.minorticks_on()
    ax4.tick_params(which='both', direction='in')
    ax4.xaxis.set_ticks_position('both')
    ax4.yaxis.set_ticks_position('both')
    plt.tight_layout(rect=[0, 0., 1, 0.88])
    handles, labels = [], []
    for h, l in zip(*ax1.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
    fig.legend(handles, labels, loc='upper center', ncol=len(handles), fontsize=21)
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    full_retrain = [0.135182157, 0.281044066, 0.341998488]
    full_retrain_wt_gp = [0.185392991, 0.347992837, 0.425070107]
    full_retrain_time = [231.375, 1156.915, 2152.129]
    full_retrain_wt_gp_time = [254.065, 1117.596, 2297.379]
    epochs = [1, 5, 10]
    pdf = PdfPages('retrain_tenrec.pdf')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, constrained_layout=True, figsize=(10, 6))
    ax1.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='Original training', color='#FF8C00')
    ax1.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), 'x-', markersize=3, label='Training enhanced by GP', color='#008B8B')
    for i, txt in enumerate(full_retrain):
        ax1.text(i + 0.1, full_retrain[i] - 0.01, '{:.1f}s'.format(full_retrain_time[i]),
                 fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    for i, txt in enumerate(full_retrain_wt_gp_time):
        ax1.text(i + 0.1, full_retrain_wt_gp[i], '{:.1f}s'.format(full_retrain_wt_gp_time[i]),
                 fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    ax1.set_xticks(np.arange(len(epochs)))
    ax1.set_xticklabels(epochs, fontsize=21)
    ax1.yaxis.set_major_locator(MultipleLocator(0.05))
    ax1.tick_params(axis='y', labelsize=21)
    ax1.set_ylabel('JS@50', fontsize=21)
    ax1.set_title('MF-BPR', fontsize=21)
    ax1.grid(True, which='major', linestyle=':', linewidth=0.8)
    ax1.minorticks_on()
    ax1.tick_params(which='both', direction='in')
    ax1.xaxis.set_ticks_position('both')
    ax1.yaxis.set_ticks_position('both')

    full_retrain = [0.146110639, 0.225682482, 0.240437642]
    full_retrain_wt_gp = [0.212120742, 0.278439343, 0.2914094]
    full_retrain_time = [266.022, 1338.157, 2651.559]
    full_retrain_wt_gp_time = [303.972, 1435.668, 2777.312]
    epochs = [1, 5, 10]
    ax2.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='Original training', color='#FF8C00')
    ax2.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), 'x-', markersize=3, label='Training enhanced by GP', color='#008B8B')
    for i, txt in enumerate(full_retrain):
        ax2.text(i + 0.1, full_retrain[i] - 0.01, '{:.1f}s'.format(full_retrain_time[i]),
                 fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    for i, txt in enumerate(full_retrain_wt_gp_time):
        ax2.text(i + 0.1, full_retrain_wt_gp[i], '{:.1f}s'.format(full_retrain_wt_gp_time[i]),
                 fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    ax2.set_xticks(np.arange(len(epochs)))
    ax2.set_xticklabels(epochs, fontsize=21)
    ax2.yaxis.set_major_locator(MultipleLocator(0.05))
    ax2.tick_params(axis='y', labelsize=21)
    ax2.set_title('LightGCN-BPR', fontsize=21)
    ax2.grid(True, which='major', linestyle=':', linewidth=0.8)
    ax2.minorticks_on()
    ax2.tick_params(which='both', direction='in')
    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')

    full_retrain = [0.135210618, 0.318030983, 0.37006554]
    full_retrain_wt_gp = [0.351232648, 0.425643593, 0.425643533]
    full_retrain_time = [268.459, 1188.96, 2468.592]
    full_retrain_wt_gp_time = [470.074, 2387.18, 4150.449]
    epochs = [1, 5, 10]
    ax3.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='Original training', color='#FF8C00')
    ax3.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), 'x-', markersize=3, label='Training enhanced by GP', color='#008B8B')
    for i, txt in enumerate(full_retrain):
        ax3.text(i + 0.1, full_retrain[i] - 0.01, '{:.1f}s'.format(full_retrain_time[i]),
                 fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    for i, txt in enumerate(full_retrain_wt_gp_time):
        ax3.text(i + 0.1, full_retrain_wt_gp[i], '{:.1f}s'.format(full_retrain_wt_gp_time[i]),
                 fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    ax3.set_xticks(np.arange(len(epochs)))
    ax3.set_xticklabels(epochs, fontsize=21)
    ax3.yaxis.set_major_locator(MultipleLocator(0.05))
    ax3.tick_params(axis='y', labelsize=21)
    ax3.set_xlabel('Retraining Epoch', fontsize=21)
    ax3.set_ylabel('JS@50', fontsize=21)
    ax3.set_title('MF-BCE', fontsize=21)
    ax3.grid(True, which='major', linestyle=':', linewidth=0.8)
    ax3.minorticks_on()
    ax3.tick_params(which='both', direction='in')
    ax3.xaxis.set_ticks_position('both')
    ax3.yaxis.set_ticks_position('both')

    full_retrain = [0.195164144, 0.296233118, 0.376415819]
    full_retrain_wt_gp = [0.36368373, 0.439675152, 0.470000029]
    full_retrain_time = [313.102, 1683.847, 3039.408]
    full_retrain_wt_gp_time = [664.125, 3397.645, 5864.576]
    epochs = [1, 5, 10]
    ax4.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='Original training', color='#FF8C00')
    ax4.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), 'x-', markersize=3, label='Training enhanced by GP', color='#008B8B')
    for i, txt in enumerate(full_retrain):
        ax4.text(i + 0.1, full_retrain[i] - 0.01, '{:.1f}s'.format(full_retrain_time[i]),
                 fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    for i, txt in enumerate(full_retrain_wt_gp_time):
        ax4.text(i + 0.1, full_retrain_wt_gp[i], '{:.1f}s'.format(full_retrain_wt_gp_time[i]),
                 fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    ax4.set_xticks(np.arange(len(epochs)))
    ax4.set_xticklabels(epochs, fontsize=21)
    ax4.yaxis.set_major_locator(MultipleLocator(0.05))
    ax4.tick_params(axis='y', labelsize=21)
    ax4.set_xlabel('Retraining Epoch', fontsize=21)
    ax4.set_title('LightGCN-BCE', fontsize=21)
    ax4.grid(True, which='major', linestyle=':', linewidth=0.8)
    ax4.minorticks_on()
    ax4.tick_params(which='both', direction='in')
    ax4.xaxis.set_ticks_position('both')
    ax4.yaxis.set_ticks_position('both')
    plt.tight_layout(rect=[0, 0., 1, 0.88])
    handles, labels = [], []
    for h, l in zip(*ax1.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
    fig.legend(handles, labels, loc='upper center', ncol=len(handles), fontsize=21)
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
    pdf = PdfPages('attack_time.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(9, 3.5))
    ax.bar(idx - width / 2, mean_all, width, label='All consumed time', yerr=std_all,
           color='#e2f0d9', edgecolor='black', linewidth=1, hatch='/')
    ax.bar(idx + width / 2, mean_retraining, width, label='Retraining time', yerr=std_retraining,
           color='#ddebf7', edgecolor='black', linewidth=1, hatch='.')
    ax.set_xticks(idx)
    ax.set_xticklabels(methods, fontsize=21)
    ax.tick_params(axis='y', labelsize=21)
    ax.set_ylabel('Consumed Time (s)', fontsize=21)
    ax.grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.8)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in')
    ax.legend(fontsize=18, loc='upper right', ncol=2)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)
    pdf.close()
    """
    grad_sims = np.load('grad_sim_gowalla.npy')[:100, :]
    epochs = np.arange(grad_sims.shape[0])
    pdf = PdfPages('grad_sim.pdf')
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(14, 5))
    ax1.plot(epochs, grad_sims[:, 0], '-', markersize=0, label='Interacted user-item pairs', color='#008B8B')
    ax1.fill_between(epochs, grad_sims[:, 0] - grad_sims[:, 1], grad_sims[:, 0] + grad_sims[:, 1], alpha=0.5, color='#008B8B')
    ax1.plot(epochs, grad_sims[:, 2], '-', markersize=0, label='Random user-item pairs', color='#FF8C00')
    ax1.fill_between(epochs, grad_sims[:, 2] - grad_sims[:, 3], grad_sims[:, 2] + grad_sims[:, 3], alpha=0.5, color='#FF8C00')
    ax1.set_xlabel('Training Epoch', fontsize=21)
    ax1.set_ylabel('Cosine Similarity', fontsize=21)
    ax1.set_ylim(-1, 1)
    ax1.tick_params(axis='both', labelsize=21)
    ax1.set_title('Gowalla', fontsize=21)
    ax1.grid(True, which='major', linestyle=':', linewidth=0.8)
    ax1.minorticks_on()
    ax1.tick_params(which='both', direction='in')
    ax1.xaxis.set_ticks_position('both')
    ax1.yaxis.set_ticks_position('both')

    grad_sims = np.load('grad_sim_yelp.npy')[:100, :]
    epochs = np.arange(grad_sims.shape[0])
    ax2.plot(epochs, grad_sims[:, 0], '-', markersize=0, label='Interacted user-item pairs', color='#008B8B')
    ax2.fill_between(epochs, grad_sims[:, 0] - grad_sims[:, 1], grad_sims[:, 0] + grad_sims[:, 1], alpha=0.5, color='#008B8B')
    ax2.plot(epochs, grad_sims[:, 2], '-', markersize=0, label='Random user-item pairs', color='#FF8C00')
    ax2.fill_between(epochs, grad_sims[:, 2] - grad_sims[:, 3], grad_sims[:, 2] + grad_sims[:, 3], alpha=0.5, color='#FF8C00')
    ax2.set_xlabel('Training Epoch', fontsize=21)
    ax2.set_ylim(-1, 1)
    ax2.tick_params(axis='both', labelsize=21)
    ax2.set_title('Yelp', fontsize=21)
    ax2.grid(True, which='major', linestyle=':', linewidth=0.8)
    ax2.minorticks_on()
    ax2.tick_params(which='both', direction='in')
    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')
    plt.tight_layout(rect=[0, 0., 1, 0.86])
    handles, labels = [], []
    for h, l in zip(*ax1.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
    fig.legend(handles, labels, loc='upper center', ncol=len(handles), fontsize=21)
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    threshold_odd_mean = np.array([2.037, 2.121, 1.172])
    threshold_odd_std = np.array([1.164, 1.095, 0.698])
    threshold_even_mean = np.array([2.121, 0.884, 0.388])
    threshold_even_std = np.array([1.095, 0.564, 0.492])
    base_mean = np.array([0.428, 0.428, 0.428])
    base_std = np.array([0.634447791, 0.634447791, 0.634447791])
    threshold_value = [-np.inf, 0., np.inf]
    pdf = PdfPages('hyper.pdf')
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(14, 5))
    ax1.plot(np.arange(len(threshold_value)), threshold_odd_mean, 's-', markersize=3,
             label='odd threshold $\\xi_{odd}$', color='#FF8C00')
    ax1.fill_between(np.arange(len(threshold_value)), threshold_odd_mean - threshold_odd_std,
                     threshold_odd_mean + threshold_odd_std, alpha=0.1, color='#FF8C00')
    ax1.plot(np.arange(len(threshold_value)), threshold_even_mean, 'x-', markersize=3,
             label='even threshold $\\xi_{even}$', color='#008B8B')
    ax1.fill_between(np.arange(len(threshold_value)), threshold_even_mean - threshold_even_std,
                     threshold_even_mean + threshold_even_std, alpha=0.1, color='#008B8B')
    ax1.plot(np.arange(len(threshold_value)), base_mean, 'o-', markersize=3,
             label='DPA2DL', color='#6B8E23')
    ax1.fill_between(np.arange(len(threshold_value)), base_mean - base_std,
                     base_mean + base_std, alpha=0.1, color='#6B8E23')
    ax1.set_title('Hyperparameter Analysis of threshold $\\xi$', fontsize=19)
    ax1.set_xticks(np.arange(len(threshold_value)))
    ax1.set_xticklabels(threshold_value, fontsize=21)
    ax1.yaxis.set_major_locator(MultipleLocator(0.5))
    ax1.tick_params(axis='y', labelsize=21)
    ax1.set_ylabel('Recall@50 (%)', fontsize=21)
    ax1.grid(True, which='major', linestyle=':', linewidth=0.8)
    ax1.minorticks_on()
    ax1.tick_params(which='both', direction='in')
    ax1.legend(loc='upper center', ncol=3, fontsize=8)
    ax1.xaxis.set_ticks_position('both')
    ax1.yaxis.set_ticks_position('both')

    alpha_odd_mean = np.array([2.121, 1.401, 0.455, 0.25])
    alpha_odd_std = np.array([1.095, 0.774, 0.526, 0.288])
    alpha_even_mean = np.array([0.816, 2.121, 1.874, 1.415])
    alpha_even_std = np.array([0.564, 1.095, 1.317, 1.147])
    base_mean = np.array([0.428, 0.428, 0.428, 0.428])
    base_std = np.array([0.634447791, 0.634447791, 0.634447791, 0.634447791])
    alpha_value = [1., 10., 100., 1000.]
    ax2.plot(np.arange(len(alpha_value)), alpha_odd_mean, 's-', markersize=3,
             label='odd weight $\\alpha_{odd}$', color='#FF8C00')
    ax2.fill_between(np.arange(len(alpha_value)), alpha_odd_mean - alpha_odd_std,
                     alpha_odd_mean + alpha_odd_std, alpha=0.1, color='#FF8C00')
    ax2.plot(np.arange(len(alpha_value)), alpha_even_mean, 'x-', markersize=3,
             label='even weight $\\alpha_{even}$', color='#008B8B')
    ax2.fill_between(np.arange(len(alpha_value)), alpha_even_mean - alpha_even_std,
                     alpha_even_mean + alpha_even_std, alpha=0.1, color='#008B8B')
    ax2.plot(np.arange(len(alpha_value)), base_mean, 'o-', markersize=3,
             label='DPA2DL', color='#6B8E23')
    ax2.fill_between(np.arange(len(alpha_value)), base_mean - base_std,
                     base_mean + base_std, alpha=0.1, color='#6B8E23')
    ax2.set_title('Hyperparameter Analysis of weight $\\alpha$', fontsize=19)
    ax2.set_xticks(np.arange(len(alpha_value)))
    ax2.set_xticklabels(alpha_value, fontsize=21)
    ax2.yaxis.set_major_locator(MultipleLocator(0.5))
    ax2.tick_params(axis='y', labelsize=21)
    ax2.grid(True, which='major', linestyle=':', linewidth=0.8)
    ax2.minorticks_on()
    ax2.tick_params(which='both', direction='in')
    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')
    ax2.legend(loc='upper center', ncol=3, fontsize=8)
    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)
    pdf.close()
    """

if __name__ == '__main__':
    main()