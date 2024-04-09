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
import matplotlib.ticker as ticker
plt.rc('font', family='Times New Roman')
plt.rcParams['pdf.fonttype'] = 42


def main():

    full_retrain = [0.072709367, 0.278802127, 0.320085526, 0.349425495, 0.382153869]
    full_retrain_wt_gp = [0.140626445, 0.303476125, 0.347834647, 0.380880475, 0.398396462]
    full_retrain_time = [4.05, 19.135, 38.178, 199.45, 382.09]
    full_retrain_wt_gp_time = [3.837, 20.095, 41.023, 202.052, 412.404]
    epochs = [1, 5, 10, 50, 100]
    pdf = PdfPages('retrain_gowalla.pdf')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, constrained_layout=True, figsize=(9, 7), sharex='all')
    ax1.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='Original training', color='#B1BCC6', linewidth=2)
    ax1.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), 'x-', markersize=3, label='Training enhanced by GP', color='#B5C0AE', linewidth=2)
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
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.minorticks_on()
    ax1.tick_params(which='both', direction='in')
    ax1.xaxis.set_ticks_position('both')
    ax1.yaxis.set_ticks_position('both')

    full_retrain = [0.284897178, 0.314809233, 0.357316643, 0.610431731, 0.610431731]
    full_retrain_wt_gp = [0.334004045, 0.454284012, 0.426267385, 0.624437809, 0.624437809]
    full_retrain_time = [3.965, 20.566, 39.341, 198.775, 391.665]
    full_retrain_wt_gp_time = [4.138, 21.172, 42.72, 204.173, 407.382]
    epochs = [1, 5, 10, 50, 100]
    ax2.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='Original training', color='#B1BCC6', linewidth=2)
    ax2.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), 'x-', markersize=3, label='Training enhanced by GP', color='#B5C0AE', linewidth=2)
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
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.minorticks_on()
    ax2.tick_params(which='both', direction='in')
    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')

    full_retrain = [0.092861742, 0.45056802, 0.500385225, 0.583127856, 0.654973984]
    full_retrain_wt_gp = [0.393685371, 0.546095967, 0.59344399, 0.656317115, 0.719439507]
    full_retrain_time = [4.626, 21.54, 47.484, 234.485, 455.21]
    full_retrain_wt_gp_time = [5.656, 24.673, 52.94, 250.384, 531.372]
    epochs = [1, 5, 10, 50, 100]
    ax3.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='Original training', color='#B1BCC6', linewidth=2)
    ax3.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), 'x-', markersize=3, label='Training enhanced by GP', color='#B5C0AE', linewidth=2)
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
    ax3.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax3.minorticks_on()
    ax3.tick_params(which='both', direction='in')
    ax3.xaxis.set_ticks_position('both')
    ax3.yaxis.set_ticks_position('both')
    ax3.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    full_retrain = [0.386606574, 0.489965975, 0.5626055, 0.617218792, 0.617218792]
    full_retrain_wt_gp = [0.431357354, 0.554632962, 0.581400335, 0.618135214, 0.61813283]
    full_retrain_time = [5.397, 23.875, 49.476, 240.912, 494.612]
    full_retrain_wt_gp_time = [5.395, 27.987, 52.184, 272.318, 520.435]
    epochs = [1, 5, 10, 50, 100]
    ax4.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='Original training', color='#B1BCC6', linewidth=2)
    ax4.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), 'x-', markersize=3, label='Training enhanced by GP', color='#B5C0AE', linewidth=2)
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
    ax4.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax4.minorticks_on()
    ax4.tick_params(which='both', direction='in')
    ax4.xaxis.set_ticks_position('both')
    ax4.yaxis.set_ticks_position('both')
    plt.tight_layout(rect=[0, 0., 1, 0.91])
    handles, labels = [], []
    for h, l in zip(*ax1.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
    fig.legend(handles, labels, loc='upper center', ncol=len(handles), fontsize=21)
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    """
    full_retrain = [0.135182157, 0.281044066, 0.341998488]
    full_retrain_wt_gp = [0.185392991, 0.347992837, 0.425070107]
    full_retrain_time = [231.375, 1156.915, 2152.129]
    full_retrain_wt_gp_time = [254.065, 1117.596, 2297.379]
    epochs = [1, 5, 10]
    pdf = PdfPages('retrain_tenrec.pdf')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, constrained_layout=True, figsize=(8, 5))
    ax1.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='Original training', color='#B1BCC6', linewidth=2)
    ax1.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), 'x-', markersize=3, label='Training enhanced by GP', color='#CBD8C3', linewidth=2)
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
    ax2.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='Original training', color='#B1BCC6', linewidth=2)
    ax2.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), 'x-', markersize=3, label='Training enhanced by GP', color='#CBD8C3', linewidth=2)
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
    ax3.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='Original training', color='#B1BCC6', linewidth=2)
    ax3.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), 'x-', markersize=3, label='Training enhanced by GP', color='#CBD8C3', linewidth=2)
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
    ax4.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='Original training', color='#B1BCC6', linewidth=2)
    ax4.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), 'x-', markersize=3, label='Training enhanced by GP', color='#CBD8C3', linewidth=2)
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

    mean_retraining = [783.2894, 698.841, 821.3908]
    mean_all = [1027.1948, 714.0858, 860.0182]
    methods = ['PGA', 'RevAdv', 'DPA2DL']
    idx = np.arange(len(methods))[::-1]
    width = 0.4
    pdf = PdfPages('retrain_time.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(9, 3.5))
    ax.barh(idx, mean_retraining, width, label='Retraining time',
            color='#ddebf7', edgecolor='black', linewidth=1, hatch='.')
    ax.barh(idx, [mean_all[i] - mean_retraining[i] for i in range(len(methods))], width, label='Other time',
            left=mean_retraining, color='#e2f0d9', edgecolor='black', linewidth=1, hatch='/')

    ax.set_yticks(idx)
    ax.set_yticklabels(methods, fontsize=21)
    ax.tick_params(axis='x', labelsize=21)
    ax.set_xlabel('Consumed Time (s)', fontsize=21)
    ax.grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.8)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.tight_layout(rect=[0, 0., 1, 0.82])
    handles, labels = [], []
    for h, l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
    fig.legend(handles, labels, loc='upper center', ncol=len(handles), fontsize=21)
    pdf.savefig()
    plt.close(fig)
    pdf.close()
    """

    grad_sims = np.load('grad_sim_gowalla.npy')[:100, :]
    epochs = np.arange(grad_sims.shape[0])
    pdf = PdfPages('grad_sim.pdf')
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(8, 3), sharey='all')
    ax1.plot(epochs, grad_sims[:, 0], '-', markersize=0, label='Interacted user-item pairs', color='#B1BCC6', linewidth=2)
    ax1.fill_between(epochs, grad_sims[:, 0] - grad_sims[:, 1], grad_sims[:, 0] + grad_sims[:, 1], alpha=0.3, color='#ddebf7')
    ax1.plot(epochs, grad_sims[:, 2], '-', markersize=0, label='Random user-item pairs', color='#B5C0AE', linewidth=2)
    ax1.fill_between(epochs, grad_sims[:, 2] - grad_sims[:, 3], grad_sims[:, 2] + grad_sims[:, 3], alpha=0.3, color='#e2f0d9')
    ax1.set_xlabel('Training Epoch', fontsize=13)
    ax1.set_ylabel('Cosine Similarity', fontsize=13)
    ax1.set_ylim(-0.3, 0.8)
    ax1.tick_params(axis='both', labelsize=13)
    ax1.set_title('Gowalla', fontsize=13)
    ax1.grid(True, which='both', linestyle=':', linewidth=1.2)
    ax1.minorticks_on()
    ax1.tick_params(which='both', direction='in')
    ax1.xaxis.set_ticks_position('both')
    ax1.yaxis.set_ticks_position('both')

    grad_sims = np.load('grad_sim_yelp.npy')[:100, :]
    epochs = np.arange(grad_sims.shape[0])
    ax2.plot(epochs, grad_sims[:, 0], '-', markersize=0, label='Interacted user-item pairs', color='#B1BCC6', linewidth=2)
    ax2.fill_between(epochs, grad_sims[:, 0] - grad_sims[:, 1], grad_sims[:, 0] + grad_sims[:, 1], alpha=0.3, color='#ddebf7')
    ax2.plot(epochs, grad_sims[:, 2], '-', markersize=0, label='Random user-item pairs', color='#B5C0AE', linewidth=2)
    ax2.fill_between(epochs, grad_sims[:, 2] - grad_sims[:, 3], grad_sims[:, 2] + grad_sims[:, 3], alpha=0.3, color='#e2f0d9')
    ax2.set_xlabel('Training Epoch', fontsize=13)
    ax2.set_ylim(-0.3, 0.8)
    ax2.tick_params(axis='both', labelsize=13)
    ax2.set_title('Yelp', fontsize=13)
    ax2.grid(True, which='both', linestyle=':', linewidth=1.2)
    ax2.minorticks_on()
    ax2.tick_params(which='both', direction='in')
    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')
    plt.tight_layout(rect=[0, 0., 1, 0.86])
    handles, labels = [], []
    for h, l in zip(*ax1.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
    fig.legend(handles, labels, loc='upper center', ncol=len(handles), fontsize=13)
    pdf.savefig()
    plt.close(fig)
    pdf.close()


    threshold_odd_mean = np.array([1.978, 2.005, 1.662])
    threshold_even_mean = np.array([2.005, 0.811, 0.361])
    base_mean = np.array([0.428, 0.428, 0.428])
    threshold_value = [-np.inf, 0., np.inf]
    pdf = PdfPages('hyper.pdf')
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(10, 3.5), sharey='all')
    ax1.plot(np.arange(len(threshold_value)), threshold_odd_mean, 's-', markersize=3,
             label='$\\xi_{odd}$', color='#B1BCC6', linewidth=2)
    ax1.plot(np.arange(len(threshold_value)), threshold_even_mean, 'x-', markersize=3,
             label='$\\xi_{even}$', color='#B5C0AE', linewidth=2)
    ax1.plot(np.arange(len(threshold_value)), base_mean, 'o-', markersize=3,
             label='DPA2DL', color='black', linewidth=2)
    ax1.set_title('Analysis of threshold $\\xi$', fontsize=19)
    ax1.set_xticks(np.arange(len(threshold_value)))
    ax1.set_xticklabels(threshold_value, fontsize=21)
    ax1.yaxis.set_major_locator(MultipleLocator(0.5))
    ax1.tick_params(axis='y', labelsize=21)
    ax1.set_ylabel('Recall@50 (%)', fontsize=21)
    ax1.grid(True, which='both', linestyle=':', linewidth=0.8)
    ax1.minorticks_on()
    ax1.set_ylim(0., 2.2)
    ax1.tick_params(which='both', direction='in')
    ax1.xaxis.set_ticks_position('both')
    ax1.yaxis.set_ticks_position('both')

    alpha_odd_mean = np.array([1.67, 2.005, 1.344, 0.364, 0.23])
    alpha_even_mean = np.array([0.4, 0.834, 2.005, 1.798, 1.453])
    base_mean = np.array([0.428, 0.428, 0.428, 0.428, 0.428])
    alpha_value = [0.1, 1., 10., 100., 1000.]
    ax2.plot(np.arange(len(alpha_value)), alpha_odd_mean, 's-', markersize=3,
             label='$\\alpha_{odd}$', color='#B1BCC6', linewidth=2)
    ax2.plot(np.arange(len(alpha_value)), alpha_even_mean, 'x-', markersize=3,
             label='$\\alpha_{even}$', color='#B5C0AE', linewidth=2)
    ax2.plot(np.arange(len(alpha_value)), base_mean, 'o-', markersize=3,
             label='DPA2DL', color='black', linewidth=2)
    ax2.set_title('Analysis of weight $\\alpha$', fontsize=19)
    ax2.set_xticks(np.arange(len(alpha_value)))
    ax2.set_xticklabels(alpha_value, fontsize=21)
    ax2.yaxis.set_major_locator(MultipleLocator(0.5))
    ax2.tick_params(axis='y', labelsize=21)
    ax2.grid(True, which='both', linestyle=':', linewidth=0.8)
    ax2.minorticks_on()
    ax2.set_ylim(0., 2.2)
    ax2.tick_params(which='both', direction='in')
    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')
    plt.tight_layout(rect=[0, 0., 1, 0.88])
    handles, labels = [], []
    for h, l in zip(*ax1.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
    fig.legend(handles, labels, loc='upper center', ncol=len(handles), fontsize=16, bbox_to_anchor=(0.29, 1.), columnspacing=1.)
    handles, labels = [], []
    for h, l in zip(*ax2.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
    fig.legend(handles, labels, loc='upper center', ncol=len(handles), fontsize=16, bbox_to_anchor=(0.755, 1.), columnspacing=1.)
    pdf.savefig()
    plt.close(fig)
    pdf.close()


if __name__ == '__main__':
    main()