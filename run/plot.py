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

def darken_color(color, amount=0.5):
    """
    将给定颜色变暗一些
    amount: 取值范围从0(无变化)到1(完全黑)
    """
    import matplotlib.colors as mcolors
    c = mcolors.colorConverter.to_rgb(color)
    return max(c[0] - amount, 0), max(c[1] - amount, 0), max(c[2] - amount, 0)


def main():

    full_retrain = [0.072709367, 0.278802127, 0.320085526, 0.349425495, 0.382153869]
    full_retrain_wt_gp = [0.140626445, 0.303476125, 0.347834647, 0.380880475, 0.398396462]
    full_retrain_time = [4.05, 19.135, 38.178, 199.45, 382.09]
    full_retrain_wt_gp_time = [3.837, 20.095, 41.023, 202.052, 412.404]
    epochs = [1, 5, 10, 50, 100]
    pdf = PdfPages('retrain_gowalla.pdf')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, constrained_layout=True, figsize=(9, 7))
    ax1.plot(np.arange(len(epochs)), np.array(full_retrain), 'x--', markersize=7, label='Original training',
             color='#7570A0', linewidth=2, markeredgecolor=darken_color('#7570A0', 0.2))
    ax1.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), '3-', markersize=10, label='Training enhanced by GP',
             color='#F7B76D', linewidth=2, markeredgecolor=darken_color('#F7B76D', 0.4))
    for i, txt in enumerate(full_retrain):
        ax1.text(i + 0.1, full_retrain[i] - 0.02, '{:.1f}s'.format(full_retrain_time[i]),
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
    ax1.grid(True, which='major', linestyle='--', linewidth=0.8)
    ax1.minorticks_on()
    ax1.tick_params(which='both', direction='in')
    ax1.xaxis.set_ticks_position('both')
    ax1.yaxis.set_ticks_position('both')

    full_retrain = [0.284897178, 0.314809233, 0.357316643, 0.610431731, 0.610431731]
    full_retrain_wt_gp = [0.334004045, 0.454284012, 0.426267385, 0.624437809, 0.624437809]
    full_retrain_time = [3.965, 20.566, 39.341, 198.775, 391.665]
    full_retrain_wt_gp_time = [4.138, 21.172, 42.72, 204.173, 407.382]
    epochs = [1, 5, 10, 50, 100]
    ax2.plot(np.arange(len(epochs)), np.array(full_retrain), 'x--', markersize=7, label='Original training',
             color='#7570A0', linewidth=2, markeredgecolor=darken_color('#7570A0', 0.2))
    ax2.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), '3-', markersize=10, label='Training enhanced by GP',
             color='#F7B76D', linewidth=2, markeredgecolor=darken_color('#F7B76D', 0.4))
    for i, txt in enumerate(full_retrain):
        ax2.text(i + 0.1, full_retrain[i] - 0.02, '{:.1f}s'.format(full_retrain_time[i]),
                 fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    for i, txt in enumerate(full_retrain_wt_gp_time):
        ax2.text(i + 0.1, full_retrain_wt_gp[i], '{:.1f}s'.format(full_retrain_wt_gp_time[i]),
                 fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    ax2.set_xticks(np.arange(len(epochs)))
    ax2.set_xticklabels(epochs, fontsize=21)
    ax2.yaxis.set_major_locator(MultipleLocator(0.1))
    ax2.tick_params(axis='y', labelsize=21)
    ax2.set_title('LightGCN-BPR', fontsize=21)
    ax2.grid(True, which='major', linestyle='--', linewidth=0.8)
    ax2.minorticks_on()
    ax2.tick_params(which='both', direction='in')
    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')

    full_retrain = [0.092861742, 0.45056802, 0.500385225, 0.583127856, 0.654973984]
    full_retrain_wt_gp = [0.393685371, 0.546095967, 0.59344399, 0.656317115, 0.719439507]
    full_retrain_time = [4.626, 21.54, 47.484, 234.485, 455.21]
    full_retrain_wt_gp_time = [5.656, 24.673, 52.94, 250.384, 531.372]
    epochs = [1, 5, 10, 50, 100]
    ax3.plot(np.arange(len(epochs)), np.array(full_retrain), 'x--', markersize=7, label='Original training',
             color='#7570A0', linewidth=2, markeredgecolor=darken_color('#7570A0', 0.2))
    ax3.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), '3-', markersize=10, label='Training enhanced by GP',
             color='#F7B76D', linewidth=2, markeredgecolor=darken_color('#F7B76D', 0.4))
    for i, txt in enumerate(full_retrain):
        ax3.text(i + 0.1, full_retrain[i] - 0.02, '{:.1f}s'.format(full_retrain_time[i]),
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
    ax3.grid(True, which='major', linestyle='--', linewidth=0.8)
    ax3.minorticks_on()
    ax3.tick_params(which='both', direction='in')
    ax3.xaxis.set_ticks_position('both')
    ax3.yaxis.set_ticks_position('both')
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

    full_retrain = [0.386606574, 0.489965975, 0.5626055, 0.617218792, 0.617218792]
    full_retrain_wt_gp = [0.431357354, 0.554632962, 0.581400335, 0.618135214, 0.61813283]
    full_retrain_time = [5.397, 23.875, 49.476, 240.912, 494.612]
    full_retrain_wt_gp_time = [5.395, 27.987, 52.184, 272.318, 520.435]
    epochs = [1, 5, 10, 50, 100]
    ax4.plot(np.arange(len(epochs)), np.array(full_retrain), 'x--', markersize=7, label='Original training',
             color='#7570A0', linewidth=2, markeredgecolor=darken_color('#7570A0', 0.2))
    ax4.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), '3-', markersize=10, label='Training enhanced by GP',
             color='#F7B76D', linewidth=2, markeredgecolor=darken_color('#F7B76D', 0.4))
    for i, txt in enumerate(full_retrain):
        ax4.text(i + 0.1, full_retrain[i] - 0.02, '{:.1f}s'.format(full_retrain_time[i]),
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
    ax4.grid(True, which='major', linestyle='--', linewidth=0.8)
    ax4.minorticks_on()
    ax4.tick_params(which='both', direction='in')
    ax4.xaxis.set_ticks_position('both')
    ax4.yaxis.set_ticks_position('both')
    plt.tight_layout(rect=[0, 0., 1, 0.91])
    handles, labels = [], []
    for h, l in zip(*ax1.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
    fig.legend(handles, labels, loc='upper center', ncol=len(handles), fontsize=18)
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
    ax1.plot(np.arange(len(epochs)), np.array(full_retrain), 'x--', markersize=5, label='Original training', color='#B1BCC6', linewidth=2)
    ax1.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), '*-', markersize=7, label='Training enhanced by GP', color='#CBD8C3', linewidth=2)
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
    ax2.plot(np.arange(len(epochs)), np.array(full_retrain), 'x--', markersize=5, label='Original training', color='#B1BCC6', linewidth=2)
    ax2.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), '*-', markersize=7, label='Training enhanced by GP', color='#CBD8C3', linewidth=2)
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
    ax3.plot(np.arange(len(epochs)), np.array(full_retrain), 'x--', markersize=5, label='Original training', color='#B1BCC6', linewidth=2)
    ax3.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), '*-', markersize=7, label='Training enhanced by GP', color='#CBD8C3', linewidth=2)
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
    ax4.plot(np.arange(len(epochs)), np.array(full_retrain), 'x--', markersize=5, label='Original training', color='#B1BCC6', linewidth=2)
    ax4.plot(np.arange(len(epochs)), np.array(full_retrain_wt_gp), '*-', markersize=7, label='Training enhanced by GP', color='#CBD8C3', linewidth=2)
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

    mean_retraining = [783.2894, 698.841, 850.432]
    mean_all = [1027.1948, 714.0858, 885.1246]
    methods = ['PGA', 'RevAdv', 'DPA2DL']
    idx = np.arange(len(methods))
    width = 0.4
    pdf = PdfPages('retrain_time.pdf')
    fig, (ax0, ax2) = plt.subplots(nrows=2, ncols=1, constrained_layout=True, figsize=(9, 4),
                                        gridspec_kw={'height_ratios': [3, 1]})
    ax0.barh(idx, mean_retraining, width, label='Retraining time',
            color='#ddebf7', edgecolor='black', linewidth=1, hatch='.')
    ax0.barh(idx, [mean_all[i] - mean_retraining[i] for i in range(len(methods))], width, label='Other time',
            left=mean_retraining, color='#e2f0d9', edgecolor='black', linewidth=1, hatch='/')
    ax0.set_ylim(-0.5, 2.5)
    ax0.set_yticks(idx)
    ax0.set_yticklabels(methods, fontsize=18)
    ax0.tick_params(axis='x', labelsize=18)
    ax0.set_xlabel('Consumed Time (s) on Gowalla', fontsize=18)
    ax0.grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.8)
    ax0.tick_params(which='both', direction='in')
    ax0.xaxis.set_ticks_position('both')
    ax0.yaxis.set_ticks_position('both')

    """
    mean_retraining = [2212.7436, 2546.9886]
    mean_all = [2621.9806, 2691.1378]
    methods = ['PGA', 'DPA2DL']
    idx = np.arange(len(methods))
    ax1.barh(idx, mean_retraining, width, label='Retraining time',
            color='#ddebf7', edgecolor='black', linewidth=1, hatch='.')
    ax1.barh(idx, [mean_all[i] - mean_retraining[i] for i in range(len(methods))], width, label='Other time',
            left=mean_retraining, color='#e2f0d9', edgecolor='black', linewidth=1, hatch='/')
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_yticks(idx)
    ax1.set_yticklabels(methods, fontsize=18)
    ax1.tick_params(axis='x', labelsize=18)
    ax1.set_xlabel('Consumed Time (s) on Yelp', fontsize=18)
    ax1.grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.8)
    ax1.tick_params(which='both', direction='in')
    ax1.xaxis.set_ticks_position('both')
    ax1.yaxis.set_ticks_position('both')
    """

    mean_retraining = [16160.217]
    mean_all = [20004.6245]
    methods = ['DPA2DL']
    idx = np.arange(len(methods))
    ax2.barh(idx, mean_retraining, width, label='Retraining time',
            color='#ddebf7', edgecolor='black', linewidth=1, hatch='.')
    ax2.barh(idx, [mean_all[i] - mean_retraining[i] for i in range(len(methods))], width, label='Other time',
            left=mean_retraining, color='#e2f0d9', edgecolor='black', linewidth=1, hatch='/')
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_yticks(idx)
    ax2.set_yticklabels(methods, fontsize=18)
    ax2.tick_params(axis='x', labelsize=18)
    ax2.set_xlabel('Consumed Time (s) on Tenrec  ', fontsize=18)
    ax2.grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.8)
    ax2.tick_params(which='both', direction='in')
    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')
    plt.tight_layout(rect=[0, 0., 1, 0.86])
    handles, labels = [], []
    for h, l in zip(*ax0.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
    fig.legend(handles, labels, loc='upper center', ncol=len(handles), fontsize=18)
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    grad_sims = np.load('grad_sim_gowalla.npy')[:100, :]
    epochs = np.arange(grad_sims.shape[0])
    pdf = PdfPages('grad_sim.pdf')
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(8, 3.5), sharey='all')
    ax1.plot(epochs, grad_sims[:, 0], '-', markersize=0, label='Interacted user-item pairs', color='#7570A0', linewidth=2)
    ax1.fill_between(epochs, grad_sims[:, 0] - grad_sims[:, 1], grad_sims[:, 0] + grad_sims[:, 1], alpha=0.1, color='#7570A0')
    ax1.plot(epochs, grad_sims[:, 2], '-', markersize=0, label='Random user-item pairs', color='#F7B76D', linewidth=2)
    ax1.fill_between(epochs, grad_sims[:, 2] - grad_sims[:, 3], grad_sims[:, 2] + grad_sims[:, 3], alpha=0.1, color='#F7B76D')
    ax1.set_xlabel('Training Epoch', fontsize=18)
    ax1.set_ylabel('Similarity', fontsize=18)
    ax1.set_ylim(-0.3, 0.8)
    ax1.tick_params(axis='both', labelsize=18)
    ax1.set_title('Gowalla', fontsize=18)
    ax1.grid(True, which='major', linestyle='--', linewidth=0.8)
    ax1.minorticks_on()
    ax1.tick_params(which='both', direction='in')
    ax1.xaxis.set_ticks_position('both')
    ax1.yaxis.set_ticks_position('both')

    grad_sims = np.load('grad_sim_yelp.npy')[:100, :]
    epochs = np.arange(grad_sims.shape[0])
    ax2.plot(epochs, grad_sims[:, 0], '-', markersize=0, label='Interacted user-item pairs', color='#7570A0', linewidth=2)
    ax2.fill_between(epochs, grad_sims[:, 0] - grad_sims[:, 1], grad_sims[:, 0] + grad_sims[:, 1], alpha=0.1, color='#7570A0')
    ax2.plot(epochs, grad_sims[:, 2], '-', markersize=0, label='Random user-item pairs', color='#F7B76D', linewidth=2)
    ax2.fill_between(epochs, grad_sims[:, 2] - grad_sims[:, 3], grad_sims[:, 2] + grad_sims[:, 3], alpha=0.1, color='#F7B76D')
    ax2.set_xlabel('Training Epoch', fontsize=18)
    ax2.set_ylim(-0.3, 0.8)
    ax2.tick_params(axis='both', labelsize=18)
    ax2.set_title('Yelp', fontsize=18)
    ax2.grid(True, which='major', linestyle='--', linewidth=0.8)
    ax2.minorticks_on()
    ax2.tick_params(which='both', direction='in')
    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')
    plt.tight_layout(rect=[-0.04, 0., 1, 0.85])
    handles, labels = [], []
    for h, l in zip(*ax1.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
    fig.legend(handles, labels, loc='upper center', ncol=len(handles), fontsize=16)
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    threshold_odd_mean = np.array([1.849, 1.863, 1.64])
    threshold_even_mean = np.array([1.863, 0.761, 0.395])
    base_mean = np.array([0.4584, 0.4584, 0.4584])
    threshold_value = [-np.inf, 0., np.inf]
    pdf = PdfPages('hyper.pdf')
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(10, 3.5), sharey='all')
    ax1.plot(np.arange(len(threshold_value)), threshold_odd_mean, 'x-', markersize=7,
             label='$\\xi_{odd}$', color='#7570A0', linewidth=2, markeredgecolor=darken_color('#7570A0', 0.2))
    ax1.plot(np.arange(len(threshold_value)), threshold_even_mean, '3-', markersize=10,
             label='$\\xi_{even}$', color='#F7B76D', linewidth=2, markeredgecolor=darken_color('#F7B76D', 0.4))
    ax1.plot(np.arange(len(threshold_value)), base_mean, '2-', markersize=10,
             label='DPA2DL', color='#527F76', linewidth=2, markeredgecolor=darken_color('#527F76', 0.2))
    ax1.set_title('Analysis of threshold $\\xi$', fontsize=19)
    ax1.set_xticks(np.arange(len(threshold_value)))
    ax1.set_xticklabels(threshold_value, fontsize=21)
    ax1.yaxis.set_major_locator(MultipleLocator(0.5))
    ax1.tick_params(axis='y', labelsize=21)
    ax1.set_ylabel('Recall@50 (%)', fontsize=21)
    ax1.grid(True, which='major', linestyle='--', linewidth=0.8)
    ax1.minorticks_on()
    ax1.set_ylim(0., 2.2)
    ax1.tick_params(which='major', direction='in')
    ax1.xaxis.set_ticks_position('both')
    ax1.yaxis.set_ticks_position('both')

    alpha_odd_mean = np.array([1.582, 1.863, 1.394, 0.517, 0.225])
    alpha_even_mean = np.array([0.376, 0.688, 1.863, 1.583, 1.499])
    base_mean = np.array([0.4584, 0.4584, 0.4584, 0.4584, 0.4584])
    alpha_value = [0.1, 1., 10., 100., 1000.]
    ax2.plot(np.arange(len(alpha_value)), alpha_odd_mean, 'x-', markersize=7,
             label='$\\alpha_{odd}$', color='#7570A0', linewidth=2, markeredgecolor=darken_color('#7570A0', 0.2))
    ax2.plot(np.arange(len(alpha_value)), alpha_even_mean, '3-', markersize=10,
             label='$\\alpha_{even}$', color='#F7B76D', linewidth=2, markeredgecolor=darken_color('#F7B76D', 0.4))
    ax2.plot(np.arange(len(alpha_value)), base_mean, '2-', markersize=10,
             label='DPA2DL', color='#527F76', linewidth=2, markeredgecolor=darken_color('#527F76', 0.2))
    ax2.set_title('Analysis of weight $\\alpha$', fontsize=19)
    ax2.set_xticks(np.arange(len(alpha_value)))
    ax2.set_xticklabels(alpha_value, fontsize=21)
    ax2.yaxis.set_major_locator(MultipleLocator(0.5))
    ax2.tick_params(axis='y', labelsize=21)
    ax2.grid(True, which='major', linestyle='--', linewidth=0.8)
    ax2.minorticks_on()
    ax2.set_ylim(0., 2.2)
    ax2.tick_params(which='both', direction='in')
    ax2.xaxis.set_ticks_position('both')
    ax2.yaxis.set_ticks_position('both')
    plt.tight_layout(rect=[0., 0., 1, 0.88])
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