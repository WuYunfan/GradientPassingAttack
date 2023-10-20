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
    # [None, None, None, None, None, None]

    full_retrain = [0.075286202, 0.281853884, 0.322035164, 0.330106795, 0.352520406, 0.386516035]
    full_retrain_wt_GP = [0.136697352, 0.290027261, 0.330552369, 0.336562991, 0.374702543, 0.395789742]
    pre_retrain = [0.216898873, 0.279095858, 0.323871017, 0.341205478, 0.335344315, 0.362560958]
    pre_retrain_wt_GP = [0.14917773, 0.281438619, 0.289617866, 0.344945639, 0.340409458, 0.366380721]
    epochs = [1, 5, 10, 20, 50, 100]
    pdf = PdfPages('retrain_gowalla_mf_bpr.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(4, 4))
    ax.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='full_retrain', color='red')
    ax.plot(np.arange(len(epochs)), np.array(full_retrain_wt_GP), 'x-', markersize=3, label='full_retrain_wt_GP', color='red')
    ax.plot(np.arange(len(epochs)), np.array(pre_retrain), 'd--', markersize=3, label='pre_retrain', color='blue')
    ax.plot(np.arange(len(epochs)), np.array(pre_retrain_wt_GP), 'v-', markersize=3, label='pre_retrain_wt_GP', color='blue')
    ax.set_xticks(np.arange(len(epochs)))
    ax.set_xticklabels(epochs)
    ax.set_xlabel('Retraining Epoch', fontsize=17)
    ax.set_ylabel('Jaccard Similarity@50', fontsize=17)
    ax.set_title('Gowalla-MF-BPR', fontsize=17)
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

    full_retrain = [0.093881384, 0.457104355, 0.49532634, 0.529270589, 0.579583585, 0.657972336]
    full_retrain_wt_GP = [0.231648549, 0.471861839, 0.508978844, 0.538056493, 0.59916538, 0.67374748]
    pre_retrain = [0.206953079, 0.542729735, 0.622267127, 0.670113921, 0.700894117, 0.683104634]
    pre_retrain_wt_GP = [0.349154055, 0.55354178, 0.633893192, 0.677671671, 0.708003521, 0.695266366]
    epochs = [1, 5, 10, 20, 50, 100]
    pdf = PdfPages('retrain_gowalla_mf_bce.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(4, 4))
    ax.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='full_retrain', color='red')
    ax.plot(np.arange(len(epochs)), np.array(full_retrain_wt_GP), 'x-', markersize=3, label='full_retrain_wt_GP', color='red')
    ax.plot(np.arange(len(epochs)), np.array(pre_retrain), 'd--', markersize=3, label='pre_retrain', color='blue')
    ax.plot(np.arange(len(epochs)), np.array(pre_retrain_wt_GP), 'v-', markersize=3, label='pre_retrain_wt_GP', color='blue')
    ax.set_xticks(np.arange(len(epochs)))
    ax.set_xticklabels(epochs)
    ax.set_xlabel('Retraining Epoch', fontsize=17)
    ax.set_ylabel('Jaccard Similarity@50', fontsize=17)
    ax.set_title('Gowalla-MF-BCE', fontsize=17)
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

    full_retrain = [0.018065507, 0.458145171, 0.605804324, 0.791859567, 0.883519351, 0.855535269]
    full_retrain_wt_GP = [0.147620425, 0.55498302, 0.654133379, 0.791915298, 0.883646131, 0.855554283]
    pre_retrain = [0.077724367, 0.782507241, 0.847826719, 0.908701897, 0.935608387, 0.947650433]
    pre_retrain_wt_GP = [0.412097782, 0.619480371, 0.84749496, 0.78099072, 0.78099072, 0.78099072]
    epochs = [1, 5, 10, 20, 50, 100]
    pdf = PdfPages('retrain_gowalla_mf_mse.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(4, 4))
    ax.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='full_retrain', color='red')
    ax.plot(np.arange(len(epochs)), np.array(full_retrain_wt_GP), 'x-', markersize=3, label='full_retrain_wt_GP', color='red')
    ax.plot(np.arange(len(epochs)), np.array(pre_retrain), 'd--', markersize=3, label='pre_retrain', color='blue')
    ax.plot(np.arange(len(epochs)), np.array(pre_retrain_wt_GP), 'v-', markersize=3, label='pre_retrain_wt_GP', color='blue')
    ax.set_xticks(np.arange(len(epochs)))
    ax.set_xticklabels(epochs)
    ax.set_xlabel('Retraining Epoch', fontsize=17)
    ax.set_ylabel('Jaccard Similarity@50', fontsize=17)
    ax.set_title('Gowalla-MF-MSE', fontsize=17)
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

    full_retrain = [0.272299916, 0.302604526, 0.345752031, 0.429740041, 0.579233468, 0.594510257]
    full_retrain_wt_GP = [0.301306635, 0.328880131, 0.359562844, 0.466831774, 0.601292491, 0.598199308]
    pre_retrain = [0.297682077, 0.353523135, 0.4299301217, 0.516808569, 0.555641949, 0.513404012]
    pre_retrain_wt_GP = [0.328485936, 0.372067511, 0.461907566, 0.533941805, 0.559640884, 0.551152229]
    epochs = [1, 5, 10, 20, 50, 100]
    pdf = PdfPages('retrain_gowalla_lgcn_bpr.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(4, 4))
    ax.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='full_retrain', color='red')
    ax.plot(np.arange(len(epochs)), np.array(full_retrain_wt_GP), 'x-', markersize=3, label='full_retrain_wt_GP', color='red')
    ax.plot(np.arange(len(epochs)), np.array(pre_retrain), 'd--', markersize=3, label='pre_retrain', color='blue')
    ax.plot(np.arange(len(epochs)), np.array(pre_retrain_wt_GP), 'v-', markersize=3, label='pre_retrain_wt_GP', color='blue')
    ax.set_xticks(np.arange(len(epochs)))
    ax.set_xticklabels(epochs)
    ax.set_xlabel('Retraining Epoch', fontsize=17)
    ax.set_ylabel('Jaccard Similarity@50', fontsize=17)
    ax.set_title('Gowalla-LightGCN-BPR', fontsize=17)
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

    full_retrain = [0.384082317, 0.491609573, 0.56656003, 0.615256667, 0.620204329, 0.620204329]
    full_retrain_wt_GP = [0.403211951, 0.516695857, 0.585517287, 0.615605414, 0.620204329, 0.639455557]
    pre_retrain = [0.424951434, 0.55832231, 0.601104677, 0.62620765, 0.65176183, 0.676508009]
    pre_retrain_wt_GP = [0.44653815, 0.574776471, 0.6027807, 0.619222641, 0.647773564, 0.668172538]
    epochs = [1, 5, 10, 20, 50, 100]
    pdf = PdfPages('retrain_gowalla_lgcn_bce.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(4, 4))
    ax.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='full_retrain', color='red')
    ax.plot(np.arange(len(epochs)), np.array(full_retrain_wt_GP), 'x-', markersize=3, label='full_retrain_wt_GP', color='red')
    ax.plot(np.arange(len(epochs)), np.array(pre_retrain), 'd--', markersize=3, label='pre_retrain', color='blue')
    ax.plot(np.arange(len(epochs)), np.array(pre_retrain_wt_GP), 'v-', markersize=3, label='pre_retrain_wt_GP', color='blue')
    ax.set_xticks(np.arange(len(epochs)))
    ax.set_xticklabels(epochs)
    ax.set_xlabel('Retraining Epoch', fontsize=17)
    ax.set_ylabel('Jaccard Similarity@50', fontsize=17)
    ax.set_title('Gowalla-LightGCN-BCE', fontsize=17)
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

    full_retrain = [1, 1, 1, 1, 1, 1]
    full_retrain_wt_GP = [1, 1, 1, 1, 1, 1]
    pre_retrain = [0.349318922, 0.402463704, 0.402463704, 0.402462363, 0.402462363, 0.402462363]
    pre_retrain_wt_GP = [0.340363413, 0.370431542, 0.370431542, 0.370431542, 0.370431542, 0.370431542]
    epochs = [1, 5, 10, 20, 50, 100]
    pdf = PdfPages('retrain_gowalla_lgcn_mse.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(4, 4))
    ax.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='full_retrain', color='red')
    ax.plot(np.arange(len(epochs)), np.array(full_retrain_wt_GP), 'x-', markersize=3, label='full_retrain_wt_GP', color='red')
    ax.plot(np.arange(len(epochs)), np.array(pre_retrain), 'd--', markersize=3, label='pre_retrain', color='blue')
    ax.plot(np.arange(len(epochs)), np.array(pre_retrain_wt_GP), 'v-', markersize=3, label='pre_retrain_wt_GP', color='blue')
    ax.set_xticks(np.arange(len(epochs)))
    ax.set_xticklabels(epochs)
    ax.set_xlabel('Retraining Epoch', fontsize=17)
    ax.set_ylabel('Jaccard Similarity@50', fontsize=17)
    ax.set_title('Gowalla-LightGCN-MSE', fontsize=17)
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

    full_retrain = [0.13598834, 0.28148815, 0.342309505]
    full_retrain_wt_GP = [0.175653771, 0.338578314, 0.373460144]
    pre_retrain = [0.205222294, 0.312482774, 0.413624197]
    pre_retrain_wt_GP = [0.255332023, 0.378011674, 0.455757767]
    epochs = [1, 5, 10]
    pdf = PdfPages('retrain_tenrec_mf_bpr.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(4, 4))
    ax.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='full_retrain', color='red')
    ax.plot(np.arange(len(epochs)), np.array(full_retrain_wt_GP), 'x-', markersize=3, label='full_retrain_wt_GP', color='red')
    ax.plot(np.arange(len(epochs)), np.array(pre_retrain), 'd--', markersize=3, label='pre_retrain', color='blue')
    ax.plot(np.arange(len(epochs)), np.array(pre_retrain_wt_GP), 'v-', markersize=3, label='pre_retrain_wt_GP', color='blue')
    ax.set_xticks(np.arange(len(epochs)))
    ax.set_xticklabels(epochs)
    ax.set_xlabel('Retraining Epoch', fontsize=17)
    ax.set_ylabel('Jaccard Similarity@50', fontsize=17)
    ax.set_title('Tenrec-MF-BPR', fontsize=17)
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

    full_retrain = [0.135694683, 0.318301648, 0.370315135]
    full_retrain_wt_GP = [0.160667032, 0.33530426, 0.380230695]
    pre_retrain = [0.217158407, 0.343279243, 0.346117646]
    pre_retrain_wt_GP = [0.231641382, 0.366947889, 0.390629679]
    epochs = [1, 5, 10]
    pdf = PdfPages('retrain_tenrec_mf_bce.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(4, 4))
    ax.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='full_retrain', color='red')
    ax.plot(np.arange(len(epochs)), np.array(full_retrain_wt_GP), 'x-', markersize=3, label='full_retrain_wt_GP', color='red')
    ax.plot(np.arange(len(epochs)), np.array(pre_retrain), 'd--', markersize=3, label='pre_retrain', color='blue')
    ax.plot(np.arange(len(epochs)), np.array(pre_retrain_wt_GP), 'v-', markersize=3, label='pre_retrain_wt_GP', color='blue')
    ax.set_xticks(np.arange(len(epochs)))
    ax.set_xticklabels(epochs)
    ax.set_xlabel('Retraining Epoch', fontsize=17)
    ax.set_ylabel('Jaccard Similarity@50', fontsize=17)
    ax.set_title('Tenrec-MF-BCE', fontsize=17)
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

    full_retrain = [0.032052275, 1, 1]
    full_retrain_wt_GP = [0.032096721, 1, 1]
    pre_retrain = [0.030159269, 0.030716877, 0.030716877]
    pre_retrain_wt_GP = [0.03031054, 0.030716877, 0.030716877]
    epochs = [1, 5, 10]
    pdf = PdfPages('retrain_tenrec_mf_mse.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(4, 4))
    ax.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='full_retrain', color='red')
    ax.plot(np.arange(len(epochs)), np.array(full_retrain_wt_GP), 'x-', markersize=3, label='full_retrain_wt_GP', color='red')
    ax.plot(np.arange(len(epochs)), np.array(pre_retrain), 'd--', markersize=3, label='pre_retrain', color='blue')
    ax.plot(np.arange(len(epochs)), np.array(pre_retrain_wt_GP), 'v-', markersize=3, label='pre_retrain_wt_GP', color='blue')
    ax.set_xticks(np.arange(len(epochs)))
    ax.set_xticklabels(epochs)
    ax.set_xlabel('Retraining Epoch', fontsize=17)
    ax.set_ylabel('Jaccard Similarity@50', fontsize=17)
    ax.set_title('Tenrec-MF-MSE', fontsize=17)
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

    full_retrain = [0.146111056, 0.225820974, 0.240686223]
    full_retrain_wt_GP = [0.156768695, 0.225820988, 0.24878861]
    pre_retrain = [0.184574515, 0.31090644, 0.350310683]
    pre_retrain_wt_GP = [0.19176507, 0.312526673, 0.362075716]
    epochs = [1, 5, 10]
    pdf = PdfPages('retrain_tenrec_lgcn_bpr.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(4, 4))
    ax.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='full_retrain', color='red')
    ax.plot(np.arange(len(epochs)), np.array(full_retrain_wt_GP), 'x-', markersize=3, label='full_retrain_wt_GP', color='red')
    ax.plot(np.arange(len(epochs)), np.array(pre_retrain), 'd--', markersize=3, label='pre_retrain', color='blue')
    ax.plot(np.arange(len(epochs)), np.array(pre_retrain_wt_GP), 'v-', markersize=3, label='pre_retrain_wt_GP', color='blue')
    ax.set_xticks(np.arange(len(epochs)))
    ax.set_xticklabels(epochs)
    ax.set_xlabel('Retraining Epoch', fontsize=17)
    ax.set_ylabel('Jaccard Similarity@50', fontsize=17)
    ax.set_title('Tenrec-LightGCN-BPR', fontsize=17)
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

    full_retrain = [0.195325017, 0.29538098, 0.375609398]
    full_retrain_wt_GP = [0.241570234, 0.295381188, 0.375609398]
    pre_retrain = [0.252017081, 0.371241152, 0.433536589]
    pre_retrain_wt_GP = [0.298810601, 0.371241361, 0.4335365]
    epochs = [1, 5, 10]
    pdf = PdfPages('retrain_tenrec_lgcn_bce.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(4, 4))
    ax.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='full_retrain', color='red')
    ax.plot(np.arange(len(epochs)), np.array(full_retrain_wt_GP), 'x-', markersize=3, label='full_retrain_wt_GP', color='red')
    ax.plot(np.arange(len(epochs)), np.array(pre_retrain), 'd--', markersize=3, label='pre_retrain', color='blue')
    ax.plot(np.arange(len(epochs)), np.array(pre_retrain_wt_GP), 'v-', markersize=3, label='pre_retrain_wt_GP', color='blue')
    ax.set_xticks(np.arange(len(epochs)))
    ax.set_xticklabels(epochs)
    ax.set_xlabel('Retraining Epoch', fontsize=17)
    ax.set_ylabel('Jaccard Similarity@50', fontsize=17)
    ax.set_title('Tenrec-LightGCN-BCE', fontsize=17)
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

    full_retrain = [0.020889737, 0.020889435, 0.999872506]
    full_retrain_wt_GP = [0.037261892, 0.037252951, 0.99987936]
    pre_retrain = [0.008863258, 0.014410376, 0.541428387]
    pre_retrain_wt_GP = [0.013287422, 0.139606193, 0.579030335]
    epochs = [1, 5, 10]
    pdf = PdfPages('retrain_tenrec_lgcn_mse.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(4, 4))
    ax.plot(np.arange(len(epochs)), np.array(full_retrain), 's--', markersize=3, label='full_retrain', color='red')
    ax.plot(np.arange(len(epochs)), np.array(full_retrain_wt_GP), 'x-', markersize=3, label='full_retrain_wt_GP', color='red')
    ax.plot(np.arange(len(epochs)), np.array(pre_retrain), 'd--', markersize=3, label='pre_retrain', color='blue')
    ax.plot(np.arange(len(epochs)), np.array(pre_retrain_wt_GP), 'v-', markersize=3, label='pre_retrain_wt_GP', color='blue')
    ax.set_xticks(np.arange(len(epochs)))
    ax.set_xticklabels(epochs)
    ax.set_xlabel('Retraining Epoch', fontsize=17)
    ax.set_ylabel('Jaccard Similarity@50', fontsize=17)
    ax.set_title('Tenrec-LightGCN-MSE', fontsize=17)
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

    mean_retrain = [3.749275, 3.814666667, 4.703283333, 4.807458333, 0.88005, 0.921591667]
    mean_retrain_wt_GP = [3.97095, 3.89775, 4.758966667, 4.914191667, 0.916416667, 0.933858333]
    std_retrain = [0.225533219, 0.215850846, 0.062322604, 0.11386256, 0.023414894, 0.024640117]
    std_retrain_wt_GP = [0.288210714, 0.281850055, 0.183178226, 0.084019343, 0.050092348, 0.029010075]
    methods = ['MF-BPR', 'LightGCN-BPR', 'MF-BCE', 'LightGCN-BCE', 'MF-MSE', 'LightGCN-MSE']
    idx = np.arange(len(methods))
    width = 0.2
    pdf = PdfPages('retain_time_gowalla.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(7, 4))
    ax.bar(idx - width, mean_retrain, width, label='retrain', yerr=std_retrain)
    ax.bar(idx + width, mean_retrain_wt_GP, width, label='retrain_wt_GP', yerr=std_retrain_wt_GP)
    ax.set_xticks(idx)
    ax.set_xticklabels(methods, fontsize=13)
    ax.set_xlabel('Recommender Method', fontsize=17)
    ax.set_ylabel('Training Time per Epoch (s)', fontsize=17)
    ax.set_title('Gowalla', fontsize=17)
    ax.legend(fontsize=13, loc=1, frameon=False)
    ax.grid(True, which='major', linestyle=':', linewidth=0.8)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    mean_retrain = [344.9129167, 391.6187333, 385.7989, 393.6217, 231.6462333, 252.93105]
    mean_retrain_wt_GP = [371.1690667, 453.1785167, 408.9144, 472.0270333, 287.9294167, 351.39295]
    std_retrain = [131.5710992, 125.522134, 161.289596, 135.4116029, 79.94709292, 43.35285593]
    std_retrain_wt_GP = [64.70370844, 92.12506417, 64.74408923, 60.74967801, 45.95772361, 62.95571001]
    methods = ['MF-BPR', 'LightGCN-BPR', 'MF-BCE', 'LightGCN-BCE', 'MF-MSE', 'LightGCN-MSE']
    idx = np.arange(len(methods))
    width = 0.2
    pdf = PdfPages('retain_time_tenrec.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(7, 4))
    ax.bar(idx - width, mean_retrain, width, label='retrain', yerr=std_retrain)
    ax.bar(idx + width, mean_retrain_wt_GP, width, label='retrain_wt_GP', yerr=std_retrain_wt_GP)
    ax.set_xticks(idx)
    ax.set_xticklabels(methods, fontsize=13)
    ax.set_xlabel('Recommender Method', fontsize=17)
    ax.set_ylabel('Training Time per Epoch (s)', fontsize=17)
    ax.set_title('Tenrec', fontsize=17)
    ax.legend(fontsize=13, loc=0, frameon=False)
    ax.grid(True, which='major', linestyle=':', linewidth=0.8)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    mean_retraining = [1206.31, 1010.91, 4762.77]
    mean_all = [2129.32, 1019.91, 5886.10]
    std_retraining = [37.00, 253.21, 414.68]
    std_all = [118.65, 253.29, 511.73]
    methods = ['PGA', 'RevAdv', 'DPA2DL']
    idx = np.arange(len(methods))
    width = 0.2
    pdf = PdfPages('attack_time_gowalla.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(7, 4))
    ax.bar(idx - width, mean_all, width, label='all_time', yerr=std_all)
    ax.bar(idx + width, mean_retraining, width, label='retraining_time', yerr=std_retraining)
    ax.set_xticks(idx)
    ax.set_xticklabels(methods, fontsize=19)
    ax.set_xlabel('Attack Method', fontsize=19)
    ax.set_ylabel('Consumed Time (s)', fontsize=19)
    ax.set_title('Gowalla', fontsize=19)
    ax.legend(fontsize=15, loc=0, frameon=False)
    ax.grid(True, which='major', linestyle=':', linewidth=0.8)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    mean_retraining = [5768.52, 4100.99]
    mean_all = [9510.01, 5079.17]
    std_retraining = [456.72, 132.90]
    std_all = [907.22, 163.26]
    methods = ['PGA', 'DPA2DL']
    idx = np.arange(len(methods))
    width = 0.2
    pdf = PdfPages('attack_time_yelp.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(7, 4))
    ax.bar(idx - width, mean_all, width, label='all_time', yerr=std_all)
    ax.bar(idx + width, mean_retraining, width, label='retraining_time', yerr=std_retraining)
    ax.set_xticks(idx)
    ax.set_xticklabels(methods, fontsize=19)
    ax.set_xlabel('Attack Method', fontsize=19)
    ax.set_ylabel('Consumed Time (s)', fontsize=19)
    ax.set_title('Yelp', fontsize=19)
    ax.legend(fontsize=15, loc=0, frameon=False)
    ax.grid(True, which='major', linestyle=':', linewidth=0.8)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    grad_sims = np.load('grad_sim_gowalla.npy')[:200, :]
    epochs = np.arange(grad_sims.shape[0])
    pdf = PdfPages('grad_sim_gowalla.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(7, 4))
    ax.plot(epochs, grad_sims[:, 0], '-', markersize=0, label='connected_nodes', color='red')
    ax.plot(epochs, grad_sims[:, 2], '-', markersize=0, label='random_nodes', color='blue')
    ax.set_xlabel('Training Epoch', fontsize=19)
    ax.set_ylabel('Cosine Similarity', fontsize=19)
    ax.set_ylim(-1, 1)
    ax.set_title('Gowalla', fontsize=19)
    ax.legend(fontsize=15, loc=0, frameon=False)
    ax.grid(True, which='major', linestyle=':', linewidth=0.8)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in')
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.tight_layout()
    pdf.savefig()
    plt.close(fig)
    pdf.close()

    grad_sims = np.load('grad_sim_yelp.npy')[:200, :]
    epochs = np.arange(grad_sims.shape[0])
    pdf = PdfPages('grad_sim_yelp.pdf')
    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(7, 4))
    ax.plot(epochs, grad_sims[:, 0], '-', markersize=0, label='connected_nodes', color='red')
    ax.plot(epochs, grad_sims[:, 2], '-', markersize=0, label='random_nodes', color='blue')
    ax.set_xlabel('Training Epoch', fontsize=19)
    ax.set_ylabel('Cosine Similarity', fontsize=19)
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, 200)
    ax.set_title('Yelp', fontsize=19)
    ax.legend(fontsize=15, loc=0, frameon=False)
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