'''
Created on 18 mar 2014

@author: Adek
'''
#1 - stratified xval
#2 therefore used xval

import matplotlib
matplotlib.use('Agg')
import os
import pickle
import string
import sys
import tokenize as token

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
from scipy import interp
import seaborn as sns

sys.path.append('/home/apopiel/multiplex/MuNeG')

from experiments.DecisionFusion import DecisionFusion
import gc
import csv


def execute_experiment(nodes, size, label, probIn, probBetween, nrOfLayers, method, folds):
    gc.collect()
    df = DecisionFusion(nodes, size, label, probIn, probBetween, nrOfLayers, fold)
    return df.processExperiment()

def append_roc_rates_for_average(mean_fprs, mean_tprs, fpr, tpr):
    new_mean_tprs = mean_tprs + interp(mean_fprs, fpr, tpr)
    new_mean_tprs[0] = 0.0
    return new_mean_tprs

def plot_roc_curve(fpr, tpr, roc_auc, method, color):
    lw = 2
    line = plt.plot(fpr, tpr, color=color, lw=lw, label='%s curve (area = %0.2f)' % (method, roc_auc))
    plt.setp(line, linewidth=6)

def plot(figure, nodes, size, label, probIn, probBetween, nrOfLayers, probe, qty=0, file_name = 'global'):
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize="20")
    plt.ylabel('True Positive Rate', fontsize="20")
    plt.xticks(fontsize="20")
    plt.yticks(fontsize="20")
    plt.title('Receiver operating characteristic', fontsize="20")
    legend = plt.legend(loc="lower right", fontsize="17", frameon=True)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    figure.set_size_inches(19, 12)
    figure.savefig('D:\\pycharm_workspace\\multiplex\\MuNeG\\results\\synthetic\\roc2\\' + file_name + '_' + str(nodes) + '_' + str(size) + '_' + str(label) + '_' + str(probIn) + '_' + str(probBetween) + '_' + str(nrOfLayers) + '_' + str(probe) +'.png', dpi=100)
    plt.close(figure)


def read_function(tokens):
    next_token = tokens.next()[1]
    while next_token <> '\n':
        go_term = next_token
        next_token = tokens.next()[1]
        while next_token <> 'GO' and next_token <> '\n':
            go_term = go_term + next_token
            next_token = tokens.next()[1]
    split_result = string.split(go_term, ',')
    return split_result[0], split_result[1]


def prepare_file():
    global tokens
    path = 'D:\\pycharm_workspace\\multiplex\\MuNeG\\DanioRerio\\functions.csv'
    path = os.path.join(os.path.dirname(__file__), '%s' % path)
    f = open(path)
    tokens = token.generate_tokens(f.readline)
    return tokens


def save_mean_rates(means, rate_type, nodes, size, label, probIn, probBetween, nrOfLayers, probe, fold=None):
    target_file = open('D:\\pycharm_workspace\\multiplex\\MuNeG\\results\\synthetic\\means\\mean_' + rate_type + '_' + str(nodes) + '_' + str(size) + '_' + str(label) + '_' + str(probIn) + '_' + str(probBetween) + '_' + str(nrOfLayers) + (str('_'+str(fold)) if fold <> None else '') + '_' + str(probe) + '.txt', 'ab')
    pickle.dump(means, target_file)
    target_file.close()

def save_aoc_results(nodes, size, label, probIn, probBetween, nrOfLayers, fold, key, aoc, probe):
    with open('D:\\pycharm_workspace\\multiplex\\MuNeG\\results\\synthetic\\aoc\\synt_aoc_' + str(nodes) + '_' + str(size) + '_' + str(label) + '_' + str(probIn) + '_' + str(probBetween) + '_' + str(nrOfLayers) + '_' + (str(fold) if fold <> '' else 'global') + '_' + str(probe) + '.csv','ab') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([nodes, size, label, probIn, probBetween, nrOfLayers, fold, key, aoc, probe])



if __name__ == "__main__":
        keys = ["reduction", "fusion_sum", "fusion_mean", "fusion_layer", "fusion_random", "fusion_convergence_max", "fusion_convergence_min"]
        colors = {"reduction":'cyan',"fusion_sum":'indigo', "fusion_mean":'seagreen', "fusion_layer":'yellow', "fusion_random":'blue', "fusion_convergence_max":'darkorange', "fusion_convergence_min" : "red"}
        names = {"reduction":'LR',"fusion_sum":'SF', "fusion_mean":'MF', "fusion_layer":'LF', "fusion_random":'RF', "fusion_convergence_max":'SCF', "fusion_convergence_min" : "FCF"}
        global_fprs = {}
        global_tprs = {}
        sum_of_weights = 0
        sns.set_style("darkgrid")
        for key in keys:
            global_fprs[key] = np.linspace(0, 1, 100)
            global_tprs[key] = 0.0

        nodes = int(sys.argv[1])
        size = int(sys.argv[2])
        probe = int(sys.argv[3])
        for label in [10]:
            for probIn in [6]:
                for probBetween in [0.5]:
                    for nrOfLayers in [21]:
                        execute = True

                        while execute:
                            mean_fprs = {}
                            mean_tprs = {}
                            figure = plt.figure()
                            for key in keys:
                                mean_fprs[key] = np.linspace(0, 1, 100)
                                mean_tprs[key] = 0.0
                            for fold in [2.0, 3.0, 4.0, 5.0, 10.0, 20.0]:
                                fold_figure = plt.figure()
                                fprs_per_method = {}
                                tprs_per_method = {}
                                fprs_per_method, tprs_per_method = execute_experiment(nodes, size, label, probIn, probBetween, nrOfLayers, 1, fold)
                                for key in keys:
                                    roc_auc = metrics.auc(fprs_per_method[key], tprs_per_method[key])
                                    plot_roc_curve(fprs_per_method[key], tprs_per_method[key], roc_auc, names[key], colors[key])
                                    save_aoc_results(nodes, size, label, probIn, probBetween, nrOfLayers, fold, key, roc_auc, probe)
                                plot(fold_figure, nodes, size, label, probIn, probBetween, nrOfLayers, probe, file_name='fold_'+str(fold))
                                save_mean_rates(fprs_per_method, 'fprs', nodes, size, label, probIn, probBetween, nrOfLayers, probe, fold=fold)
                                save_mean_rates(tprs_per_method, 'tprs', nodes, size, label, probIn, probBetween, nrOfLayers, probe, fold=fold)
                                for key in keys:
                                    mean_tprs[key] = append_roc_rates_for_average(mean_fprs[key], mean_tprs[key], fprs_per_method[key], tprs_per_method[key])
                            for key in keys:
                                mean_tprs[key] /= 6
                                mean_tprs[key][-1] = 1.0
                                roc_auc = metrics.auc(mean_fprs[key], mean_tprs[key])
                                plot_roc_curve(mean_fprs[key], mean_tprs[key], roc_auc, names[key], colors[key])
                                save_aoc_results(nodes, size, label, probIn, probBetween, nrOfLayers, '', key, roc_auc, probe)
                            plot(figure, nodes, size, label, probIn, probBetween, nrOfLayers, probe)
                            save_mean_rates(mean_fprs, 'fprs', nodes, size, label, probIn, probBetween, nrOfLayers, probe)
                            save_mean_rates(mean_tprs, 'tprs', nodes, size, label, probIn, probBetween, nrOfLayers, probe)
                            execute = False
            # for key in keys:
            #     global_tprs[key] += interp(global_fprs[key]*weight, mean_fprs[key]*weight, mean_tprs[key]*weight)
            #     global_tprs[key][0] = 0.0
        # figure = plt.figure()
        # for key in keys:
        #     global_tprs[key] /= sum_of_weights
        #     global_tprs[key][-1] = 1.0
        #     roc_auc = metrics.auc(global_fprs[key], global_tprs[key])
        #     plot_roc_curve(global_fprs[key], global_tprs[key], roc_auc, key, colors[key])
        # plot(figure, function="")





