'''
Created on 18 mar 2014

@author: Adek
'''
#1 - stratified xval
#2 therefore used xval

import csv
import os
import pickle
import string
import sys
import tokenize as token

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics as metrics
from scipy import interp

sys.path.append('/cygdrive/d/pycharm_workspace/multiplex/MuNeG/')
sys.path.append('D:\pycharm_workspace\multiplex\MuNeG')

from experiments.DecisionFusionRealSW import DecisionFusion
import gc


def execute_experiment(method, folds):
    gc.collect()
    df = DecisionFusion(method, folds)
    return df.processExperiment()

def append_roc_rates_for_average(mean_fprs, mean_tprs, fpr, tpr):
    new_mean_tprs = mean_tprs + interp(mean_fprs, fpr, tpr)
    new_mean_tprs[0] = 0.0
    return new_mean_tprs

def plot_roc_curve(fpr, tpr, roc_auc, method, color):
    lw = 2
    line = plt.plot(fpr, tpr, color=color, lw=lw, label='%s curve (area = %0.2f)' % (method, roc_auc))
    plt.setp(line, linewidth=6)

def plot(figure, type_of_exp, probe, file_name = 'global'):
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
    figure.savefig('..\\results\\real_sw\\roc2\\' + file_name + '_' + type_of_exp + '_' + str(probe) + '.png', dpi=300)
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
    path = '..\\dataset\\DanioRerio\\functions.csv'
    path = os.path.join(os.path.dirname(__file__), '%s' % path)
    f = open(path)
    tokens = token.generate_tokens(f.readline)
    return tokens


def save_mean_rates(means, rate_type, type_of_exp, probe, fold=None):
    target_file = open('..\\results\\real_sw\\roc2\\mean_' + rate_type + '_' + (str('_'+str(fold)) if fold <> None else '') + '_' + type_of_exp + '_' + str(probe) + '.txt', 'ab')
    pickle.dump(means, target_file)
    target_file.close()

def save_aoc_results(probe, aoc, key, type_of_exp, fold ='', file_name = 'global'):
    with open('..\\results\\real_sw\\aoc\\real_aoc_' + file_name + '_' + type_of_exp + '_' + str(probe) + '.csv','ab') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([key, fold, aoc, probe])

def calculate_layer_results(fold, type_of_exp, probe, fprs_per_method, tprs_per_method, file_name='global'):
    auc_for_layer = {}
    for l in xrange(1, 7):
        key_for_layer = 'L' + str(l)
        roc_auc = metrics.auc(fprs_per_method[key_for_layer], tprs_per_method[key_for_layer])
        save_aoc_results(probe, roc_auc, key_for_layer, type_of_exp, fold=fold, file_name=file_name)
        auc_for_layer.update({l: roc_auc})
    layer_results = auc_for_layer.values()
    min_layer_result = min(layer_results)
    max_layer_result = max(layer_results)
    avg_layer_result = sum(layer_results) / float(len(layer_results))
    save_aoc_results(probe, max_layer_result, "max_layer", type_of_exp, fold=fold, file_name=file_name)
    save_aoc_results(probe, min_layer_result, "min_layer", type_of_exp, fold=fold, file_name=file_name)
    save_aoc_results(probe, avg_layer_result, "avg_layer", type_of_exp, fold=fold, file_name=file_name)


def main(probe, type_of_exp):
        keys = ["reduction", "fusion_sum", "fusion_mean", "fusion_layer", "fusion_random", "fusion_convergence_max", "fusion_convergence_min"]
        colors = {"reduction":'cyan',"fusion_sum":'indigo', "fusion_mean":'seagreen', "fusion_layer":'yellow', "fusion_random":'blue', "fusion_convergence_max":'darkorange', "fusion_convergence_min" : "red"}
        names = {"reduction":'LR',"fusion_sum":'SF', "fusion_mean":'MF', "fusion_layer":'LF', "fusion_random":'RF', "fusion_convergence_max":'SCF', "fusion_convergence_min" : "FCF"}
        global_fprs = {}
        global_tprs = {}
        sum_of_weights = 0
        for key in keys:
            global_fprs[key] = np.linspace(0, 1, 100)
            global_tprs[key] = 0.0
        tokens = prepare_file()
        execute = True
        nrOfLayers = 6
        while execute:
            mean_fprs = {}
            mean_tprs = {}
            sns.set_style("darkgrid")
            figure = plt.figure()
            for key in keys:
                mean_fprs[key] = np.linspace(0, 1, 100)
                mean_tprs[key] = 0.0
            for key in xrange(1, nrOfLayers+1):
                key_for_layer = 'L' + str(key)
                mean_fprs[key_for_layer] = np.linspace(0, 1, 100)
                mean_tprs[key_for_layer] = 0.0
            for fold in [2.0, 3.0, 4.0, 5.0, 10.0, 20.0]:
                fold_figure = plt.figure()
                fprs_per_method = {}
                tprs_per_method = {}
                fprs_per_method, tprs_per_method = execute_experiment(1, fold)

                for key in keys:
                    roc_auc = metrics.auc(fprs_per_method[key], tprs_per_method[key])
                    plot_roc_curve(fprs_per_method[key], tprs_per_method[key], roc_auc, names[key], colors[key])
                    save_aoc_results(probe, roc_auc, key, type_of_exp, fold=fold, file_name='fold_'+str(fold))

                calculate_layer_results(fold, type_of_exp, probe, fprs_per_method, tprs_per_method, file_name='fold_'+str(fold))

                plot(fold_figure, type_of_exp, probe, file_name='fold_'+str(fold))
                save_mean_rates(fprs_per_method, 'fprs', type_of_exp, probe, fold=fold)
                save_mean_rates(tprs_per_method, 'tprs', type_of_exp, probe, fold=fold)
                for key in keys:
                    mean_tprs[key] = append_roc_rates_for_average(mean_fprs[key], mean_tprs[key], fprs_per_method[key], tprs_per_method[key])
                for key in xrange(1, nrOfLayers+1):
                    key_for_layer = 'L' + str(key)
                    mean_tprs[key_for_layer] = append_roc_rates_for_average(mean_fprs[key_for_layer], mean_tprs[key_for_layer], fprs_per_method[key_for_layer], tprs_per_method[key_for_layer])
            for key in keys:
                mean_tprs[key] /= 6
                mean_tprs[key][-1] = 1.0
                roc_auc = metrics.auc(mean_fprs[key], mean_tprs[key])
                plot_roc_curve(mean_fprs[key], mean_tprs[key], roc_auc, names[key], colors[key])
                save_aoc_results(probe, roc_auc, key, type_of_exp)
            auc_for_layer = {}
            for key in xrange(1, nrOfLayers+1):
                key_for_layer = 'L' + str(key)
                mean_tprs[key_for_layer] /= 6
                mean_tprs[key_for_layer][-1] = 1.0
                roc_auc = metrics.auc(mean_fprs[key_for_layer], mean_tprs[key_for_layer])
                save_aoc_results(probe, roc_auc, key_for_layer, type_of_exp)
                auc_for_layer.update({key: roc_auc})
            layer_results = auc_for_layer.values()
            min_layer_result = min(layer_results)
            max_layer_result = max(layer_results)
            avg_layer_result = sum(layer_results) / float(len(layer_results))
            save_aoc_results(probe, max_layer_result, "max_layer", type_of_exp)
            save_aoc_results(probe, min_layer_result, "min_layer", type_of_exp)
            save_aoc_results(probe, avg_layer_result, "avg_layer", type_of_exp)
            plot(figure, type_of_exp, probe)
            save_mean_rates(mean_fprs, 'fprs', type_of_exp, probe)
            save_mean_rates(mean_tprs, 'tprs', type_of_exp, probe)
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


if __name__ == '__main__':
    for i in xrange(1, 21):
        main(i, 'anakin_dark')


