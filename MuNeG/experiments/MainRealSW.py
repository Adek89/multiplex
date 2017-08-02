'''
Created on 18 mar 2014

@author: Adek
'''
#1 - stratified xval
#2 therefore used xval

import os
import pickle
import string
import sys
import tokenize as token

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
from scipy import interp

sys.path.append('/cygdrive/d/pycharm_workspace/multiplex/MuNeG/')
sys.path.append('D:\pycharm_workspace\multiplex\MuNeG')

from experiments.DecisionFusionReal import DecisionFusion
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
    plt.plot(fpr, tpr, color=color,
    lw=lw, label='ROC %s curve (area = %0.2f)' % (method, roc_auc))

def plot(figure, qty=0):
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    # plt.draw()
    file_name = 'global'
    figure.set_size_inches(19, 12)
    figure.savefig('..\\results\\real_sw\\roc2\\' + file_name + '.png', dpi=300)
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


def save_mean_rates(means, rate_type, fold=None):
    target_file = open('..\\results\\real_sw\\roc2\\mean_' + rate_type + '_' + (str('_'+str(fold)) if fold <> None else '') + '.txt', 'ab')
    pickle.dump(means, target_file)
    target_file.close()


if __name__ == "__main__":
        keys = ["reduction", "fusion_sum", "fusion_mean", "fusion_layer", "fusion_random", "fusion_convergence_max", "fusion_convergence_min"]
        colors = {"reduction":'cyan',"fusion_sum":'indigo', "fusion_mean":'seagreen', "fusion_layer":'yellow', "fusion_random":'blue', "fusion_convergence_max":'darkorange', "fusion_convergence_min" : "red"}
        global_fprs = {}
        global_tprs = {}
        sum_of_weights = 0
        for key in keys:
            global_fprs[key] = np.linspace(0, 1, 100)
            global_tprs[key] = 0.0
        tokens = prepare_file()
        execute = True
        while execute:
            mean_fprs = {}
            mean_tprs = {}
            figure = plt.figure()
            for key in keys:
                mean_fprs[key] = np.linspace(0, 1, 100)
                mean_tprs[key] = 0.0
            for fold in [2.0, 3.0, 4.0, 5.0, 10.0, 20.0]:
                fprs_per_method = {}
                tprs_per_method = {}
                fprs_per_method, tprs_per_method = execute_experiment(1, fold)

                save_mean_rates(fprs_per_method, 'fprs', fold=fold)
                save_mean_rates(tprs_per_method, 'tprs', fold=fold)
                for key in keys:
                    mean_tprs[key] = append_roc_rates_for_average(mean_fprs[key], mean_tprs[key], fprs_per_method[key], tprs_per_method[key])
            for key in keys:
                mean_tprs[key] /= 6
                mean_tprs[key][-1] = 1.0
                roc_auc = metrics.auc(mean_fprs[key], mean_tprs[key])
                plot_roc_curve(mean_fprs[key], mean_tprs[key], roc_auc, key, colors[key])
            plot(figure)
            save_mean_rates(mean_fprs, 'fprs')
            save_mean_rates(mean_tprs, 'tprs')
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





