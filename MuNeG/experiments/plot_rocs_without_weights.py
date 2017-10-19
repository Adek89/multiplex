import sys

import matplotlib

matplotlib.use('Agg')
sys.path.append('/home/apopiel/multiplex/MuNeG/')
import pickle
from scipy import interp
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import string
import numpy as np
import seaborn as sns

def plot_roc_curve(fpr, tpr, roc_auc, method, color):
    lw = 2
    line = plt.plot(fpr, tpr, color=color, lw=lw, label='%s curve (area = %0.2f)' % (method, roc_auc))
    plt.setp(line, linewidth=6)


def plot(figure, function="", qty=0):

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
    # plt.draw()
    file_name = function.replace(":", "") if function <> "" else "global"
    if file_name <> "global":
        file_name += "_" + str(qty)
    figure.set_size_inches(19, 12)
    figure.savefig('/lustre/scratch/apopiel/synthetic/roc2/global/' + file_name + '.png', dpi=300)
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


keys = ["reduction", "fusion_sum", "fusion_mean", "fusion_layer", "fusion_random", "fusion_convergence_max", "fusion_convergence_min"]
colors = {"reduction":'cyan',"fusion_sum":'indigo', "fusion_mean":'seagreen', "fusion_layer":'yellow', "fusion_random":'blue', "fusion_convergence_max":'darkorange', "fusion_convergence_min" : "red"}
names = {"reduction":'LR',"fusion_sum":'SF', "fusion_mean":'MF', "fusion_layer":'LF', "fusion_random":'RF', "fusion_convergence_max":'SCF', "fusion_convergence_min" : "FCF"}
global_fprs = {}
global_tprs = {}
i = 0.0
for key in keys:
    global_fprs[key] = np.linspace(0, 1, 100)
    global_tprs[key] = 0.0
nodes = 100
for size in [2, 4, 5, 10, 20, 25, 30, 50]:
    for label in [5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]:
        for probIn in [5, 6, 7, 8, 9]:
            for probBetween in [0.1, 0.5, 1, 2, 3, 4, 5]:
                for nrOfLayers in [2, 3, 4, 5, 6, 8, 10, 21]:
                    execute = True
                    mean_fprs = {}
                    mean_tprs = {}
                    fprs_file = open('/lustre/scratch/apopiel/synthetic/means/mean_' + 'fprs' + '_' + str(nodes) + '_' + str(size) + '_' + str(label) + '_' + str(probIn) + '_' + str(probBetween) + '_' + str(nrOfLayers) + '_11.txt', 'rb')
                    mean_fprs = pickle.load(fprs_file)
                    tprs_file = open('/lustre/scratch/apopiel/synthetic/means/mean_' + 'tprs'+ '_' + str(nodes) + '_' + str(size) + '_' + str(label) + '_' + str(probIn) + '_' + str(probBetween) + '_' + str(nrOfLayers) + '_11.txt', 'rb')
                    mean_tprs = pickle.load(tprs_file)
                    for key in keys:
                        global_tprs[key] += interp(global_fprs[key], mean_fprs[key], mean_tprs[key])
                        global_tprs[key][0] = 0.0
                    i = i + 1
                    print i
sns.set_style("darkgrid")
figure = plt.figure()
for key in keys:
    global_tprs[key] /= i
    global_tprs[key][-1] = 1.0
    roc_auc = metrics.auc(global_fprs[key], global_tprs[key])
    plot_roc_curve(global_fprs[key], global_tprs[key], roc_auc, names[key], colors[key])
plot(figure, function="")
pass