import sys
sys.path.append('D:\pycharm_workspace\multiplex\MuNeG')
import pickle
from scipy import interp
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import tokenize as token
import string
import os
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
    figure.savefig('..\\results\\real\\roc2\\' + file_name + '.png', dpi=300)
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
check = 0
while True:
    mean_fprs = {}
    mean_tprs = {}
    try:
        function, weight = read_function(tokens)
        weight = int(weight)
    except:
        break
    if weight >= 30:
        sum_of_weights += weight
        function_for_file_name = function.replace(':', '')
        fprs_file = open('..\\results\\real\\roc2\\mean_' + 'fprs' + '_' + function_for_file_name + '_' + str(weight) + '.txt', 'rb')
        mean_fprs = pickle.load(fprs_file)
        tprs_file = open('..\\results\\real\\roc2\\mean_' + 'tprs' + '_' + function_for_file_name + '_' + str(weight) + '.txt', 'rb')
        mean_tprs = pickle.load(tprs_file)
        for key in keys:
            global_tprs[key] += interp(global_fprs[key]*weight, mean_fprs[key]*weight, mean_tprs[key]*weight)
            global_tprs[key][0] = 0.0
        check += 1
        print function
print 'Check nr of functions: ' + str(check)
sns.set_style("darkgrid")
figure = plt.figure()
for key in keys:
    global_tprs[key] /= sum_of_weights
    global_tprs[key][-1] = 1.0
    roc_auc = metrics.auc(global_fprs[key], global_tprs[key])
    plot_roc_curve(global_fprs[key], global_tprs[key], roc_auc, names[key], colors[key])
plot(figure, function="")