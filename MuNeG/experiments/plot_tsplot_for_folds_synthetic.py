import csv
import os
import pickle
import string
import tokenize as token

import numpy as np
from scipy import interp


def read_function(tokens):
    line = ''
    next_token = tokens.next()[1]
    while next_token <> '\n':
        line = line + next_token
        next_token = tokens.next()[1]
    split_result = string.split(line, ',')
    return split_result[0], split_result[1]

def prepare_file():
    global tokens
    path = '/home/apopiel/functions_copy.csv'
    path = os.path.join(os.path.dirname(__file__), '%s' % path)
    f = open(path)
    tokens = token.generate_tokens(f.readline)
    return tokens


keys = ["reduction", "fusion_sum", "fusion_mean", "fusion_layer", "fusion_random", "fusion_convergence_max", "fusion_convergence_min"]
colors = {"reduction":'cyan',"fusion_sum":'indigo', "fusion_mean":'seagreen', "fusion_layer":'yellow', "fusion_random":'blue', "fusion_convergence_max":'darkorange', "fusion_convergence_min" : "red"}
names = {"reduction":'LR',"fusion_sum":'SF', "fusion_mean":'MF', "fusion_layer":'LF', "fusion_random":'RF', "fusion_convergence_max":'SCF', "fusion_convergence_min" : "FCF"}

for fold in [2.0, 3.0, 4.0, 5.0, 10.0, 20.0]:
    global_fprs = {}
    global_tprs = {}
    for key in keys:
        global_fprs[key] = np.linspace(0, 1, 100)
        global_tprs[key] = 0.0
    sum_of_weights = 0
    tokens = prepare_file()
    for node in [100]:
        for size in [2, 4, 5, 10, 20, 25, 30, 50]:
            for label in [5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]:
                for prob_in in [5, 6, 7, 8, 9]:
                    for prob_out in  [0.1, 0.5, 1, 2, 3, 4, 5]:
                        for layers in [2, 3, 4, 5, 6, 8, 10, 21]:
                            for probe in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
                                print str(sum_of_weights)
                                sum_of_weights += 1
                                mean_fprs = {}
                                mean_tprs = {}
                                fprs_file = open('/lustre/scratch/apopiel/synthetic/means/mean_' + 'fprs' + '_' + str(node) + '_' + str(size) + '_' + str(label) + '_' + str(prob_in) + '_' + str(prob_out) + '_' + str(layers) + '_' + str(fold) + '_' + str(probe) + '.txt', 'rb')
                                mean_fprs = pickle.load(fprs_file)
                                tprs_file = open('/lustre/scratch/apopiel/synthetic/means/mean_' + 'tprs' + '_' + str(node) + '_' + str(size) + '_' + str(label) + '_' + str(prob_in) + '_' + str(prob_out) + '_' + str(layers) + '_' + str(fold) + '_' + str(probe) + '.txt', 'rb')
                                mean_tprs = pickle.load(tprs_file)
                                for key in keys:
                                    global_tprs[key] += interp(global_fprs[key], mean_fprs[key], mean_tprs[key])
                                    global_tprs[key][0] = 0.0
    for key in keys:
        global_tprs[key] /= sum_of_weights
        global_tprs[key][-1] = 1.0
        for elem in xrange(0, len(global_tprs[key])):
            with open('/lustre/scratch/apopiel/synthetic/tsplot/output' + '.csv','ab') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([global_tprs[key][elem], global_fprs[key][elem], fold, key, elem])



