import csv
import os
import pickle
import string
import sys
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
WEIGHT__THRESHOLD = int(sys.argv[1])
for fold in [2.0, 3.0, 4.0, 5.0, 10.0, 20.0]:
    int_fun = 0
    global_fprs = {}
    global_tprs = {}
    for key in keys:
        global_fprs[key] = np.linspace(0, 1, 100)
        global_tprs[key] = 0.0
    sum_of_weights = 0
    tokens = prepare_file()
    while True:
        try:
            function, weight = read_function(tokens)
            weight = int(weight)
            int_fun = int_fun + 1
        except :
            break
        if weight >= WEIGHT__THRESHOLD:
            for probe in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                print str(int_fun)
                sum_of_weights += weight
                function_for_file_name = function.replace(':', '')
                mean_fprs = {}
                mean_tprs = {}
                fprs_file = open('/lustre/scratch/apopiel/real/means/mean_' + 'fprs' + '_' + function_for_file_name + '_' + str(weight) + '_' + str(fold) + '_' + str(probe) + '.txt', 'rb')
                mean_fprs = pickle.load(fprs_file)
                tprs_file = open('/lustre/scratch/apopiel/real/means/mean_' + 'tprs' + '_' + function_for_file_name + '_' + str(weight) + '_' + str(fold) + '_' + str(probe) + '.txt', 'rb')
                mean_tprs = pickle.load(tprs_file)
                for key in keys:
                    global_tprs[key] += interp(global_fprs[key]*weight, mean_fprs[key]*weight, mean_tprs[key]*weight)
                    global_tprs[key][0] = 0.0
    for key in keys:
        global_tprs[key] /= sum_of_weights
        global_tprs[key][-1] = 1.0
        for elem in xrange(0, len(global_tprs[key])):
            with open('/lustre/scratch/apopiel/real/tsplot/output_' + str(WEIGHT__THRESHOLD) + '.csv','ab') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([global_tprs[key][elem], global_fprs[key][elem], fold, key, elem])



