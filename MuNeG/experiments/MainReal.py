'''
Created on 18 mar 2014

@author: Adek
'''
#1 - stratified xval
#2 therefore used xval

import sys
sys.path.append('/cygdrive/d/pycharm_workspace/multiplex/MuNeG/')
from experiments.DecisionFusionReal import DecisionFusion


def execute_experiment(fun, method, folds):
    global df
    df = DecisionFusion(method, folds, fun)
    df.processExperiment()


if __name__ == '__main__':

        for fold in [2.0, 3.0, 4.0, 5.0, 10.0, 20.0]:
            execute_experiment(sys.argv[1], 1, fold)


