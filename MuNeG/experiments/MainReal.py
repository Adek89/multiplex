'''
Created on 18 mar 2014

@author: Adek
'''
import time
import sys
sys.path.append('/home/apopiel/MuNeG')
from experiments.DecisionFusionReal import DecisionFusion
if __name__ == '__main__':

    percentOfTrainingNodes = float(sys.argv[1])
    counter = float(sys.argv[2])
    sampleNodes = float(sys.argv[3])
    limit = int(sys.argv[4])
    path = sys.argv[5]


    df = DecisionFusion(percentOfTrainingNodes, counter, sampleNodes, limit, path)
    df.processExperiment()
    print("--- %s seconds ---" % str(time.time() - start_time))