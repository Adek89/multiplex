'''
Created on 18 mar 2014

@author: Adek
'''
import time
import sys
sys.path.append('/home/apopiel/MuNeG')
from experiments.EnsableMethods import EnsambleMethods
if __name__ == '__main__':
    start_time = time.time()

    nodes = int(sys.argv[1])
    size = int(sys.argv[2])
    label = float(sys.argv[3])
    probIn = int(sys.argv[4])
    probBetween = float(sys.argv[5])
    nrOfLayers = int(sys.argv[6])
    percentOfTrainingNodes = float(sys.argv[7])
    nrOfNodesInSubgraph = float(sys.argv[8])
    counter = int(sys.argv[9])


    em = EnsambleMethods(nodes, size, label, probIn, probBetween, nrOfLayers, percentOfTrainingNodes, nrOfNodesInSubgraph, counter)
    em.processExperiments()
    print("--- %s seconds -we--" % str(time.time() - start_time))