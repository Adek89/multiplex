
__author__ = 'Adek'
import time
import sys
sys.path.append('/home/apopiel/MuNeG')
from bin.graph.analyser import GraphAnalyser

if __name__ == '__main__':
    start_time = time.time()
    nodes = int(sys.argv[1])
    size = int(sys.argv[2])
    label = float(sys.argv[3])
    probIn = int(sys.argv[4])
    probBetween = float(sys.argv[5])
    nrOfLayers = int(sys.argv[6])
    percentOfTrainingNodes = float(sys.argv[7])
    counter = float(sys.argv[8])

    ga = GraphAnalyser(nodes, size, label, probIn, probBetween, nrOfLayers, percentOfTrainingNodes, counter)
    ga.analyse()
