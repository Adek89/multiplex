'''
Created on 18 mar 2014

@author: Adek
'''

import os
import sys
sys.path.append('D:\pycharm_workspace\multiplex\MuNeG')
import time

import networkx as nx

sys.path.append('/home/apopiel/multiplex/MuNeG')
from graph.gen.GraphGenerator import GraphGenerator



def prepare_layers():
    layer_names = []
    layer_weights = []
    for i in xrange(1, nrOfLayers + 1):
        layer_names.append("L" + str(i))
        layer_weights.append(i)
    return layer_names, layer_weights


if __name__ == '__main__':
    start_time = time.time()
    # groups = [100, 500, 1000]
    # groupSize = [50 -> 2, 30 -> 3, 25 -> 4, 20 -> 5, 10 -> 10, 5 -> 20, 4 -> 25, 2 -> 50]
    # groupLabel = [5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
    # probInGroup = [5, 6, 7, 8, 9]
    # probBetweenGroups = [0.1, 0.5, 1, 2, 3, 4, 5]
    # layers = [2 3 4 6 8 10 21]

    nodes = int(sys.argv[1])
    size = int(sys.argv[2])
    for label in [5]:
        for probIn in [5]:
            for probBetween in [1]:
                for nrOfLayers in [1]:

                    layer_names, layer_weights = prepare_layers()

                    gg = GraphGenerator(nodes, size, layer_weights, label, probIn, probBetween, layer_names)
                    graph = gg.generate()
                    path = '..\\results\\muneg_' + str(nodes) + '_' + str(size) + '_' + str(label) + '_' + str(probIn) + '_' + str(probBetween) + '_' + str(nrOfLayers) +'.gml'
                    path = os.path.join(os.path.dirname(__file__), '%s' % path)
                    nx.write_gml(graph, path)

    print("--- %s seconds ---" % str(time.time() - start_time))