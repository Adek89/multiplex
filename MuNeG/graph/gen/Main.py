'''
Created on 6 lut 2014

@author: Adek
'''
from graph.gen.GraphGenerator import GraphGenerator
'''import matplotlib.pyplot as plt'''

NUMBER_OF_NODES = 100
AVERAGE_GROUP_SIZE = 20

LAYERS_WEIGHTS = [0.1, 0.2, 0.5, 0.75, 1]

PROBABILITY_OF_EDGE_EXISTANCE_IN_SAME_GROUP = 0.6
PROBABILITY_OF_EDGE_EXISTANCE_BETWEEN_OTHER_GROUPS = 0.004


if __name__ == '__main__':
    gg = GraphGenerator(NUMBER_OF_NODES, AVERAGE_GROUP_SIZE, LAYERS_WEIGHTS, PROBABILITY_OF_EDGE_EXISTANCE_IN_SAME_GROUP, PROBABILITY_OF_EDGE_EXISTANCE_BETWEEN_OTHER_GROUPS)
    for i in range(0, 10):
        gg.generate(i)
        