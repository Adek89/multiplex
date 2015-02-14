__author__ = 'Adek'

import numpy as np
import networkx as nx
class ICA:

    #global variables
    graph = nx.MultiGraph()
    nrOfNodes = 0
    classifier = None

    #Constructor
    #Params:
    #   graph - Mutligraph - generated or real - TEST data
    #   classifier - LEARNT classifier
    def __init__(self, graph, classifier):
         self.graph = graph
         self.nrOfNodes = self.graph.nodes().__len__()
         self.classifier = classifier

    #Start of algorithm
    def execute(self):
        #1. Bootstrapping
        self.bootstrapping()
        #2. Iterative classification
        #3. Results

    def bootstrapping(self):
        pass




