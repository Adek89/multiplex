__author__ = 'Adek'

import numpy as np
import networkx as nx
class EnsambleLearning:

    graph = nx.MultiGraph()
    nrOfModels = 0
    ensambleSet = set([])
    nrOfNodesInSubgraph = 0

    def __init__(self, graph, nrOfModels, nrOfNodesInSubgraph):
        self.graph = graph
        self.nrOfModels = nrOfModels
        self.ensambleSet = set([])
        self.nrOfNodesInSubgraph = nrOfNodesInSubgraph

    def ensamble(self):
        for i in range(0, self.nrOfModels):
            sampledGraph = self.sampleGraph()
            model = self.learnModel()
            self.ensambleSet.add(model)
        return self.ensambleSet

    def sampleGraph(self):
        nrOfNodes = self.graph.nodes().__len__()

    def learnModel(self):
        pass
