__author__ = 'Adek'

import numpy as np
import networkx as nx
import random
class ICA:

    #global variables
    graph = nx.MultiGraph()
    trainingNodes = None
    testNodes = None
    nrOfNodes = 0
    classifier = None
    y = []

    NR_OF_ITERATIONS = 100

    #Constructor
    #Params:
    #   graph - Mutligraph - generated or real - TEST data
    #   classifier - LEARNT classifier
    def __init__(self, graph, trainingNodes, testNodes, classifier):
         self.graph = graph
         self.trainingNodes = trainingNodes
         self.testNodes = testNodes
         self.nrOfNodes = self.graph.nodes().__len__()
         self.classifier = classifier
         self.y = []

    #Start of algorithm
    def execute(self):
        #1. Bootstrapping
        self.bootstrapping()
        #2. Iterative classification
        self.classifyIteratively()

    def classify(self, neighbours, testNodeIndex):
        x = [0, 0]
        for neighbour in neighbours:
            if (neighbour.label == 0):
                x[0] = x[0] + 1
            else:
                x[1] = x[1] + 1
        self.y.append(self.classifier.predict(x).item())

    def classifyIteratively(self):
        nrOfTestNodes = range(self.testNodes.__len__())
        for i in range(0, self.NR_OF_ITERATIONS):
            random.shuffle(nrOfTestNodes)
            for j in nrOfTestNodes:
                currentNode = self.testNodes[j]
                neighbourhoodAll = self.graph.neighbors(currentNode)
                knownNodes = self.extractKnownNodes(neighbourhoodAll)

    def bootstrapping(self):
        testNodeIndex = 0
        for testNode in np.nditer(self.testNodes, ["refs_ok"]):
            nxTestNode = testNode.item()
            neighbourhoodAll = self.graph.neighbors(nxTestNode)
            knownNodes = self.extractKnownNodes(neighbourhoodAll)
            self.classify(knownNodes, testNodeIndex)

    def extractKnownNodes(self, neighbourhoodAll):
        neighbourArray = np.asanyarray(neighbourhoodAll)
        knownNodes = np.array(filter(lambda node: node in self.trainingNodes, neighbourArray))
        return knownNodes


