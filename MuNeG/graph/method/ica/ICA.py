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
    y = dict([])

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
         self.y = dict([])

    #Start of algorithm
    def execute(self):
        self.bootstrapping()
        self.classifyIteratively()
        return self.y

    def updateInputVector(self, label, x):
        if (label == 0):
            x[0] = x[0] + 1
        else:
            x[1] = x[1] + 1

    def classify(self, knownNeighbours, tempLabels, testNode):
        x = [0, 0]
        for neighbour in knownNeighbours:
            self.updateInputVector(neighbour.label, x)
        for label in tempLabels:
            self.updateInputVector(label, x)
        prediction = self.classifier.predict(x).item()
        self.y.update({testNode : prediction})

    def extractTempLabels(self, tempLabels, unknownList):
        for unknownNode in unknownList:
            currentPrediction = self.y.get(unknownNode)
            tempLabels.append(currentPrediction)

    def performIteration(self, nrOfTestNodes):
        for j in nrOfTestNodes:
            currentNode = self.testNodes[j]
            neighbourhoodAll = self.graph.neighbors(currentNode)
            knownNodes = self.extractKnownNodes(neighbourhoodAll)
            knownNodesList = knownNodes.tolist()
            unknownList = list(set(neighbourhoodAll) - set(knownNodesList))
            tempLabels = []
            self.extractTempLabels(tempLabels, unknownList)
            self.classify(knownNodesList, tempLabels, currentNode)

    def classifyIteratively(self):
        nrOfTestNodes = range(self.testNodes.__len__())
        for i in range(0, self.NR_OF_ITERATIONS):
            random.shuffle(nrOfTestNodes)
            self.performIteration(nrOfTestNodes)

    def bootstrapping(self):
        for testNode in np.nditer(self.testNodes, ["refs_ok"]):
            nxTestNode = testNode.item()
            neighbourhoodAll = self.graph.neighbors(nxTestNode)
            knownNodes = self.extractKnownNodes(neighbourhoodAll)
            self.classify(knownNodes, [], nxTestNode)

    def extractKnownNodes(self, neighbourhoodAll):
        neighbourArray = np.asanyarray(neighbourhoodAll)
        knownNodes = np.array(filter(lambda node: node in self.trainingNodes, neighbourArray))
        return knownNodes


