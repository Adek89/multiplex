__author__ = 'Adek'

import networkx as nx
import numpy as np
import graph.method.common.CommonUtils as commons
from graph.method.ica.ICA import ICA

class SingleModelICA:
    graph = nx.MultiGraph()
    percentTraining = 0.0
    nrOfFolds = 0
    classifier = None

    # classifier - to learn
    def __init__(self, graph, percentTraining, nrOfFolds, classifier):
        self.graph = graph
        self.percentTraining = percentTraining
        self.nrOfFolds = nrOfFolds
        self.classifier = classifier

    def prepareClassifierInputs(self, node):
        nxNode = node.item()
        neigbourhood = self.graph.neighbors(nxNode)
        y = nxNode.label
        x = [0, 0]
        for neighbour in neigbourhood:
            if (neighbour.label == 0):
                x[0] = x[0] + 1
            else:
                x[1] = x[1] + 1
        return x, y

    def trainClassifier(self, nodesArray, training):
        trainingNodes = nodesArray[training]
        x = []
        y = []
        for node in np.nditer(trainingNodes, ["refs_ok"]):
            nodeX, nodeY = self.prepareClassifierInputs(node)
            x.append(nodeX)
            y.append(nodeY)
        self.classifier.fit(x, y)
        return trainingNodes

    def executeICA(self, testNodes, trainingNodes):
        ica = ICA(self.graph, trainingNodes, testNodes, self.classifier)
        ica.execute()

    def crossValidation(self, commonUtils, items, nodesArray):
        for training, validation in commonUtils.k_fold_cross_validation(items, self.nrOfFolds, self.percentTraining):
            trainingNodes = self.trainClassifier(nodesArray, training)
            testNodes = nodesArray[validation]
            self.executeICA(testNodes, trainingNodes)

    def classify(self):
        nodes = self.graph.nodes()
        sortedNodes = sorted(nodes, key=lambda node: node.id)
        nodesArray = np.asanyarray(sortedNodes).transpose()
        nrOfNodes = nodes.__len__()
        commonUtils = commons.CommonUtils()
        items = range(nrOfNodes)
        self.crossValidation(commonUtils, items, nodesArray)




