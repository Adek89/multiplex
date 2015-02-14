__author__ = 'Adek'

import networkx as nx

import graph.method.common.CommonUtils as commons


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

    def classify(self):
        nodes = self.graph.nodes()
        nrOfNodes = nodes.__len__()
        commonUtils = commons.CommonUtils()
        for i in xrange(self.nrOfFolds):  #nrOfFolds times
            items = range(nrOfNodes)
            trainingId, testId = commonUtils.k_fold_cross_validation(items, self.nrOfFolds, self.percentTraining)



