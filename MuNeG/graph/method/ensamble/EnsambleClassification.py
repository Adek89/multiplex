__author__ = 'Adek'

import networkx as nx
import graph.method.common.CommonUtils as commons
import random
class EnsambleClassification:

    models = set([])
    graph = nx.MultiGraph()
    percentTraining = 0.0
    nodesLabels = dict([]) # V
    modelNodeLabels = dict([]) # Y^
    labelingSets = dict([])#Y^iT

    def __init__(self, models, graph, percentTraining):
        self.models = models
        self.graph = graph
        self.percentTraining = percentTraining
        for node in graph:
            self.nodesLabels.update({node.id : node.label})

    def classify(self):
        labelsList = range(2)
        items = range(self.graph.nodes().__len__())
        commonUtils = commons.CommonUtils()
        for training, validation in commonUtils.k_fold_cross_validation(items, 6, self.percentTraining):
            for i in range(self.models.__len__()):
                labels = dict([])
                for node in self.graph:
                    if node.id in validation:
                        choosedLabel = random.choice(labelsList)
                        labels.update({node.id : choosedLabel})
                    else:
                        labels.update({node.id : node.label})
                self.modelNodeLabels.update({self.models[i]: labels})
                setsOfLabels = dict([])
                for v in validation:
                    setsOfLabels.update({v: set([])})
                self.labelingSets.update({self.models[i]: setsOfLabels})
            break


