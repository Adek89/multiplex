__author__ = 'Adek'

import networkx as nx
import graph.method.common.CommonUtils as commons
import random
from graph.method.ica.ICA import ICA
class EnsambleClassification:

    models = set([])
    graph = nx.MultiGraph()
    percentTraining = 0.0
    nodesLabels = dict([]) # V
    modelNodeLabels = dict([]) # Y^
    labelingLists = dict([])#Y^iT
    ica = dict([])
    averageLabelings = dict([])
    finalLabelings = dict([])
    folds = dict([])
    fold = 0

    def __init__(self, models, graph, percentTraining, fold):
        self.models = models
        self.graph = graph
        self.percentTraining = percentTraining
        self.fold = fold
        for node in graph:
            self.nodesLabels.update({node.id : node.label})

    def meanOfAllLabels(self, validation):
        for i in range(self.models.__len__()):
            currLabeling = self.labelingLists[self.models[i]]
            for v in validation:
                nodeLabels = currLabeling[v]
                calculatedLabel = sum(nodeLabels) / float(nodeLabels.__len__())
                self.averageLabelings[self.models[i]].update({v: calculatedLabel})

    def finalLabelsAssigning(self, validation):
        self.meanOfAllLabels(validation)
        for i in range(self.models.__len__()):
            for v in validation:
                self.finalLabelings.update({v: self.finalLabelings[v] + self.averageLabelings[self.models[i]][v]})
        for key, value in self.finalLabelings.items():
            self.finalLabelings[key] = int(round(value / self.models.__len__()))

    def icaIteration(self, j, testNodes):
        currentModel = self.models[j]
        ica = self.ica[currentModel]
        ica.performIteration(range(testNodes.__len__()))
        return currentModel, ica

    def inference(self, testNodes):
        for j in range(self.models.__len__()):
            currentModel, ica = self.icaIteration(j, testNodes)
        for i in range(1, 1000):
            for j in range(self.models.__len__()):
                currentModel, ica = self.icaIteration(j, testNodes)
                currentLabeling = [self.ica[m].y for m in self.models]
                for testNode in testNodes:
                    acrossModels = [el[testNode] for el in currentLabeling]
                    averageLabeling = float(sum(acrossModels)) / float(self.models.__len__())
                    self.modelNodeLabels[currentModel].update({testNode: round(averageLabeling)})
                    currentLabelingList = self.labelingLists[currentModel][testNode.id]
                    currentLabelingList.append(averageLabeling)
                ica.y = dict([(node, self.modelNodeLabels[currentModel][node]) for node in testNodes])

    def initiateLabels(self, graphNodes, labelsList, validation):
        labels = dict([])
        for node in graphNodes:
            if node.id in validation:
                choosedLabel = random.choice(labelsList)
                labels.update({node: choosedLabel})
            else:
                labels.update({node: node.label})
        return labels

    def structurePreparation(self, graphNodes, labelsList, testNodes, trainNodes, validation):
        self.finalLabelings = dict([])
        self.modelNodeLabels = dict([])
        self.averageLabelings = dict([])
        self.ica = dict([])
        for v in validation:
            self.finalLabelings.update({v: 0})
        for i in range(self.models.__len__()):
            labels = self.initiateLabels(graphNodes, labelsList, validation)
            self.modelNodeLabels.update({self.models[i]: labels})
            listsOfLabels = dict([])
            listOfEndLabeling = dict([])
            for v in validation:
                listsOfLabels.update({v: list()})
                listOfEndLabeling.update({v: 0})
            self.labelingLists.update({self.models[i]: listsOfLabels})
            self.averageLabelings.update({self.models[i]: listOfEndLabeling})
            ica = ICA(self.graph, trainNodes, testNodes, self.models[i])
            ica.y = dict([(node, labels[node]) for node in testNodes])
            self.ica.update({self.models[i]: ica})

    def singleFold(self, commonUtils, graphNodes, items, labelsList):
        iter = 0
        for training, validation in commonUtils.k_fold_cross_validation(items, 5, self.percentTraining):
            if (self.fold == iter):
                break
            else:
                iter += 1
        trainNodes = filter(lambda node: node.id in training, graphNodes)
        testNodes = filter(lambda node: node.id in validation, graphNodes)

        self.structurePreparation(graphNodes, labelsList, testNodes, trainNodes, validation)
        self.inference(testNodes)
        self.finalLabelsAssigning(validation)
        self.folds.update({self.fold: self.finalLabelings})

    def classify(self):
        labelsList = range(2)
        graphNodes = self.graph.nodes()
        print 'graph Nodes length: ' + str(graphNodes.__len__())
        items = range(graphNodes.__len__())
        print 'items length: ' + str(items.__len__())
        commonUtils = commons.CommonUtils()
        self.singleFold(commonUtils, graphNodes, items, labelsList)
        self.finalLabelings = dict([])
        list = [ e[1] for e in self.folds.items()]
        for l in list:
            self.finalLabelings.update(l)
        return self.finalLabelings

#layers = set([ (edata['layer']) for u,v,edata in graph.edges(data=True)])


