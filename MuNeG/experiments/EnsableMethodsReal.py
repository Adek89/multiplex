__author__ = 'Adek'

import random
import re
import csv
import threading
import copy

import networkx as nx

from bin.graph.method.lbp.NetworkUtils import NetworkUtils
from bin.method.ensamble.EnsambleLearning import EnsambleLearning
import bin.graph.method.ica.ClassifiersUtil as cu
from bin.graph.method.ensamble.EnsambleClassification import EnsambleClassification
from bin.graph.evaluation.EvaluationTools import EvaluationTools


class EnsambleMethods:

    NUMBER_OF_NODES = 0;
    AVERAGE_GROUP_SIZE = 0
    LAYERS_WEIGHTS = []
    GROUP_LABEL_HOMOGENITY = 0
    PROBABILITY_OF_EDGE_EXISTANCE_IN_SAME_GROUP = 0
    PROBABILITY_OF_EDGE_EXISTANCE_BETWEEN_OTHER_GROUPS = 0
    LAYERS_NAME = []
    NUMBER_OF_GROUPS = 0
    nrOfLayers = 0
    percentOfTrainignNodes = 0
    counter = 0


    real = nx.MultiGraph()
    realClassMat = []
    realNrOfClasses = 0

    nu = NetworkUtils()

    models = []

    nrOfNodesInSubgraph = 0

    ensambleSet = set([])

    finalLabelings = dict([])

    flatAccuracy = 0

    graphs = dict([])
    nodes = dict([])
    layerResults = dict([])

    threads = dict([])

    FILE_PATH = "/home/adrian/tmp_local/outputreal"
    # FILE_PATH = "output"

    def __init__(self, graph, nrOfNodesInSubgraph, percentOfKnownNodes):
        self.real = graph
        self.nrOfNodesInSubgraph = nrOfNodesInSubgraph
        self.models = cu.knownModels()
        self.percentOfTrainignNodes = percentOfKnownNodes
        layers = set([ (edata['layer']) for u,v,edata in self.real.edges(data=True)])
        layersNr = layers.__len__()
        self.nrOfLayers = layersNr
        self.initLayers(layersNr)


    def initLayers(self, nrOfLayers):
        for i in xrange(0, nrOfLayers):
            self.LAYERS_WEIGHTS.append(i + 1)
            self.LAYERS_NAME.append("L"+str(i+1))

    def prepareNumberOfGroups(self, nrOfNodes, nrOfGroups):
        while True:
            dividedInt = nrOfNodes % nrOfGroups
            if (not dividedInt == 0):
                nrOfNodes = nrOfNodes + 1
            else:
                break
        self.NUMBER_OF_NODES = nrOfNodes
        self.AVERAGE_GROUP_SIZE = nrOfNodes / nrOfGroups

    def preprocessing(self):
        self.realClassMat, self.realNrOfClasses = self.nu.createClassMat(self.real)

    def preprocessingLayer(self, graph):
        return self.nu.createClassMat(graph)

    def learning(self):
        el = EnsambleLearning(copy.deepcopy(self.real), self.models, self.nrOfNodesInSubgraph)
        self.ensambleSet, self.real = el.ensamble()

    def learningLayer(self, graph):
        el = EnsambleLearning(copy.deepcopy(graph), cu.knownModels(), self.nrOfNodesInSubgraph)
        return el.ensamble()

    def classify(self, fold):
        ec = EnsambleClassification(map(lambda x: x, self.ensambleSet), self.real, self.percentOfTrainignNodes, fold)
        self.finalLabelings = ec.classify()

    def classifyLayer(self, ensambleSet, graph, fold):
        ec = EnsambleClassification(map(lambda x: x, ensambleSet), graph, self.percentOfTrainignNodes, fold)
        return ec.classify()

    def prepareOriginalLabels(self, defaultClassMat, nrOfClasses):
        classMatForEv = []
        for i in sorted(self.finalLabelings.keys()):
            maxi = 0
            for j in range(1, nrOfClasses):
                if (defaultClassMat[i][j] > defaultClassMat[i][maxi]):
                    maxi = j
            classMatForEv.append(maxi)
        return classMatForEv

    def evaluation(self, syntheticResult):
        self.syntheticLabels = self.prepareOriginalLabels(self.realClassMat, self.realNrOfClasses)
        print 'Original labels: ' + str(self.syntheticLabels)
        ev = EvaluationTools()
        accuracy = ev.calculateAccuracy(self.syntheticLabels, syntheticResult)
        return accuracy

    def layerEvaluation(self, finalLabelings):
        syntheticResult = [v for e, v in sorted(finalLabelings.items())]
        return syntheticResult


    def layeredResults(self, layerResults):
        results = [k for v, k in layerResults.items()]
        iter = 0
        avgResults = []
        for e in results[0]:
            sum = 0
            elems = 0
            for r in results:
                sum += r[iter]
                elems += 1
            iter += 1
            avgResults.append(int(round(float(sum) / float(elems))))
        return avgResults

    def executeSingleLayer(self, fold, l):
        graph = self.graphs[str(l)]
        syntheticClassMat, syntheticNrOfClasses = self.preprocessingLayer(graph)
        ensambleSet, graph = self.learningLayer(graph)
        finalLabelings = self.classifyLayer(ensambleSet, graph, fold)
        layerAccuracy = self.layerEvaluation(finalLabelings)
        print 'Layer result, layer ' + str(l) + ': ' + str(layerAccuracy)
        self.layerResults.update({l: layerAccuracy})

    def layeredExperiment(self, fold):
        self.separate_layer(self.real, range(1, self.nrOfLayers + 1))
        for l in range(1, self.nrOfLayers + 1):
            thread = threading.Thread(target=self.executeSingleLayer, args=(fold, l))
            self.threads.update({l:thread})
            thread.start()
        for (l, t) in self.threads.items():
            t.join()
        avgResults = self.layeredResults(self.layerResults)
        layeredAccuracy = self.evaluation(avgResults)
        return layeredAccuracy

    def processExperiments(self):
        fold = random.choice(range(0,5))
        self.preprocessing()
        self.learning()
        self.classify(fold)
        syntheticLabels = [v for e, v in sorted(self.finalLabelings.items())]
        print 'Flat result' + str(syntheticLabels)
        self.flatAccuracy = self.evaluation(syntheticLabels)

        layeredAccuracy = self.layeredExperiment(fold)
        layers = set([ (edata['layer']) for u,v,edata in self.real.edges(data=True)])
        with open(self.FILE_PATH + str(self.counter) + '.csv', 'ab') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow([self.real.nodes().__len__(), layers.__len__(), self.percentOfTrainignNodes, self.nrOfNodesInSubgraph,
                            self.flatAccuracy, layeredAccuracy])


    def separate_layer(self, graph, layers):
        for i in layers:
            self.graphs[str(i)] = nx.Graph()
            self.nodes[str(i)] = set([])
        for edge in graph.edges(data=True):
            for label in layers:
                temp = ".*'layer': 'L"+str(label)+"'.*"
                #layer filter
                if re.match( str(temp),str(edge)):
                    break
            n0 = edge[0]
            n1 = edge[1]
            self.addToGraph(self.graphs[str(label)], n0, n1, self.nodes[str(label)], edge)
        gNodes = set(graph.nodes())
        rests = set([])
        for i in layers:
            rests = gNodes.difference(self.nodes[str(i)])
            for node in rests:
                self.graphs[str(i)].add_node(node)

    def addToGraph(self, g, n0, n1, nodes, edge):

        if not g.has_node(n0):
            nodes.add(n0)
        if not g.has_node(n1):
            nodes.add(n1)

        g.add_edge(n0, n1, edge[2])
