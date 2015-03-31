from sklearn.semi_supervised.label_propagation import LabelPropagation

__author__ = 'Adek'

from graph.gen.GraphGenerator import GraphGenerator
import networkx as nx
from graph.method.lbp.NetworkUtils import NetworkUtils
from graph.method.ensamble.EnsambleLearning import EnsambleLearning
import graph.method.ica.ClassifiersUtil as cu
from graph.method.ensamble.EnsambleClassification import EnsambleClassification
from graph.evaluation.EvaluationTools import EvaluationTools
from graph.method.lbp.LBPTools import LBPTools
import random
import re
from graph.analyser.GraphAnalyser import GraphAnalyser
import csv
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


    synthetic = nx.MultiGraph()
    syntheticClassMat = []
    syntheticNrOfClasses = 0

    nu = NetworkUtils()

    models = []

    nrOfNodesInSubgraph = 0

    ensambleSet = set([])

    finalLabelings = dict([])

    flatAccuracy = 0

    graphs = dict([])
    nodes = dict([])

    FILE_PATH = "/home/apopiel/tmp_local/output"
    # FILE_PATH = "output"

    def __init__(self,  nrOfNodes, nrOfGroups, grLabelHomogenity, prEdgeInGroup,
                 prEdgeBetweenGroups, nrOfLayers, percentOfTrainignNodes, nrOfNodesInSubgraph, counter):
        self.NUMBER_OF_NODES = nrOfNodes
        self.NUMBER_OF_GROUPS = nrOfGroups
        self.GROUP_LABEL_HOMOGENITY = grLabelHomogenity
        self.PROBABILITY_OF_EDGE_EXISTANCE_IN_SAME_GROUP = prEdgeInGroup
        self.PROBABILITY_OF_EDGE_EXISTANCE_BETWEEN_OTHER_GROUPS = prEdgeBetweenGroups
        self.initLayers(nrOfLayers)
        self.nrOfLayers = nrOfLayers
        self.percentOfTrainignNodes = percentOfTrainignNodes
        self.counter = counter
        self.models = cu.knownModels()
        self.nrOfNodesInSubgraph = nrOfNodesInSubgraph
        self.prepareNumberOfGroups(nrOfNodes, nrOfGroups)



    def generateSyntheticData(self):
        self.gg = GraphGenerator(self.NUMBER_OF_NODES, self.AVERAGE_GROUP_SIZE, self.LAYERS_WEIGHTS,
                                 self.GROUP_LABEL_HOMOGENITY, self.PROBABILITY_OF_EDGE_EXISTANCE_IN_SAME_GROUP,
                                 self.PROBABILITY_OF_EDGE_EXISTANCE_BETWEEN_OTHER_GROUPS, self.LAYERS_NAME)
        self.synthetic = self.gg.generate()
        print 'Step 1 graph generated'
        ga = GraphAnalyser(self.synthetic, self.percentOfTrainignNodes, self.counter)
        ga.analyse()


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
        self.syntheticClassMat, self.syntheticNrOfClasses = self.nu.createClassMat(self.synthetic)

    def learning(self):
        el = EnsambleLearning(self.synthetic, self.models, self.nrOfNodesInSubgraph)
        self.ensambleSet = el.ensamble()

    def classify(self, fold):
        ec = EnsambleClassification(map(lambda x: x, self.ensambleSet), self.synthetic, self.percentOfTrainignNodes, fold)
        self.finalLabelings = ec.classify()

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
        self.syntheticLabels = self.prepareOriginalLabels(self.syntheticClassMat, self.syntheticNrOfClasses)
        ev = EvaluationTools()
        accuracy = ev.calculateAccuracy(self.syntheticLabels, syntheticResult)
        return accuracy

    def layerEvaluation(self):
        syntheticResult = [v for e, v in sorted(self.finalLabelings.items())]
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

    def layeredExperiment(self, fold):
        self.separate_layer(self.synthetic, range(1, self.nrOfLayers + 1))
        layerResults = dict([])
        for l in range(1, self.nrOfLayers + 1):
            graph = self.graphs[str(l)]
            self.synthetic = graph
            self.preprocessing()
            self.learning()
            self.classify(fold)
            layerAccuracy = self.layerEvaluation()
            layerResults.update({l: layerAccuracy})
        avgResults = self.layeredResults(layerResults)
        layeredAccuracy = self.evaluation(avgResults)
        return layeredAccuracy

    def processExperiments(self):
        fold = random.choice(range(0,5))
        self.generateSyntheticData()
        self.preprocessing()
        self.learning()
        self.classify(fold)
        syntheticLabels = [v for e, v in sorted(self.finalLabelings.items())]
        self.flatAccuracy = self.evaluation(syntheticLabels)

        layeredAccuracy = self.layeredExperiment(fold)
        with open(self.FILE_PATH + str(self.counter) + '.csv', 'ab') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow([self.NUMBER_OF_NODES, self.NUMBER_OF_GROUPS, self.GROUP_LABEL_HOMOGENITY,
                              self.PROBABILITY_OF_EDGE_EXISTANCE_IN_SAME_GROUP, self.PROBABILITY_OF_EDGE_EXISTANCE_BETWEEN_OTHER_GROUPS,
                                self.nrOfLayers, self.percentOfTrainignNodes, self.nrOfNodesInSubgraph,
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
        rests = dict([])
        for i in layers:
            rests[str(i)] = gNodes.difference(self.nodes[str(i)])
            for node in rests[str(i)]:
                self.graphs[str(i)].add_node(node)

    def addToGraph(self, g, n0, n1, nodes, edge):

        if not g.has_node(n0):
            nodes.add(n0)
        if not g.has_node(n1):
            nodes.add(n1)

        g.add_edge(n0, n1, edge[2])
