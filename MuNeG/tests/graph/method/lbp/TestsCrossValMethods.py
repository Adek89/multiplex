import unittest
import mockito as mockito

import copy
import networkx as nx
import numpy as np
from graph.method.lbp.CrossValMethods import CrossValMethods
from graph.method.common.CommonUtils import CommonUtils
from graph.method.lbp.FlatLBP import FlatLBP
from graph.method.lbp.LoopyBeliefPropagation import LoopyBeliefPropagation
from graph.gen.Node import Node
from graph.gen.Group import Group
from graph.evaluation.EvaluationTools import EvaluationTools
from graph.method.lbp.LBPTools import LBPTools
from graph.method.lbp.RwpLBP import RwpLBP

EXPECTED_RW_MEAN_RESULT = 0.286
EXPECTED_RW_SUM_RESULT = 1.0
EXPECTED_FUSION_MEAN_RESULT = 0.286
EXPECTED_FUSION_SUM_RESULT = 0.375
EXPECTED_FLAT_RESULT = 0.17
NUMBER_OF_CLASSES = 2
DEFAULT_ASSIGN = [1, 0, 1, 0, 1]

__author__ = 'Adrian'

class TestStringMethods(unittest.TestCase):


    methods = CrossValMethods()
    ev = EvaluationTools()

    def test_flatCrossVal(self):
        #given
        graph, \
        nrOfNodes, \
        adjMatPrep, \
        defaultClassMat, \
        isRandomWalk, \
        items, \
        commonUtils, \
        layerWeights, \
        lbp, \
        lbpSteps, \
        lbpThreshold, \
        nrOfFolds, \
        percentOfKnownNodes, \
        prepareClassMat, \
        prepareLayers, \
        flatLBP,\
        tools,\
        rwp = self.prepareExperimentData()
        edges, nodes, nodesList = self.prepareNodesAndEdges()
        #when
        mockito.when(graph).edges_iter(mockito.any(), data=True)\
            .thenReturn(self.generateEdges(10, nodesList, edges))\
            .thenReturn(self.generateEdges(10, nodesList, edges))\
            .thenReturn(self.generateEdges(10, nodesList, edges))\
            .thenReturn(self.generateEdges(10, nodesList, edges))\
            .thenReturn(self.generateEdges(10, nodesList, edges))
        mockito.when(graph).nodes().thenReturn(nodes)
        fold_sum = self.methods.flatCrossVal(items, nrOfFolds, graph, nrOfNodes, defaultClassMat, lbpSteps, lbpThreshold, commonUtils.k_fold_cross_validation,
                                  flatLBP.prepareFoldClassMat, lbp.lbp, layerWeights, isRandomWalk, percentOfKnownNodes, adjMatPrep, prepareLayers, prepareClassMat)
        #then
        roundedResult = self.prepareFlatResult(fold_sum, tools)

        assert roundedResult == EXPECTED_FLAT_RESULT

    def test_multiCrossVal(self):
        #given
        graph, \
        nrOfNodes, \
        adjMatPrep, \
        defaultClassMat, \
        isRandomWalk, \
        items, \
        commonUtils, \
        layerWeights, \
        lbp, \
        lbpSteps, \
        lbpThreshold, \
        nrOfFolds, \
        percentOfKnownNodes, \
        prepareClassMat, \
        prepareLayers, \
        flatLBP,\
        tools,\
        rwp = self.prepareExperimentData()
        edges, nodes, nodesList = self.prepareNodesAndEdges()
        edgesList = [(nodesList[0], nodesList[2], edges[0]),
                     (nodesList[1], nodesList[3], edges[0]),
                     (nodesList[2], nodesList[4], edges[0]),
                     (nodesList[0], nodesList[1], edges[0]),
                     (nodesList[1], nodesList[4], edges[0]),
                     (nodesList[0], nodesList[2], edges[1]),
                     (nodesList[1], nodesList[3], edges[1]),
                     (nodesList[1], nodesList[4], edges[1]),
                     (nodesList[0], nodesList[1], edges[1]),
                     (nodesList[2], nodesList[3], edges[1])]
        #when
        mockito.when(graph).edges_iter(mockito.any(), data=True)\
            .thenReturn(self.generateEdges(10, nodesList, edges))\
            .thenReturn(self.generateEdges(10, nodesList, edges))\
            .thenReturn(self.generateEdges(10, nodesList, edges))\
            .thenReturn(self.generateEdges(10, nodesList, edges))\
            .thenReturn(self.generateEdges(10, nodesList, edges))
        mockito.when(graph).nodes().thenReturn(nodes)
        mockito.when(graph).edges(data=True).thenReturn(edgesList)
        fold_sum, fuz_mean_occ, sum = self.methods.multiLayerCrossVal(items, nrOfFolds, graph, nrOfNodes, defaultClassMat, lbpSteps, lbpThreshold, commonUtils.k_fold_cross_validation,
                                  tools.giveCorrectData, lbp.lbp, layerWeights, isRandomWalk, percentOfKnownNodes, adjMatPrep, tools.separate_layer, tools.prepareClassMatForFold)
        #then
        resultMean, resultSum = self.prepareFusionResults(fold_sum, fuz_mean_occ, sum, tools)
        roundedResultMean = round(resultMean, 3)

        assert resultSum == EXPECTED_FUSION_SUM_RESULT
        assert roundedResultMean == EXPECTED_FUSION_MEAN_RESULT

    def test_rwcCrossVal(self):
        #given
        graph, \
        nrOfNodes, \
        adjMatPrep, \
        defaultClassMat, \
        isRandomWalk, \
        items, \
        commonUtils, \
        layerWeights, \
        lbp, \
        lbpSteps, \
        lbpThreshold, \
        nrOfFolds, \
        percentOfKnownNodes, \
        prepareClassMat, \
        prepareLayers, \
        flatLBP,\
        tools,\
        rwp = self.prepareExperimentData()
        edges, nodes, nodesList = self.prepareNodesAndEdges()
        edgesList = [(nodesList[0], nodesList[2], edges[0]),
                     (nodesList[1], nodesList[3], edges[0]),
                     (nodesList[2], nodesList[4], edges[0]),
                     (nodesList[0], nodesList[1], edges[0]),
                     (nodesList[1], nodesList[4], edges[0]),
                     (nodesList[0], nodesList[2], edges[1]),
                     (nodesList[1], nodesList[3], edges[1]),
                     (nodesList[1], nodesList[4], edges[1]),
                     (nodesList[0], nodesList[1], edges[1]),
                     (nodesList[2], nodesList[3], edges[1])]
        #when
        mockito.when(graph).edges_iter(mockito.any(), data=True)\
            .thenReturn(self.generateEdges(10, nodesList, edges))\
            .thenReturn(self.generateEdges(10, nodesList, edges))\
            .thenReturn(self.generateEdges(10, nodesList, edges))\
            .thenReturn(self.generateEdges(10, nodesList, edges))\
            .thenReturn(self.generateEdges(10, nodesList, edges))
        mockito.when(graph).nodes().thenReturn(nodes)
        mockito.when(graph).edges(data=True).thenReturn(edgesList)
        fold_sum, fuz_mean_occ, sum = self.methods.multiLayerCrossVal(items, nrOfFolds, graph, nrOfNodes, defaultClassMat, lbpSteps, lbpThreshold, commonUtils.k_fold_cross_validation,
                                  tools.giveCorrectData, rwp.propagation, layerWeights, True, percentOfKnownNodes, rwp.prepare_adjetency_matrix, tools.separate_layer, tools.prepareClassMatForFold)
        #then
        resultMean, resultSum = self.prepareFusionResults(fold_sum, fuz_mean_occ, sum, tools)
        roundedResultMean = round(resultMean, 3)

        assert resultSum == EXPECTED_RW_SUM_RESULT
        assert roundedResultMean == EXPECTED_RW_MEAN_RESULT


    def prepareFusionResults(self, fold_sum, fuz_mean_occ, sum, tools):
        fusion_mean = self.calculateFusionMean(fuz_mean_occ, sum)
        toEvaluateSum = tools.prepareToEvaluate(fold_sum, NUMBER_OF_CLASSES)
        toEvaluateMean = tools.prepareToEvaluate(fusion_mean, NUMBER_OF_CLASSES)
        resultSum = self.ev.calculateFMacro(DEFAULT_ASSIGN, toEvaluateSum, NUMBER_OF_CLASSES)
        resultMean = self.ev.calculateFMacro(DEFAULT_ASSIGN, toEvaluateMean, NUMBER_OF_CLASSES)
        return resultMean, resultSum

    def prepareFlatResult(self, fold_sum, tools):
        toEvaluate = tools.prepareToEvaluate(fold_sum, NUMBER_OF_CLASSES)
        result = self.ev.calculateFMacro(DEFAULT_ASSIGN, toEvaluate, NUMBER_OF_CLASSES)
        roundedResult = round(result, 2)
        return roundedResult

    def calculateFusionMean(self, fuz_mean_occ, sum):
        fusion_mean = copy.deepcopy(sum)
        for iter in range(0, len(sum)):
            fusion_mean[iter][1] = sum[iter][1] / fuz_mean_occ[iter]
            fusion_mean[iter][2] = sum[iter][2] / fuz_mean_occ[iter]
        return fusion_mean

    def prepareNodesAndEdges(self):
        groupRed = Group('r', 1)
        groupBlue = Group('b', 2)
        node1 = Node(groupRed, 1, 0)
        node2 = Node(groupBlue, 0, 1)
        node3 = Node(groupRed, 1, 2)
        node4 = Node(groupBlue, 0, 3)
        node5 = Node(groupBlue, 1, 4)
        nodes = ({node1, node2, node3, node4, node5})
        nodeList = [node1, node2, node3, node4, node5]
        edge1 = dict([('layer', 'L1'), ('conWeight', 0.5), ('weight', 1)])
        edge2 = dict([('layer', 'L2'), ('conWeight', 0.5), ('weight', 2)])
        edgesData = ([edge1, edge2])
        return edgesData, nodes, nodeList

    def generateEdges(self, nrOfEdges, nodes, edgesData):
            i = 0
            for i in range(0, nrOfEdges):
                if (i == 0):
                    yield (nodes[0], nodes[2], edgesData[0])
                elif (i == 1):
                    yield (nodes[1], nodes[3], edgesData[0])
                elif (i == 2):
                    yield (nodes[2], nodes[4], edgesData[0])
                elif (i == 3):
                    yield (nodes[0], nodes[1], edgesData[0])
                elif (i == 4):
                    yield (nodes[1], nodes[4], edgesData[0])
                elif (i == 5):
                    yield (nodes[0], nodes[2], edgesData[1])
                elif (i == 6):
                    yield (nodes[1], nodes[3], edgesData[1])
                elif (i == 7):
                    yield (nodes[1], nodes[4], edgesData[1])
                elif (i == 8):
                    yield (nodes[0], nodes[1], edgesData[1])
                elif (i == 9):
                    yield (nodes[2], nodes[3], edgesData[1])

    def prepareTestClassMat(self):
        defaultClassMat = np.ndarray(shape=(5, 2))
        defaultClassMat[0] = [0, 1]
        defaultClassMat[1] = [1, 0]
        defaultClassMat[2] = [0, 1]
        defaultClassMat[3] = [1, 0]
        defaultClassMat[4] = [0, 1]
        return defaultClassMat

    def prepareExperimentData(self):
        items = [0, 1, 2, 3, 4]
        defaultClassMat = self.prepareTestClassMat()
        nrOfFolds = 5
        graph = mockito.mock(nx.MultiGraph)
        nrOfNodes = items.__len__()
        lbpSteps = 1000
        lbpThreshold = 0.01
        commonUtils = CommonUtils()
        flatLBP = FlatLBP()
        lbp = LoopyBeliefPropagation()
        layerWeights = [1, 2]
        isRandomWalk = False
        percentOfKnownNodes = 0.2
        adjMatPrep = None
        prepareLayers = None
        prepareClassMat = None
        tools = LBPTools(items.__len__(), graph, defaultClassMat, lbpSteps, lbpThreshold, percentOfKnownNodes)
        rwp = RwpLBP()
        return graph, nrOfNodes, adjMatPrep, defaultClassMat, isRandomWalk, items, commonUtils, layerWeights, lbp, lbpSteps, lbpThreshold, nrOfFolds, percentOfKnownNodes, prepareClassMat, prepareLayers, flatLBP, tools, rwp

