import unittest
import mockito as mockito

import copy
import networkx as nx
import numpy as np
from graph.method.lbp.CrossValMethods import CrossValMethods
from graph.method.common.CommonUtils import CommonUtils
from graph.method.lbp.FlatLBP import FlatLBP
from graph.method.lbp.LoopyBeliefPropagation import LoopyBeliefPropagation
from graph.evaluation.EvaluationTools import EvaluationTools
from graph.method.lbp.LBPTools import LBPTools
from graph.method.lbp.RwpLBP import RwpLBP
from graph.method.common.XValWithSampling import XValMethods
from tests.utils.TestUtils import TestUtils

STRATIFIED_RESULT_MEAN = 1.0

STRATIFIED_RESULT_SUM = 0.583

EXPECTED_RW_MEAN_RESULT = 0.286
EXPECTED_RW_SUM_RESULT = 0.583
EXPECTED_FUSION_MEAN_RESULT = 0.286
EXPECTED_FUSION_SUM_RESULT = 0.167
EXPECTED_FLAT_RESULT = 0.17
NUMBER_OF_CLASSES = 2
DEFAULT_ASSIGN = [1, 0, 1, 0, 1]

__author__ = 'Adrian'

class TestStringMethods(unittest.TestCase):


    methods = CrossValMethods()
    ev = EvaluationTools()
    utils = TestUtils()

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
        edges, nodes, nodesList = self.utils.prepareNodesAndEdges()
        #when
        mockito.when(graph).edges_iter(mockito.any(), data=True)\
            .thenReturn(self.utils.generateEdges(10, nodesList, edges))\
            .thenReturn(self.utils.generateEdges(10, nodesList, edges))\
            .thenReturn(self.utils.generateEdges(10, nodesList, edges))\
            .thenReturn(self.utils.generateEdges(10, nodesList, edges))\
            .thenReturn(self.utils.generateEdges(10, nodesList, edges))
        mockito.when(graph).nodes().thenReturn(nodes)
        fold_sum = self.methods.flatCrossVal(items, nrOfFolds, graph, nrOfNodes, defaultClassMat, lbpSteps, lbpThreshold, commonUtils.k_fold_cross_validation,
                                  commonUtils.prepareFoldClassMat, lbp.lbp, layerWeights, isRandomWalk, percentOfKnownNodes, adjMatPrep, prepareLayers, prepareClassMat)
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
        edges, nodes, nodesList = self.utils.prepareNodesAndEdges()
        edgesList = self.utils.prepareEdgesList(edges, nodesList)
        #when
        mockito.when(graph).edges_iter(mockito.any(), data=True)\
            .thenReturn(self.utils.generateEdges(10, nodesList, edges))\
            .thenReturn(self.utils.generateEdges(10, nodesList, edges))\
            .thenReturn(self.utils.generateEdges(10, nodesList, edges))\
            .thenReturn(self.utils.generateEdges(10, nodesList, edges))\
            .thenReturn(self.utils.generateEdges(10, nodesList, edges))
        mockito.when(graph).nodes().thenReturn(nodes)
        mockito.when(graph).edges(data=True).thenReturn(edgesList)
        fold_sum, fuz_mean_occ, sum = self.methods.multiLayerCrossVal(items, nrOfFolds, graph, nrOfNodes, defaultClassMat, lbpSteps, lbpThreshold, commonUtils.k_fold_cross_validation,
                                  tools.giveCorrectData, lbp.lbp, layerWeights, isRandomWalk, percentOfKnownNodes, adjMatPrep, tools.separate_layer, tools.prepareClassMatForFold)
        #then
        resultMean, resultSum = self.prepareFusionResults(fold_sum, fuz_mean_occ, sum, tools)
        roundedResultMean = round(resultMean, 3)
        roundedResultSum = round(resultSum, 3)

        assert roundedResultSum == EXPECTED_FUSION_SUM_RESULT
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
        edges, nodes, nodesList = self.utils.prepareNodesAndEdges()
        edgesList = self.utils.prepareEdgesList(edges, nodesList)
        #when
        mockito.when(graph).edges_iter(mockito.any(), data=True)\
            .thenReturn(self.utils.generateEdges(10, nodesList, edges))\
            .thenReturn(self.utils.generateEdges(10, nodesList, edges))\
            .thenReturn(self.utils.generateEdges(10, nodesList, edges))\
            .thenReturn(self.utils.generateEdges(10, nodesList, edges))\
            .thenReturn(self.utils.generateEdges(10, nodesList, edges))
        mockito.when(graph).nodes().thenReturn(nodes)
        mockito.when(graph).edges(data=True).thenReturn(edgesList)
        fold_sum, fuz_mean_occ, sum = self.methods.multiLayerCrossVal(items, nrOfFolds, graph, nrOfNodes, defaultClassMat, lbpSteps, lbpThreshold, commonUtils.k_fold_cross_validation,
                                  tools.giveCorrectData, rwp.propagation, layerWeights, True, percentOfKnownNodes, rwp.prepare_adjetency_matrix, tools.separate_layer, tools.prepareClassMatForFold)
        #then
        resultMean, resultSum = self.prepareFusionResults(fold_sum, fuz_mean_occ, sum, tools)
        roundedResultMean = round(resultMean, 3)
        roundedResultSum = round(resultSum, 3)

        assert roundedResultSum == EXPECTED_RW_SUM_RESULT
        assert roundedResultMean == EXPECTED_RW_MEAN_RESULT

    def test_multiCrossVal_for_stratified_xval(self):
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
        edges, nodes, nodesList = self.utils.prepareNodesAndEdges()
        edgesList = self.utils.prepareEdgesList(edges, nodesList)
        x_val_methods = XValMethods(graph)
        x_val = x_val_methods.stratifies_x_val
        #when
        mockito.when(graph).edges_iter(mockito.any(), data=True)\
            .thenReturn(self.utils.generateEdges(10, nodesList, edges))\
            .thenReturn(self.utils.generateEdges(10, nodesList, edges))\
            .thenReturn(self.utils.generateEdges(10, nodesList, edges))\
            .thenReturn(self.utils.generateEdges(10, nodesList, edges))\
            .thenReturn(self.utils.generateEdges(10, nodesList, edges))
        mockito.when(graph).nodes().thenReturn(nodes)
        mockito.when(graph).edges(data=True).thenReturn(edgesList)
        fold_sum, fuz_mean_occ, sum = self.methods.multiLayerCrossVal(nodesList, nrOfFolds, graph, nrOfNodes, defaultClassMat, lbpSteps, lbpThreshold, x_val,
                                  tools.giveCorrectData, lbp.lbp, layerWeights, isRandomWalk, percentOfKnownNodes, adjMatPrep, tools.separate_layer, tools.prepareClassMatForFold)
        #then
        resultMean, resultSum = self.prepareFusionResults(fold_sum, fuz_mean_occ, sum, tools)
        roundedResultMean = round(resultMean, 3)
        roundedResultSum = round(resultSum, 3)

        print str(roundedResultSum)
        assert roundedResultSum == STRATIFIED_RESULT_SUM
        assert roundedResultMean == STRATIFIED_RESULT_MEAN


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

    def prepareExperimentData(self):
        items = [0, 1, 2, 3, 4]
        defaultClassMat = self.utils.prepareTestClassMat()
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
