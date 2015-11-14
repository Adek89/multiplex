__author__ = 'Adrian'
import unittest
import mockito
import networkx as nx
import numpy as np

from graph.method.lbp.RwpLBP import RwpLBP
from graph.evaluation.EvaluationTools import EvaluationTools
from tests.utils.TestUtils import TestUtils
from graph.method.lbp.LBPTools import LBPTools

ORIGINAL_LABELS = [1, 0, 1, 0, 1]
MEAN_EXPECTED_RESULT = 0.286
SUM_EXPECTED_RESULT = 1.0

utils = TestUtils()
evaluation = EvaluationTools()
class TestsRwpLBP(unittest.TestCase):

    method = RwpLBP()

    def test_start(self):
        #given
        graph = mockito.mock(nx.MultiGraph)
        defaultClassMat = utils.prepareTestClassMat()
        nrOfClasses = 2
        nrOfNodes = 5
        nrOfFolds = 5
        lbpMaxSteps = 1000
        lbpThreshold = 0.001
        layerWeights = [1, 2]
        percentOfKnownNodes = 0.2
        method_type = 2
        edgesData, nodes, nodeList = utils.prepareNodesAndEdges()
        edgesList = utils.prepareEdgesList(edgesData, nodeList)
        #when
        mockito.when(graph).edges(data=True).thenReturn(edgesList)
        mockito.when(graph).nodes().thenReturn(nodes)
        foldSumEstimated, fusionMeanEstimated = self.method.start(graph, defaultClassMat, nrOfClasses, nrOfNodes, nrOfFolds, lbpMaxSteps, lbpThreshold, layerWeights, percentOfKnownNodes,
                                                                  method_type)
        #then
        foldSumResult = evaluation.calculateFMacro(ORIGINAL_LABELS, foldSumEstimated, nrOfClasses)
        fusionMeanResult = evaluation.calculateFMacro(ORIGINAL_LABELS, fusionMeanEstimated, nrOfClasses)
        fusionMeanRounded  = round(fusionMeanResult, 3)
        assert foldSumResult == SUM_EXPECTED_RESULT
        assert fusionMeanRounded == MEAN_EXPECTED_RESULT

    def test_propagation(self):
        graph = mockito.mock(nx.MultiGraph)
        nrOfNodes = 5
        lbpSteps = 1000
        lbpThreshold = 0.001
        percentOfKnownNodes = 0.2
        defaultClassMat = utils.prepareTestClassMat()
        edgesData, nodes, nodeList = utils.prepareNodesAndEdges()
        edgesList = utils.prepareEdgesList(edgesData, nodeList)
        sortedNodes = sorted(nodes, key=lambda node: node.id)

        mockito.when(graph).edges(data=True).thenReturn(edgesList)
        mockito.when(graph).nodes().thenReturn(nodes)

        tools = LBPTools(nrOfNodes, graph, defaultClassMat, lbpSteps, lbpThreshold, percentOfKnownNodes)
        tools.separate_layer(graph, [1, 2], defaultClassMat, [4])
        adjMatL1 = tools.adjMats["1"]
        adjMatL2 = tools.adjMats["2"]

        res = np.ndarray(shape=(5, 2))
        res[0] = [0, 0]
        res[1] = [0, 0]
        res[2] = [0, 0]
        res[3] = [0, 0]
        res[4] = [2, 2]

        matrices = [adjMatL1, adjMatL2]
        newRes = self.method.propagation(matrices, [], defaultClassMat, [4], lbpSteps, lbpThreshold)
        assert newRes[0][0] < newRes[0][1]
        assert newRes[1][0] < newRes[1][1]
        assert newRes[2][0] < newRes[2][1]
        assert newRes[3][0] < newRes[3][1]
        assert newRes[4][0] == 0
        assert newRes[4][1] == 1