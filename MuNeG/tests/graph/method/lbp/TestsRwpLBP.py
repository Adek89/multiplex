__author__ = 'Adrian'
import unittest
import mockito
import networkx as nx

from graph.method.lbp.RwpLBP import RwpLBP
from graph.evaluation.EvaluationTools import EvaluationTools
from tests.utils.TestUtils import TestUtils

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
        edgesData, nodes, nodeList = utils.prepareNodesAndEdges()
        edgesList = utils.prepareEdgesList(edgesData, nodeList)
        #when
        mockito.when(graph).edges(data=True).thenReturn(edgesList)
        mockito.when(graph).nodes().thenReturn(nodes)
        foldSumEstimated, fusionMeanEstimated = self.method.start(graph, defaultClassMat, nrOfClasses, nrOfNodes, nrOfFolds, lbpMaxSteps, lbpThreshold, layerWeights, percentOfKnownNodes)
        #then
        foldSumResult = evaluation.calculateFMacro(ORIGINAL_LABELS, foldSumEstimated, nrOfClasses)
        fusionMeanResult = evaluation.calculateFMacro(ORIGINAL_LABELS, fusionMeanEstimated, nrOfClasses)
        fusionMeanRounded  = round(fusionMeanResult, 3)
        assert foldSumResult == SUM_EXPECTED_RESULT
        assert fusionMeanRounded == MEAN_EXPECTED_RESULT

    def test_propagation(self):
        graph = mockito.mock(nx.MultiGraph)
        edgesData, nodes, nodeList = utils.prepareNodesAndEdges()
        sortedNodes = sorted(nodes, key=lambda node: node.id)
        adjacency_matrix = nx.adjacency_matrix(graph, sortedNodes, weight=None)
        matrices = [adjacency_matrix]
        self.method.propagation(matrices)