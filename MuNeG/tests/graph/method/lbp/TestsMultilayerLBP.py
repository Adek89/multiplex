__author__ = 'Adrian'
import unittest
import mockito
import networkx as nx
from graph.method.lbp.Multilayer_LBP import Multilayer_LBP
from graph.evaluation.EvaluationTools import EvaluationTools
from tests.utils.TestUtils import TestUtils

ORIGINAL_LABELS = [1, 0, 1, 0, 1]
EXPECTED_FUSION_MEAN_RESULT = 0.286
EXPECTED_FOLD_SUM_RESULT = 0.286

utils = TestUtils()
evaluation = EvaluationTools()
class TestsMultilayerLBP(unittest.TestCase):

    method = Multilayer_LBP()

    def test_multilayer(self):
        #given
        graph = mockito.mock(nx.MultiGraph)
        defaultClassMat = utils.prepareTestClassMat()
        edgesData, nodes, nodeList = utils.prepareNodesAndEdges()
        edges = utils.prepareEdgesList(edgesData, nodeList)
        nrOfClasses = 2
        nrOfNodes = 5
        nrOfFolds = 5
        lbpMaxSteps = 1000
        lbpThreshold = 0.001
        layerWeights = [1, 2]
        percentOfTrainingNodes = 0.2
        method_type = 2
        #when
        mockito.when(graph).edges(data=True).thenReturn(edges)
        mockito.when(graph).nodes().thenReturn(nodes)
        foldSumEstimated, fusionMeanEstimated = self.method.start(graph, defaultClassMat, nrOfClasses, nrOfNodes,
                                                                  nrOfFolds, lbpMaxSteps, lbpThreshold, layerWeights,
                                                                  percentOfTrainingNodes, method_type)
        #then
        foldSumResult = evaluation.calculateFMacro(ORIGINAL_LABELS, foldSumEstimated, nrOfClasses)
        fusionMeanResult = evaluation.calculateFMacro(ORIGINAL_LABELS, fusionMeanEstimated, nrOfClasses)
        fusionMeanRounded = round(fusionMeanResult, 3)
        foldSumResultRounded = round(foldSumResult, 3)
        assert foldSumResultRounded == EXPECTED_FOLD_SUM_RESULT
        assert fusionMeanRounded == EXPECTED_FUSION_MEAN_RESULT