__author__ = 'Adrian'
import unittest
import mockito
import networkx as nx
from graph.method.lbp.LBPTools import LBPTools
from tests.utils.TestUtils import TestUtils

NUMBER_OF_LAYERS = 2
EXPECTED_LAYER_NAME = 'L2'
EXPECTED_CLASSMAT_FOR_EVALUATION = [0, 1, 0, 0, 1]

nrOfNodes = 5
graph = mockito.mock(nx.MultiGraph())
utils = TestUtils()
defaultClassMat = utils.prepareTestClassMat()
lbpMaxSteps = 1000
lbpThreshold = 0.01
percentOfTrainingNodes = 0.2
training = [0, 1, 2, 3]

class LBPTools(unittest.TestCase):

    tools = LBPTools(nrOfNodes, graph, defaultClassMat, lbpMaxSteps, lbpThreshold, percentOfTrainingNodes)

    def test_prepareClassMatForFold(self):
        #given
        self.tools.classMats["1"] = defaultClassMat.copy()
        #when
        self.tools.prepareClassMatForFold(1, training)
        #then
        folds = self.tools.folds["1"]
        assert folds[0].tolist() == [0.0, 1.0]
        assert folds[1].tolist() == [1.0, 0.0]
        assert folds[2].tolist() == [0.0, 1.0]
        assert folds[3].tolist() == [1.0, 0.0]
        assert folds[4].tolist() == [0.5, 0.5]

    def test_separateLayer(self):
        #given
        layers = [1, 2]
        edgesData, nodes, nodeList = utils.prepareNodesAndEdges()
        edges = utils.prepareEdgesList(edgesData, nodeList)
        #when
        mockito.when(graph).edges(data=True).thenReturn(edges)
        mockito.when(graph).nodes().thenReturn(nodes)
        self.tools.separate_layer(graph, layers, defaultClassMat)
        #then
        class_mats = self.tools.classMats
        graphs = self.tools.graphs
        assert class_mats.__len__() == NUMBER_OF_LAYERS
        assert graphs.__len__() == NUMBER_OF_LAYERS
        assert graphs["2"].edges(data=True)[2][2]['layer'] == EXPECTED_LAYER_NAME

    def test_prepareToEvaluate(self):
        foldSum = [[1, 3, 2], [2, 0, 5], [3, 2, 2], [4, 5, 0], [5, 2, 3]]
        classMatForEv = self.tools.prepareToEvaluate(foldSum, 2)
        assert classMatForEv == EXPECTED_CLASSMAT_FOR_EVALUATION
