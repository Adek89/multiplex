__author__ = 'Adrian'
import unittest
import networkx as nx
import numpy as np
import mockito
from graph.method.lbp.LoopyBeliefPropagation import LoopyBeliefPropagation
from tests.utils.TestUtils import TestUtils

utils = TestUtils()
NOT_EXPECTED_ELEMENT = 0.5
EXPECTED_RESULT_LENGTH = 5

class TestsLoopyBeliefPropagation(unittest.TestCase):
    lbp = LoopyBeliefPropagation()

    def test_lbp(self):
        #given
        graph = mockito.mock(nx.MultiGraph)
        edgesData, nodes, nodeList = utils.prepareNodesAndEdges()
        class_mat = utils.prepareTestClassMatWithUnknownNodes()
        sortedNodes = sorted(nodes, key=lambda node: node.id)
        repetitions = 1000
        epsilon = 0.01
        trainingInstances = [0, 1, 2, 3]
        testingInstances = [4]
        #when
        mockito.when(graph).edges_iter(mockito.any(), data=True)\
            .thenReturn(utils.generateEdges(10, nodeList, edgesData))
        adjacency_matrix = nx.adjacency_matrix(graph, sortedNodes, weight=None)
        #then
        res = self.lbp.lbp(adjacency_matrix, class_mat, repetitions, epsilon, trainingInstances, testingInstances)
        assert res.__len__() == EXPECTED_RESULT_LENGTH
        assert res[0].tolist() == [0.0, 1.0]
        assert res[1].tolist() == [1.0, 0.0]
        assert res[2].tolist() == [0.0, 1.0]
        assert res[3].tolist() == [1.0, 0.0]
        assert res[4][0] <> NOT_EXPECTED_ELEMENT
        assert res[4][1] <> NOT_EXPECTED_ELEMENT

    def test_notStopConditionReached(self):
        defaultClassMat = utils.prepareTestClassMat()
        result = self.lbp.stopConditionReached(defaultClassMat, 0.01)
        assert not result

    def test_stopConditionReached(self):
        classMat = np.ndarray(shape=(5, 2))
        classMat[0] = [0, 0]
        classMat[1] = [0, 0]
        classMat[2] = [0, 0]
        classMat[3] = [0, 0]
        classMat[4] = [0, 0.001]
        result = self.lbp.stopConditionReached(classMat, 0.01)
        assert result

    def test_normalize(self):
        graph = mockito.mock(nx.MultiGraph)
        edgesData, nodes, nodeList = utils.prepareNodesAndEdges()
        sortedNodes = sorted(nodes, key=lambda node: node.id)
        res = np.ndarray(shape=(5, 2))
        res[0] = [0, 0]
        res[1] = [0, 0]
        res[2] = [0, 0]
        res[3] = [0, 0]
        res[4] = [2, 2]
        mockito.when(graph).edges_iter(mockito.any(), data=True)\
            .thenReturn(utils.generateEdges(10, nodeList, edgesData))
        adjacency_matrix = nx.adjacency_matrix(graph, sortedNodes, weight=None)
        tiny = np.finfo(np.double).tiny
        instances = [4]
        self.lbp.normalize(adjacency_matrix, instances, tiny, res)
        assert res[4].tolist() == [float(2)/float(3), float(2)/float(3)]
