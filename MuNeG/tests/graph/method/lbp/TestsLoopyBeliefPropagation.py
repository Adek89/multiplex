__author__ = 'Adrian'
import unittest
import networkx as nx
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
        sortedNodes = sorted(nodes)
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

    def test_stopConditionReached(self):
        pass
        # self.lbp.stopConditionReached()
