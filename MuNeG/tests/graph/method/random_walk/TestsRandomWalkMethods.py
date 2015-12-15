__author__ = 'Adrian'
import unittest
import mockito
import networkx as nx
from graph.method.random_walk.RandomWalkMethods import RandomWalkMethods
from graph.method.lbp.LBPTools import LBPTools
from tests.utils.TestUtils import TestUtils


utils = TestUtils()
class TestsRandomWalkMethods(unittest.TestCase):


    def test_random_walk_classical(self):
        methods = RandomWalkMethods()

        #given
        graph = mockito.mock(nx.MultiGraph)

        edgesData, nodes, nodeList = utils.prepareNodesAndEdges()
        edgesList = utils.prepareEdgesList(edgesData, nodeList)
        class_mat = utils.prepareTestClassMatWithUnknownNodes()
        default_class_mat = utils.prepareTestClassMat()

        mockito.when(graph).edges_iter(mockito.any(), data=True)\
            .thenReturn(utils.generateEdges(10, nodeList, edgesData))\
            .thenReturn(utils.generateEdges(10, nodeList, edgesData))\
            .thenReturn(utils.generateEdges(10, nodeList, edgesData))\
            .thenReturn(utils.generateEdges(10, nodeList, edgesData))\
            .thenReturn(utils.generateEdges(10, nodeList, edgesData))
        mockito.when(graph).edges_iter(mockito.any())\
            .thenReturn(utils.generateEdges(10, nodeList, edgesData))\
            .thenReturn(utils.generateEdges(10, nodeList, edgesData))\
            .thenReturn(utils.generateEdges(10, nodeList, edgesData))\
            .thenReturn(utils.generateEdges(10, nodeList, edgesData))\
            .thenReturn(utils.generateEdges(10, nodeList, edgesData))
        mockito.when(graph).edges(data=True).thenReturn(edgesList)
        mockito.when(graph).nodes().thenReturn(nodes)
        result = methods.random_walk_classical(graph, default_class_mat, [1, 2], 5, 1)
        assert result.__len__() == 5

