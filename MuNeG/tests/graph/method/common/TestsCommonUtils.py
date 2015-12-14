__author__ = 'Adrian'
import unittest
import mockito
import networkx as nx
from graph.method.common.CommonUtils import CommonUtils
from tests.utils.TestUtils import TestUtils
class TestsXValWithSampling(unittest.TestCase):

    methods = CommonUtils()
    utils = TestUtils()

    def test_prepareFoldClassMat(self):
        #given
        graph = mockito.mock(nx.MultiGraph)
        defaultClassMat = self.utils.prepareTestClassMat()
        validation = [4]
        edgesData, nodes, nodeList = self.utils.prepareNodesAndEdges()
        #when
        mockito.when(graph).nodes().thenReturn(nodes)
        mockito.when(graph).edges_iter(mockito.any(), data=True)\
            .thenReturn(self.utils.generateEdges(10, nodeList, edgesData))

        classMat, adjMat, sortedNodes = self.methods.prepareFoldClassMat(graph, defaultClassMat, validation)
        #then
        mockito.verify(graph).nodes()
        mockito.verify(graph).edges_iter(mockito.any(), data=True)
        assert classMat[0].tolist() == [0, 1]
        assert classMat[1].tolist() == [1, 0]
        assert classMat[2].tolist() == [0, 1]
        assert classMat[3].tolist() == [1, 0]
        assert classMat[4].tolist() == [0.5, 0.5]