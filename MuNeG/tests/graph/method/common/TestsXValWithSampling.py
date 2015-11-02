__author__ = 'Adrian'
import unittest
import networkx as nx
import mockito as mockito

from tests.utils.TestUtils import TestUtils
from graph.method.common.XValWithSampling import XValMethods


class TestsXValWithSampling(unittest.TestCase):

    graph = mockito.mock(nx.MultiGraph)
    utils = TestUtils()

    def test_loading(self):
        edges, nodes, nodesList = self.utils.prepareNodesAndEdges()
        edgesList = self.utils.prepareEdgesList(edges, nodesList)

        mockito.when(self.graph).edges_iter(mockito.any(), data=True)\
            .thenReturn(self.utils.generateEdges(10, nodesList, edges))\
            .thenReturn(self.utils.generateEdges(10, nodesList, edges))\
            .thenReturn(self.utils.generateEdges(10, nodesList, edges))\
            .thenReturn(self.utils.generateEdges(10, nodesList, edges))\
            .thenReturn(self.utils.generateEdges(10, nodesList, edges))
        mockito.when(self.graph).nodes().thenReturn(nodes)
        mockito.when(self.graph).edges(data=True).thenReturn(edgesList)
        mockito.when(self.graph).degree_iter().thenReturn(iter([(0,4)]))

        methods = XValMethods(self.graph)

        methods.loading()
