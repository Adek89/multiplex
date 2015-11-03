__author__ = 'Adrian'
import unittest
import networkx as nx
import mockito as mockito

from tests.utils.TestUtils import TestUtils
from graph.method.common.XValWithSampling import XValMethods


class TestsXValWithSampling(unittest.TestCase):

    graph = nx.MultiGraph()
    utils = TestUtils()

    def test_loading(self):
        edges, nodes, nodesList = self.utils.prepareNodesAndEdges()
        edgesList = self.utils.prepareEdgesList(edges, nodesList)

        self.graph.add_nodes_from(nodesList)
        self.graph.add_edges_from(edgesList)

        #mix more clustering then in default data
        self.graph.add_edge(nodesList[1], nodesList[2], layer='L1', conWeight=0.5, weight=1)

        methods = XValMethods(self.graph)

        methods.loading_whole_graph()
