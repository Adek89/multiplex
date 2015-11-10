__author__ = 'Adrian'
import unittest
import networkx as nx
import mockito as mockito

from tests.utils.TestUtils import TestUtils
from graph.method.common.XValWithSampling import XValMethods


class TestsXValWithSampling(unittest.TestCase):


    utils = TestUtils()

    def prepare_graph(self):
        graph = nx.MultiGraph()
        edges, nodes, nodesList = self.utils.prepareNodesAndEdges()
        edgesList = self.utils.prepareEdgesList(edges, nodesList)
        graph.add_nodes_from(nodesList)
        graph.add_edges_from(edgesList)
        # mix more clustering then in default data
        graph.add_edge(nodesList[1], nodesList[2], layer='L1', conWeight=0.5, weight=1)
        return graph

    def test_loading(self):
        graph = self.prepare_graph()

        methods = XValMethods(graph)

        result_matrix = methods.loading_whole_graph()

        assert result_matrix.shape == (5,3)
        assert result_matrix[1,2] == 0.5

    def test_loading_surveyed_nodes(self):
        graph = self.prepare_graph()
        methods = XValMethods(graph)
        nodes = graph.nodes()
        known_nodes = list([nodes[0], nodes[1]])
        matrix = methods.loading_surveyed_matrix(known_nodes)

        assert matrix.shape == (2,3)

    def test_sampling(self):
        graph = self.prepare_graph()

        methods = XValMethods(graph)
        nodes = graph.nodes()
        known_nodes = list([nodes[0], nodes[1]])
        sorted_ids = methods.create_sample_list(1)

        assert sorted_ids == [3, 4, 0, 2, 1]

    def test_stratified_x_val(self):
        graph = self.prepare_graph()
        methods = XValMethods(graph)
        nodes = graph.nodes()
        i = 0
        for train_index, test_index in methods.stratifies_x_val(nodes, 5):
            if i == 0:
                assert test_index == [0,1]
            elif i == 1:
                assert test_index == [2, 3]
            elif i == 2:
                assert test_index == [4]
            else:
                assert test_index == []
            i = i + 1


