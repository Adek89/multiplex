__author__ = 'Adrian'
import unittest
import networkx as nx
import graph.reader.syntethic.MuNeGGraphReader as reader
from graph.gen.GraphGenerator import GraphGenerator
class TestsDanioRerioReader(unittest.TestCase):


    def generate_graph(self):
        gg = GraphGenerator(100, 5, [1, 2], 9, 9, 1, ["L1", "L2"])
        graph = gg.generate()
        nx.write_gml(graph, "test.gml")
        return graph

    def assert_nodes(self, generated_nodes, readed_nodes):
        for node in readed_nodes:
            searched_node = filter(lambda n: n.id == node.id, generated_nodes)[0]
            assert node.id == searched_node.id
            assert node.label == searched_node.label
            node_group = node.group
            searched_group = searched_node.group
            assert node_group.gType == searched_group.gType
            assert node_group.gNumber == searched_group.gNumber

    def check_nodes(self, graph, readed_graph):
        generated_nodes = graph.nodes()
        readed_nodes = readed_graph.nodes()
        assert generated_nodes.__len__() == readed_nodes.__len__()
        self.assert_nodes(generated_nodes, readed_nodes)

    def check_edges(self, graph, readed_graph):
        readed_edges = readed_graph.edges(data=True)
        generated_edges = graph.edges(data=True)
        assert generated_edges.__len__() == readed_edges.__len__()
        for edge in readed_edges:
            check_edges = lambda e: (e[0].id == edge[0].id or e[0].id == edge[1].id) and (
                e[1].id == edge[1].id or e[1].id == edge[0].id) and (
                                    e[2]['layer'] == edge[2]['layer'] and e[2]['weight'] == edge[2]['weight'] and e[2][
                                        'conWeight'] == edge[2]['conWeight'])
            searched_edges = filter(check_edges, generated_edges)
            assert len(searched_edges) == 1

    def test_read(self):
        graph = self.generate_graph()
        readed_graph = reader.read_from_gml("", "test.gml")
        self.check_nodes(graph, readed_graph)
        self.check_edges(graph, readed_graph)
