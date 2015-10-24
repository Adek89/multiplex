__author__ = 'Adrian'
import unittest
from graph.reader.DanioRerio.DanioRerioReader import DanioRerioReader
#@Before
reader = DanioRerioReader()
reader.read()
class TestsDanioRerioReader(unittest.TestCase):



    def test_read(self):
        graph = reader.graph
        edges = graph.edges()
        nodes = graph.nodes()
        sortedNodes = sorted(nodes, key=lambda node: node.id)
        assert edges.__len__() == 188
        assert nodes.__len__() == 155
        assert sortedNodes[14].id == 14
        assert sortedNodes[14].name == 'nkx2.2b'
        assert sortedNodes[14].functions.__len__() == 8

    def test_create_go_term_map(self):
        map = reader.create_go_terms_map()
        max_key = max(map, key=map.get)
        value = map[max_key]
        assert value == 53
        assert max_key == 'GO:0005634'

    def test_label_assigning(self):
        reader.assign_labels('GO:0005634')
        nodes = reader.graph.nodes()
        i = 0
        for node in nodes:
            if node.label == 1:
                i = i + 1
        assert i == 53


