__author__ = 'Adrian'
import unittest
from graph.reader.DanioRerio.DanioRerioReader import DanioRerioReader
class TestsDanioRerioReader(unittest.TestCase):


    def test_read(self):
        reader = DanioRerioReader()
        reader.read()
        graph = reader.graph
        edges = graph.edges()
        nodes = graph.nodes()
        sortedNodes = sorted(nodes, key=lambda node: node.id)
        assert edges.__len__() == 188
        assert nodes.__len__() == 155
        assert sortedNodes[14].id == 14
        assert sortedNodes[14].name == 'nkx2.2b'


