__author__ = 'Adrian'
import unittest
from graph.reader.AirPublic.AirPublicReader import AirPublicReader

class TestsAirPublicReader(unittest.TestCase):

    def test_read(self):
        reader = AirPublicReader()
        reader.read()
        graph = reader.graph
        nodes = graph.nodes()
        sortedNodes = sorted(nodes, key=lambda node: node.id)
        assert nodes.__len__() == 450
        assert sortedNodes[258].name == 'XXXX'