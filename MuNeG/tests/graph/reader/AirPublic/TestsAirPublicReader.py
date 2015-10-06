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

    def test_classes(self):
        reader = AirPublicReader()
        reader.read()
        graph = reader.graph
        edges = graph.edge
        i = 0
        for name, values in edges.items():
            if values.__len__() > 5:
                i = i + 1
        assert i == 202

    def test_assign_classes(self):
        reader = AirPublicReader()
        reader.read()
        reader.calculate_classes()
        graph = reader.graph
        i = 0
        for node in graph.nodes():
            if node.label == 1:
                i = i +1
        assert i == 202



