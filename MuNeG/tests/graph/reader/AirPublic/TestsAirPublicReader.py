__author__ = 'Adrian'
import unittest
from graph.reader.AirPublic.AirPublicReader import AirPublicReader

class TestsAirPublicReader(unittest.TestCase):

    def test_read(self):
        reader = AirPublicReader()
        reader.read()
        graph = reader.graph
        pass