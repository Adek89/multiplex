import unittest
from bin.graph.method.lbp.FlatLBP import FlatLBP
from bin.graph.method.lbp.CrossValMethods import CrossValMethods


__author__ = 'Adrian'

class TestStringMethods(unittest.TestCase):



    def test_flatCrossVal(self):
        methods = CrossValMethods()