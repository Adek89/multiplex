__author__ = 'Adrian'
import unittest
from graph.method.lbp.FlatLBP import FlatLBP
from graph.evaluation.EvaluationTools import EvaluationTools
class TestsFlatLBP(unittest.TestCase):

    methods = FlatLBP()
    ev = EvaluationTools()

    def test_prepareFoldClassMat(self):
        pass