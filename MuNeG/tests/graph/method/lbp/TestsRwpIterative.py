__author__ = 'Adrian'
import unittest
import mockito
import networkx as nx

from graph.evaluation.EvaluationTools import EvaluationTools
from tests.utils.TestUtils import TestUtils
from graph.method.lbp.LBPTools import LBPTools

from graph.method.lbp.RwpIterative import RwpIterative

utils = TestUtils()
evaluation = EvaluationTools()

class TestsRwpLBP(unittest.TestCase):




    def test_random_walk(self):
         #given
        graph = mockito.mock(nx.MultiGraph)

        edgesData, nodes, nodeList = utils.prepareNodesAndEdges()
        edgesList = utils.prepareEdgesList(edgesData, nodeList)
        class_mat = utils.prepareTestClassMatWithUnknownNodes()
        default_class_mat = utils.prepareTestClassMat()
        tools = LBPTools(nodes.__len__(), graph, default_class_mat, 100, 0.1, 0.2)

        mockito.when(graph).edges(data=True).thenReturn(edgesList)
        mockito.when(graph).nodes().thenReturn(nodes)
        tools.separate_layer(graph, [1, 2], default_class_mat, [4])
        method = RwpIterative(tools.graphs)
        results = method.random_walk(graph, class_mat, 100, 10)
        pass
