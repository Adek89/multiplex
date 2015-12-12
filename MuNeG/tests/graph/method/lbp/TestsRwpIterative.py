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
        tools = LBPTools(nodes.__len__(), graph, default_class_mat, 100, 0.1, 0.8)

        mockito.when(graph).edges_iter(nodeList[4], data=True)\
            .thenReturn(utils.generateEdges(10, nodeList, edgesData))\
            .thenReturn(utils.generateEdges(10, nodeList, edgesData))\
            .thenReturn(utils.generateEdges(10, nodeList, edgesData))\
            .thenReturn(utils.generateEdges(10, nodeList, edgesData))\
            .thenReturn(utils.generateEdges(10, nodeList, edgesData))
        mockito.when(graph).edges(data=True).thenReturn(edgesList)
        mockito.when(graph).nodes().thenReturn(nodes)
        tools.separate_layer(graph, [1, 2], default_class_mat, [4])
        method = RwpIterative(tools.graphs)
        results = method.random_walk(graph, class_mat, 1, 100, 10)
        pass

    def generateEdges(self, nrOfEdges, nodes, edgesData):
            i = 0
            for i in range(0, nrOfEdges):
                if (i == 0):
                    yield (nodes[0], nodes[2], edgesData[0])
                elif (i == 1):
                    yield (nodes[1], nodes[3], edgesData[0])
                elif (i == 2):
                    yield (nodes[2], nodes[4], edgesData[0])
                elif (i == 3):
                    yield (nodes[0], nodes[1], edgesData[0])
                elif (i == 4):
                    yield (nodes[1], nodes[4], edgesData[0])
                elif (i == 5):
                    yield (nodes[0], nodes[2], edgesData[1])
                elif (i == 6):
                    yield (nodes[1], nodes[3], edgesData[1])
                elif (i == 7):
                    yield (nodes[1], nodes[4], edgesData[1])
                elif (i == 8):
                    yield (nodes[0], nodes[1], edgesData[1])
                elif (i == 9):
                    yield (nodes[2], nodes[3], edgesData[1])