__author__ = 'Adrian'
import unittest
import mockito
import networkx as nx
from graph.method.lbp.FlatLBP import FlatLBP
from graph.evaluation.EvaluationTools import EvaluationTools
from tests.utils.TestUtils import TestUtils

NUMBER_OF_ELEMENTS = 5

class TestsFlatLBP(unittest.TestCase):

    methods = FlatLBP()
    ev = EvaluationTools()
    utils = TestUtils()

    def test_prepareFoldClassMat(self):
        #given
        graph = mockito.mock(nx.MultiGraph)
        defaultClassMat = self.utils.prepareTestClassMat()
        validation = [4]
        edgesData, nodes, nodeList = self.utils.prepareNodesAndEdges()
        #when
        mockito.when(graph).nodes().thenReturn(nodes)
        mockito.when(graph).edges_iter(mockito.any(), data=True)\
            .thenReturn(self.utils.generateEdges(10, nodeList, edgesData))

        classMat, adjMat, sortedNodes = self.methods.prepareFoldClassMat(graph, defaultClassMat, validation)
        #then
        mockito.verify(graph).nodes()
        mockito.verify(graph).edges_iter(mockito.any(), data=True)
        assert classMat[0].tolist() == [0, 1]
        assert classMat[1].tolist() == [1, 0]
        assert classMat[2].tolist() == [0, 1]
        assert classMat[3].tolist() == [1, 0]
        assert classMat[4].tolist() == [0.5, 0.5]

    def test_startFlatLBP(self):
        #given
        defaultClassMat,\
        edgesData,\
        graph,\
        lbpLoops,\
        lbpThreshold,\
        nodeList,\
        nodes,\
        nrOfClasses,\
        nrOfFolds,\
        nrOfNodes,\
        percentOfKnownNodes = self.prepareExperimentData()
        #when
        mockito.when(graph).nodes().thenReturn(nodes)
        mockito.when(graph).edges_iter(mockito.any(), data=True)\
            .thenReturn(self.utils.generateEdges(10, nodeList, edgesData))\
            .thenReturn(self.utils.generateEdges(10, nodeList, edgesData))\
            .thenReturn(self.utils.generateEdges(10, nodeList, edgesData))\
            .thenReturn(self.utils.generateEdges(10, nodeList, edgesData))\
            .thenReturn(self.utils.generateEdges(10, nodeList, edgesData))
        #then
        foldSumEstimated = self.methods.start(graph, nrOfFolds, defaultClassMat,
                                              nrOfClasses, lbpLoops, lbpThreshold, nrOfNodes, percentOfKnownNodes)
        mockito.verify(graph, times=5).nodes()
        mockito.verify(graph, times=5).edges_iter(mockito.any(), data=True)
        assert foldSumEstimated.__len__() == NUMBER_OF_ELEMENTS
        assert 0 in foldSumEstimated or 1 in foldSumEstimated
        assert 0.5 not in foldSumEstimated


    def prepareExperimentData(self):
        graph = mockito.mock(mockito.mock(nx.MultiGraph))
        defaultClassMat = self.utils.prepareTestClassMat()
        edgesData, nodes, nodeList = self.utils.prepareNodesAndEdges()
        nrOfFolds = 5
        nrOfClasses = 2
        lbpLoops = 1000
        lbpThreshold = 0.01
        nrOfNodes = 5
        percentOfKnownNodes = 0.2
        return defaultClassMat, edgesData, graph, lbpLoops, lbpThreshold, nodeList, nodes, nrOfClasses, nrOfFolds, nrOfNodes, percentOfKnownNodes