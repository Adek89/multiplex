import unittest
import mockito as mockito

import networkx as nx
import numpy as np
from graph.method.lbp.CrossValMethods import CrossValMethods
from graph.method.common.CommonUtils import CommonUtils
from graph.method.lbp.FlatLBP import FlatLBP
from graph.method.lbp.LoopyBeliefPropagation import LoopyBeliefPropagation
from graph.gen.Node import Node
from graph.gen.Group import Group


__author__ = 'Adrian'

class TestStringMethods(unittest.TestCase):


    methods = CrossValMethods()

    def prepareNodesAndEdges(self):
        groupRed = Group('r', 1)
        groupBlue = Group('b', 2)
        node1 = Node(groupRed, 1, 0)
        node2 = Node(groupBlue, 0, 1)
        node3 = Node(groupRed, 1, 2)
        node4 = Node(groupBlue, 0, 3)
        node5 = Node(groupBlue, 1, 4)
        nodes = ({node1, node2, node3, node4, node5})
        nodeList = [node1, node2, node3, node4, node5]
        edge1 = dict([('layer', 'L1'), ('conWeight', 0.5), ('weight', 1)])
        edge2 = dict([('layer', 'L2'), ('conWeight', 0.5), ('weight', 2)])
        edgesData = ([edge1, edge2])
        return edgesData, nodes, nodeList

    def test_flatCrossVal(self):
        #given
        graph, \
        nrOfNodes, \
        adjMatPrep, \
        defaultClassMat, \
        isRandomWalk, \
        items, \
        commonUtils, \
        layerWeights, \
        lbp, \
        lbpSteps, \
        lbpThreshold, \
        nrOfFolds, \
        percentOfKnownNodes, \
        prepareClassMat, \
        prepareLayers, \
        flatLBP = self.prepareExperimentData()
        edges, nodes, nodesList = self.prepareNodesAndEdges()

        mockito.when(graph).edges_iter(mockito.any(), data=True).thenReturn(self.generateEdges(10, nodesList, edges)).thenReturn(self.generateEdges(10, nodesList, edges))
        mockito.when(graph).nodes().thenReturn(nodes)
        fold_sum = self.methods.flatCrossVal(items, nrOfFolds, graph, nrOfNodes, defaultClassMat, lbpSteps, lbpThreshold, commonUtils.k_fold_cross_validation,
                                  flatLBP.prepareFoldClassMat, lbp.lbp, layerWeights, isRandomWalk, percentOfKnownNodes, adjMatPrep, prepareLayers, prepareClassMat)
        print(fold_sum)

    def generateEdges(self, nrOfEdges, nodes, edgesData):
            i = 0
            for i in range(0, nrOfEdges):
                if (i == 0):
                    yield (nodes[0], nodes[2], edgesData[0])
                elif (i == 1):
                    yield (nodes[1], nodes[3], edgesData[0])
                elif (i == 2):
                    yield (nodes[3], nodes[4], edgesData[0])
                elif (i == 3):
                    yield (nodes[0], nodes[1], edgesData[0])
                elif (i == 4):
                    yield (nodes[2], nodes[3], edgesData[0])
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

    def prepareTestClassMat(self):
        defaultClassMat = np.ndarray(shape=(5, 2))
        defaultClassMat[0] = [0, 1]
        defaultClassMat[1] = [1, 0]
        defaultClassMat[2] = [0, 1]
        defaultClassMat[3] = [1, 0]
        defaultClassMat[4] = [0, 1]
        return defaultClassMat

    def prepareExperimentData(self):
        items = [1, 0, 1, 0, 1]
        defaultClassMat = self.prepareTestClassMat()
        nrOfFolds = 5
        graph = mockito.mock(nx.MultiGraph)
        nrOfNodes = items.__len__()
        lbpSteps = 2
        lbpThreshold = 0.001
        commonUtils = CommonUtils()
        flatLBP = FlatLBP()
        lbp = LoopyBeliefPropagation()
        layerWeights = [1, 2]
        isRandomWalk = False
        percentOfKnownNodes = 0.2
        adjMatPrep = None
        prepareLayers = None
        prepareClassMat = None
        return graph, nrOfNodes, adjMatPrep, defaultClassMat, isRandomWalk, items, commonUtils, layerWeights, lbp, lbpSteps, lbpThreshold, nrOfFolds, percentOfKnownNodes, prepareClassMat, prepareLayers, flatLBP

