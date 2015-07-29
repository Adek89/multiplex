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
        node = Node(groupRed, 1, 0)
        neighbor = Node(groupBlue, 1, 0)
        nodes = ({node, neighbor})
        data = dict([('layer', 'L1'), ('conWeight', 0.5), ('weight', 1)])
        return data, neighbor, node, nodes

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
        data, neighbor, node, nodes = self.prepareNodesAndEdges()

        mockito.when(graph).edges_iter(mockito.any(), data=True).thenReturn(self.generateEdges(2, node, neighbor, data))
        mockito.when(graph).nodes().thenReturn(nodes)
        fold_sum = self.methods.flatCrossVal(items, nrOfFolds, graph, nrOfNodes, defaultClassMat, lbpSteps, lbpThreshold, commonUtils.k_fold_cross_validation,
                                  flatLBP.prepareFoldClassMat, lbp.lbp, layerWeights, isRandomWalk, percentOfKnownNodes, adjMatPrep, prepareLayers, prepareClassMat)
        print(fold_sum)

    def generateEdges(self, nrOfEdges, node, neighbor, data):
            i = 0
            for i in range(0, nrOfEdges):
                yield (node, neighbor, data)

    def prepareExperimentData(self):
        items = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        defaultClassMat = np.ndarray(shape=(10, 2))
        for i in range(0, 10, 2):
            defaultClassMat[i] = [0, 1]
            defaultClassMat[i + 1] = [1, 0]
        nrOfFolds = 2
        graph = mockito.mock(nx.MultiGraph)
        nrOfNodes = items.__len__()
        lbpSteps = 2
        lbpThreshold = 0.1
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

