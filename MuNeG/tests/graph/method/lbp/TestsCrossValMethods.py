import unittest
import mockito as mockito

import networkx as nx
import numpy as np
from bin.graph.method.lbp.CrossValMethods import CrossValMethods
from bin.graph.method.common.CommonUtils import CommonUtils
from bin.graph.method.lbp.FlatLBP import FlatLBP
from bin.graph.method.lbp.LoopyBeliefPropagation import LoopyBeliefPropagation


__author__ = 'Adrian'

class TestStringMethods(unittest.TestCase):


    methods = CrossValMethods()

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
        mockito.when(graph).nodes().thenReturn([])
        fold_sum = self.methods.flatCrossVal(items, nrOfFolds, graph, nrOfNodes, defaultClassMat, lbpSteps, lbpThreshold, commonUtils.k_fold_cross_validation,
                                  flatLBP.prepareFoldClassMat, lbp, layerWeights, isRandomWalk, percentOfKnownNodes, adjMatPrep, prepareLayers, prepareClassMat)
        print(fold_sum)

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

