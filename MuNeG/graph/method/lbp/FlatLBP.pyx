'''
Created on 18 mar 2014

@author: Adek
'''
import time
import numpy as np
cimport numpy as np
import networkx as nx
import csv
from random import shuffle
from graph.method.lbp.LoopyBeliefPropagation import LoopyBeliefPropagation
cimport graph.method.lbp.CrossValMethods as crossValMethods
cimport graph.method.lbp.LBPTools as tool
cimport graph.method.common.CommonUtils as commonUtils
DTYPE = np.int
ctypedef np.int_t DTYPE_t
DOUBTYPE = np.float
ctypedef np.float_t DOUBTYPE_t
cdef class FlatLBP:

    def __cinit__(self):
        '''
        Constructor
        '''
    
    #TODO Add np.ndarray types when parameters will be passed with types, and method could be changes to cdef
    cpdef prepareFoldClassMat(self, graph, np.ndarray  defaultClassMat, list validation):
        cdef np.ndarray classMat = defaultClassMat.copy()
        cdef int nrOfClasses = classMat.shape[1]
        cdef int i
        cdef list row
        cdef list sortedNodes
        cdef np.ndarray[DOUBTYPE_t, ndim=2] adjMat
        for i in range(0, defaultClassMat.__len__()):
            if i in validation:
                row = self.prepareUnobservdRow(nrOfClasses)
                classMat[i] = row
                
        sortedNodes = sorted(graph.nodes())
        adjMat = nx.adjacency_matrix(graph, sortedNodes)
        
        return classMat, adjMat, sortedNodes
        
    cpdef list start(self, graph, int nrOfNodes, np.ndarray defaultClassMat, int nrOfClasses, int lbpSteps, float lbpThreshold, int numberOfFolds,
                        float percentOfTrainignNodes):
        cdef list fold_sum = []
        cdef int i
        for i in range(1,nrOfNodes+1,1):
            fold_sum.append([i,0,0])
        
        
        cdef int fold_number = 1
        cdef list items = range(nrOfNodes)
        cdef float timer = time.time()
        
        cdef np.ndarray[DTYPE_t, ndim=1] fuz_mean_occ = np.array([], dtype=DTYPE)

        cdef method = crossValMethods.CrossValMethods()
        #TODO add tzpe when changed to cdef class
        lbp = LoopyBeliefPropagation()
        cdef tool.LBPTools tools = tool.LBPTools(nrOfNodes, graph, defaultClassMat, lbpSteps, lbpThreshold, percentOfTrainignNodes)
        common = commonUtils.CommonUtils()

        fold_sum = tools.crossVal(items, numberOfFolds, graph, nrOfNodes, 
                       defaultClassMat, lbpSteps, lbpThreshold, 
                       common.k_fold_cross_validation, self.prepareFoldClassMat,
                       lbp.lbp, None, method.flatCrossVal, False, percentOfTrainignNodes, None, None, None)
        
        cdef list foldSumEstimated = tools.prepareToEvaluate(fold_sum, nrOfClasses)
        return foldSumEstimated
        
        
    cdef list prepareUnobservdRow(self, int nrOfClasses):
        cdef list row = []
        cdef int i
        for i in range(0, nrOfClasses):
            row.append(0.5)
        return row