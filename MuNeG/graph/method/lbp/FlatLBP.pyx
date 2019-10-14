'''
Created on 18 mar 2014

@author: Adek
'''
import time

import numpy as np
from graph.method.lbp.LoopyBeliefPropagation import LoopyBeliefPropagation
cimport graph.method.lbp.CrossValMethods as crossValMethods
cimport graph.method.lbp.LBPTools as tool
cimport graph.method.common.CommonUtils as commonUtils
DTYPE = np.int
ctypedef np.int_t DTYPE_t
DOUBTYPE = np.float
ctypedef np.float_t DOUBTYPE_t
from graph.method.common.XValWithSampling import XValMethods
cdef class FlatLBP:

    def __cinit__(self):
        '''
        Constructor
        '''
        
    cpdef start(self, graph, int nrOfNodes, np.ndarray defaultClassMat, int nrOfClasses, int lbpSteps, float lbpThreshold, int numberOfFolds,
                        float percentOfTrainignNodes, int method_type, list folds):
        cdef list fold_sum = []
        cdef int i
        for i in range(1,nrOfNodes+1,1):
            fold_sum.append([i,0,0])
        
        
        cdef int fold_number = 1
        cdef list items = graph.nodes() if method_type == 1 else range(nrOfNodes)
        cdef float timer = time.time()
        
        cdef np.ndarray[DTYPE_t, ndim=1] fuz_mean_occ = np.array([], dtype=DTYPE)

        cdef method = crossValMethods.CrossValMethods()
        #TODO add tzpe when changed to cdef class
        lbp = LoopyBeliefPropagation()
        cdef tool.LBPTools tools = tool.LBPTools(nrOfNodes, graph, defaultClassMat, lbpSteps, lbpThreshold, percentOfTrainignNodes)
        common = commonUtils.CommonUtils()
        x_val_methods = XValMethods(graph)
        x_val = x_val_methods.stratifies_x_val if method_type == 1 else common.k_fold_cross_validation
        fold_sum, avg_homogenity = tools.crossVal(items, numberOfFolds, graph, nrOfNodes,
                       defaultClassMat, lbpSteps, lbpThreshold,
                       x_val, folds, common.prepareFoldClassMat,
                       lbp.lbp, None, method.flatCrossVal, False, percentOfTrainignNodes, None, None, None)
        
        cdef list foldSumEstimated = tools.prepareToEvaluate(fold_sum, nrOfClasses)
        return fold_sum, foldSumEstimated, avg_homogenity