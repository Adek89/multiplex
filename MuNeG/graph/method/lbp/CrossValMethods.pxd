cimport numpy as np
cdef class CrossValMethods:
    '''
    classdocs
    '''
    cpdef tuple flatCrossVal(self, list items, int numberOfFolds, graph, int nrOfNodes,
                      defaultClassMat, int lbpSteps, float lbpThreshold, object k_fold_cross_validation, folds,
                      object separationMethod, object lbp, list layerWeights, isRandomWalk,float percentOfKnownNodes, object adjMatPrep,
                      object prepareLayers, object prepareClassMat)

    cpdef multiLayerCrossVal(self, list items, int numberOfFolds, graph, int nrOfNodes,
                     np.ndarray defaultClassMat, int lbpSteps, float lbpThreshold, k_fold_cross_validation, folds,
                     separationMethod, lbp, list layerWeights, isRandomWalk, float percentOfKnownNodes, prepareAdjMat, prepareLayers, prepareClassMat)

