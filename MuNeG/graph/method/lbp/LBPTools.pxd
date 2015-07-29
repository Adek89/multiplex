cimport numpy as np
cimport graph.gen.Node as n
cdef class LBPTools:
    '''
    classdocs
    '''
    cdef int nrOfNodes
    cdef graph
    cdef np.ndarray defaultClassMat
    cdef int lbpMaxSteps
    cdef float lbpThreshold

    cdef int percentOfTrainingNodes
    cdef dict folds
    cdef dict adjMats
    cdef dict nodes
    cdef dict classMats
    cdef dict graphs
    cdef dict rests

    cpdef crossVal(self, list items, int numberOfFolds, graph, int nrOfNodes,
                     np.ndarray defaultClassMat, int lbpSteps, float lbpThreshold,
                     k_fold_cross_validation, separationMethod, lbp, list layerWeights, crossValMethod, isRandomWalk, percentOfKnownNodes, adjMarPrep, prepareLayers, prepareClassMat)
    cdef list prepareUnobservdRow(self, int nrOfClasses)
    cdef list prepareEmptyRow(self, int nrOfClasses)
    cpdef giveCorrectData(self, int label)
    cpdef np.ndarray giveCorrectClassMat(self, int label)
    cdef void addToGraph(self, g, n0, n1, set nodes, np.ndarray classMat, list training, int nrOfClasses, edge)
    cpdef prepareClassMatForFold(self, int layer, list training)
    cpdef separate_layer(self, graph, list layers, np.ndarray defaultClassMat, list training)
    cdef fillEmptyRow(self, g, set rest, set nodes, int nrOfClasses, np.ndarray classMat)
    cpdef list prepareToEvaluate(self, list lbpClassMat, int nrOfClasses)
