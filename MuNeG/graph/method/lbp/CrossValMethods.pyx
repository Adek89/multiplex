'''
Created on 09.04.2014

@author: apopiel
'''
cimport cython
from graph.method.lbp.LoopyBeliefPropagation import LoopyBeliefPropagation
import numpy as np
cimport numpy as np
cimport graph.gen.Node as n
DTYPE=np.float64
ctypedef np.float64_t DTYPE_t

cdef class CrossValMethods:
    '''
    classdocs
    '''


    def __cinit__(self):
        '''
        Constructor
        '''

    cpdef list flatCrossVal(self, list items, int numberOfFolds, graph, int nrOfNodes,
                      defaultClassMat, int lbpSteps, float lbpThreshold, object k_fold_cross_validation,
                      object separationMethod, object lbp, list layerWeights, isRandomWalk, float percentOfKnownNodes, object adjMatPrep,
                      object prepareLayers, object prepareClassMat):
        cdef list fold_sum = []
        cdef int i
        for i in range(1,nrOfNodes+1,1):
            fold_sum.append([i,0,0])
        cdef int fold_number = 1
        cdef list training
        cdef list validation
        cdef list results_agregator
        cdef int num_of_res
        cdef list sum
        cdef int iter
        cdef n.Node node
        for training, validation in k_fold_cross_validation(items, numberOfFolds, percentOfKnownNodes):
            print "-----FOLD %d-----" % fold_number
            
            #separate layers
            results_agregator = []
            num_of_res = 0
            class_Mat, adjMat, nodes = separationMethod(graph, defaultClassMat, validation)
                    
                #-------------LBP----------------------
            class_mat = lbp(adjMat, class_Mat, lbpSteps, lbpThreshold, training, validation)
            
            #create zero result matrix
            sum = []
            iter = 0
            for node in nodes:
                if (node.id in validation):
                    sum.append([node.id,class_mat[iter,0],class_mat[iter,1]])
                else:  
                    sum.append([node.id,0,0])  
                iter+=1
            
            #fusion - sum
            for i in range(0,nrOfNodes,1):
                fold_sum[i][1]+=sum[i][1]
                fold_sum[i][2]+=sum[i][2]
            # print sum
            fold_number = fold_number + 1
            
        return fold_sum
        
    cpdef multiLayerCrossVal(self, list items, int numberOfFolds, graph, int nrOfNodes,
                     np.ndarray defaultClassMat, int lbpSteps, float lbpThreshold, k_fold_cross_validation,
                     separationMethod, lbp, list layerWeights, isRandomWalk, float percentOfKnownNodes, prepareAdjMat, prepareLayers, prepareClassMat):
        cdef int fold_number = 1
        cdef list fold_sum = []
        cdef list adjTransMatrixes = []
        cdef int i
        for i in range(1,nrOfNodes+1,1):
            fold_sum.append([i,0,0])
            
        cdef np.ndarray fuz_mean_occ = np.array([])
        prepareLayers(graph, layerWeights, defaultClassMat, [])
        cdef list training
        cdef list validation
        cdef list results_agregator
        cdef num_of_res
        cdef int layer_label
        cdef np.ndarray class_mat
        cdef list results
        cdef res
        cdef int iter
        cdef int node
        cdef list full_res
        cdef np.ndarray n
        cdef list sum
        cdef list result
        cdef np.ndarray adjMatPy
        cdef np.float64_t sumElement
        cdef np.ndarray rows
        cdef np.ndarray repmatInput
        for training, validation in k_fold_cross_validation(items, numberOfFolds, percentOfKnownNodes):
            print "-----FOLD %d-----" % fold_number
            
            #separate layers
            results_agregator = []
            num_of_res = 0
            
            
            for layer_label in layerWeights:
                prepareClassMat(layer_label, training)
                class_Mat, adjMat, nodes = separationMethod(layer_label)
                    
                for i in range(nrOfNodes):
                    if i >=class_Mat.shape[0]:
                        try:
                            self.training.remove(i)
                        except ValueError:
                            pass
                for i in range(nrOfNodes):
                    if i >=class_Mat.shape[0]:
                        try:
                            self.validation.remove(i)
                        except ValueError:
                            pass
                        
                if (isRandomWalk):
                    results = []
                    adjMat = prepareAdjMat(adjMat,graph)
                    adjTransMatrixes.append(adjMat)
                    
                    class_mat = lbp(adjTransMatrixes, results, class_Mat, training, lbpSteps, lbpThreshold)
                else:    
                    #-------------LBP----------------------
                    class_mat = lbp(adjMat, class_Mat, lbpSteps, lbpThreshold, training, validation)
                    
                res = []
                iter = 0
                #assign results to nodes
                for node in nodes:
                    res.append([node,class_mat[iter,0],class_mat[iter,1]])
                    iter+=1
                    
                #assign results to full node list
                res = np.asarray(sorted(res))
                full_res = []
                iter = 0
                for n in res:
                    while iter!= n[0] or iter >= nrOfNodes:
                        full_res.append([iter,0,0])
                        iter+=1
                    full_res.append([iter,n[1],n[2]])
                    iter+=1
                    
                #push out results
                results_agregator.append(full_res)
                num_of_res += 1
            
            
            #create zero result matrix
            sum = []
            for i in range(1,nrOfNodes+1,1):
                sum.append([i,0,0])
            
            #fusion - sum
            for result in results_agregator:
                for i in range(0,nrOfNodes,1):
                    sum[i][1]+=result[i][1]
                    sum[i][2]+=result[i][2]
            for i in range(0,nrOfNodes,1):
                fold_sum[i][1]+=sum[i][1]
                fold_sum[i][2]+=sum[i][2]
            # print sum
            
            #fusion - mean part1
            adjMatPy = np.array(adjMat)
            sumElement = np.finfo(np.double).tiny
            rows = adjMatPy[validation,:]      
            repmatInput = np.sum(rows, axis=1) + sumElement 
#             print "sum ",len(sum)," repmat ",len(repmatInput)
            fuz_mean_occ = np.append(fuz_mean_occ,repmatInput)
            fold_number = fold_number + 1
            
        return fold_sum, fuz_mean_occ, sum
        