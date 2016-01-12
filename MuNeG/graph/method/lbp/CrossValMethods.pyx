'''
Created on 09.04.2014

@author: apopiel
'''
cimport cython
from graph.method.lbp.LoopyBeliefPropagation import LoopyBeliefPropagation
import numpy as np
cimport numpy as np
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
        for training, validation in k_fold_cross_validation(items, numberOfFolds, percentOfKnownNodes):
            print "-----FOLD %d-----" % fold_number
            print "Training: "
            print training
            print "Validation: "
            print validation
            
            #separate layers
            results_agregator = []
            num_of_res = 0
            class_Mat, adjMat, nodes = separationMethod(graph, defaultClassMat, validation)

            print "class_mat after separation: "
            print class_Mat
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

    def prepare_sum_for_fusion_mean(self, nrOfNodes):
        sum_for_fusion_mean = []
        for j in xrange(0, nrOfNodes, 1):
            sum_for_fusion_mean.append([j, 0, 0])
        return sum_for_fusion_mean

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
        prepareLayers(graph, layerWeights, defaultClassMat)
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
        cdef np.float64_t sumElement
        cdef rows
        cdef repmatInput
        fusion_mean = self.prepare_sum_for_fusion_mean(nrOfNodes)
        for training, validation in k_fold_cross_validation(items, numberOfFolds, percentOfKnownNodes):
            print 'Multilayer method'
            print "-----FOLD %d-----" % fold_number
            print "Training: "
            print training
            print "Validation: "
            print validation
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
                if i in validation:
                    fold_sum[i][1]+=sum[i][1]
                    fold_sum[i][2]+=sum[i][2]
            # print sum

            fusion_mean = self.prepare_powered_fusion_mean(results_agregator, separationMethod, layerWeights, nrOfNodes, fusion_mean, validation)
            #fusion - mean part1
            # adjMatPy = np.array(adjMat)
            sumElement = np.finfo(np.double).tiny
            rows = adjMat[validation,:]
            repmatInput = rows.sum(axis=1) + sumElement
#             print "sum ",len(sum)," repmat ",len(repmatInput)
            fuz_mean_occ = np.append(fuz_mean_occ,repmatInput)
            fold_number = fold_number + 1
        return fold_sum, fusion_mean

    def prepare_powered_fusion_mean(self, results_agregator, separation_method, layer_weights, nr_of_nodes, fusion_mean, validation):
        i = 0
        print 'Start fusion mean: ' + str(fusion_mean)
        for results_on_layer in results_agregator:
            fuz_mean_occ = np.array([])
            layer = layer_weights[i]
            results_on_layer = filter(lambda res : res[0] in validation, results_on_layer)
            results_on_layer = sorted(results_on_layer,key=lambda row : row[0])
            print 'Results on layer variable: ' + str(results_on_layer)
            class_Mat, adjMat, nodes = separation_method(layer)
            sumElement = np.finfo(np.double).tiny
            rows = adjMat[validation,:]
            repmatInput = rows.sum(axis=1) + sumElement
            fuz_mean_occ = np.append(fuz_mean_occ,repmatInput)

            print 'Variable fuzz mean occ: ' + str(fuz_mean_occ)

            iter = 0
            for result in results_on_layer:
                node_id = result[0]
                fusion_mean[node_id][1]+=result[1]/fuz_mean_occ[iter]
                fusion_mean[node_id][2]+=result[2]/fuz_mean_occ[iter]
                iter += 1
            i += 1
            print 'Variable fusion_mean: ' + str(fusion_mean) + ' after layer: ' + str(layer)
        return fusion_mean
