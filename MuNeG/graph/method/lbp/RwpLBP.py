'''
Created on 28 mar 2014

@author: MKulisiewicz
'''

import copy

import numpy as np
import scipy.sparse as sp

from graph.method.lbp import CrossValMethods
from graph.method.lbp.LBPTools import LBPTools
from graph.method.common.CommonUtils import CommonUtils
from graph.method.common.XValWithSampling import XValMethods


class RwpLBP:
    
    
    def __init__(self):
        pass
    
    def start(self, graph, defaultClassMat, nrOfClasses, nrOfNodes, nrOfFolds, lbpMaxSteps, lbpThreshold, layerWeights, percentOfKnownNodes):
        np.set_printoptions(threshold=np.nan, linewidth= np.nan, precision = 2)
        
        training = range(0,10)  #training set should be changing by cross validation
        
        adjTransMatrixes = []   #adjetency matrixes for each layer
        results = []    #
        nrOfLayers = layerWeights.__len__()
        tools = LBPTools(nrOfNodes, graph, defaultClassMat, lbpMaxSteps, lbpThreshold, percentOfKnownNodes)
        items = graph.nodes()
        
        # print "start class matrix"
        # print defaultClassMat.T
        
        method = CrossValMethods.CrossValMethods()
        common = CommonUtils()
        x_val_methods = XValMethods(graph)
        fold_sum, fuz_mean_occ, sum = tools.crossVal(items, nrOfFolds, graph, nrOfNodes, 
                       defaultClassMat, lbpMaxSteps, lbpThreshold, 
                       x_val_methods.stratifies_x_val, tools.giveCorrectData,
                       self.propagation, layerWeights, method.multiLayerCrossVal, True, percentOfKnownNodes, self.prepare_adjetency_matrix, tools.separate_layer, tools.prepareClassMatForFold)

        fusion_mean = copy.deepcopy(sum)
        for iter in range(0, len(sum)):
            # print sum[iter][1],fuz_mean_occ[iter]
            fusion_mean[iter][1]=sum[iter][1]/fuz_mean_occ[iter]
            fusion_mean[iter][2]=sum[iter][2]/fuz_mean_occ[iter]
        
        foldSumEstimated = tools.prepareToEvaluate(fold_sum, nrOfClasses)
        fusionMeanEstimated = tools.prepareToEvaluate(fusion_mean, nrOfClasses)
        
        return foldSumEstimated, fusionMeanEstimated
    

    def propagation(self, adjTransMatrixes, results, classMat, training, lbpMaxSteps, lbpThreshold):
        step = 1
        res_in = classMat
        isStopCondition = False
        while not isStopCondition:
            
            for adjMat in adjTransMatrixes:
                res = self.lbp(adjMat, res_in, training)
                results.append(res)
                
            res = self.results_fusion(results)
            
#             print res.T
            
            if self.stopConditionReached(res-res_in, lbpThreshold, step, lbpMaxSteps):
                isStopCondition = True
            
            res_in = res
            step += 1
        return res
    
    def results_fusion(self,results):
        
        res = results[0]
        count = 0
#         print "00000"
#         print results[0].T
        for result in results:
#             print "part_res: "
#             print result.T
            res = res + result
            count += 1
        res = res - results[0]
        res = res / count
        # print "full_res: "
        # print res.T
        return res
    
    def prepare_adjetency_matrix(self, adjMat,g):
        edgelist = g.edges(data=True)
#         print edgelist
        
        for edge in edgelist:
            # n1_id = vars(edge[0])['id']
            # n2_id = vars(edge[1])['id']
            n1_id = edge[0].id
            n2_id = edge[1].id
            weightToSet = edge[2]['conWeight']
#             print type(adjMat)
            adjMat[n1_id,n2_id]=weightToSet
#             print 'edge: ',vars(edge[0])['id'],',',vars(edge[1])['id'],' weight:',edge[2]['conWeight']
            pass
        
        adjMat = np.transpose(adjMat)
        
#         print "Before: ", adjMat
        adjMat = self.normalize_rows(adjMat)
#         print "After: ", adjMat
#         print adjMat.shape
        return adjMat
    
    def normalize_rows(self,adjMat):
        i = 0
        new_adjMat = adjMat[0].copy()
        for row in adjMat:
            row_sum = row.sum()
            if row_sum>0:
                j = 0
                new_row= row/row_sum
                
                new_adjMat = sp.vstack([new_adjMat,new_row])
            else:
                new_adjMat = sp.vstack([new_adjMat,row])
            i+=1
        new_adjMat = self.delete_row_csr(new_adjMat.tocsr(), 0)
        return new_adjMat

    def delete_row_csr(self, mat, i):
        if not isinstance(mat, sp.csr_matrix):
            raise ValueError("works only for CSR format -- use .tocsr() first")
        n = mat.indptr[i+1] - mat.indptr[i]
        if n > 0:
            mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i+1]:]
            mat.data = mat.data[:-n]
            mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i+1]:]
            mat.indices = mat.indices[:-n]
        mat.indptr[i:-1] = mat.indptr[i+1:]
        mat.indptr[i:] -= n
        mat.indptr = mat.indptr[:-1]
        mat._shape = (mat._shape[0]-1, mat._shape[1])
        return mat
    
    def lbp(self, adjMat, classMat, trainingInstances):
        res_in = classMat
        res = adjMat.dot(res_in)
        res[trainingInstances,:]=classMat[trainingInstances,:]
        return res
    
    def stopConditionReached(self, delta, epsilon, step, maxSteps):
        absVal = np.abs(delta)
        maxRow = np.max(absVal, 0)
        maxValue = np.max(maxRow)
        return (maxValue<epsilon or step>=maxSteps) 