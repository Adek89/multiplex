'''
Created on 11 mar 2014

@author: MKulisiewicz
'''

cimport numpy as np

import csv
import time

from graph.method.lbp.LoopyBeliefPropagation import LoopyBeliefPropagation
from graph.method.lbp.NetworkUtils import NetworkUtils

cimport graph.method.lbp.CrossValMethods as cvm
cimport graph.method.lbp.LBPTools as tool
cimport graph.method.common.CommonUtils as commonUtils
from graph.method.common.XValWithSampling import XValMethods
cdef class Multilayer_LBP:
    
    
    cdef list training
    cdef list validation
    
    FILE_PATH = "multilayer_decision_fusion.csv"
    
    nu = NetworkUtils()
    
    def __cinit__(self):
        pass


    cpdef start(self, graph, np.ndarray defaultClassMat, int nrOfClasses, int nrOfNodes, int nrOfFolds, int lbpMaxSteps, float lbpThreshold, list layerWeights, float percentOfTrainingNodes, int method_type):
        
        
        
        cdef int fold_number = 1
        cdef list items = graph.nodes() if method_type == 1 else range(nrOfNodes)
        timer = time.time()
        
        
        cdef method = cvm.CrossValMethods()
        lbp = LoopyBeliefPropagation()
        
        cdef tools = tool.LBPTools(nrOfNodes, graph, defaultClassMat, lbpMaxSteps, lbpThreshold, percentOfTrainingNodes)
        cdef common = commonUtils.CommonUtils()
        x_val_methods = XValMethods(graph)
        x_val = x_val_methods.stratifies_x_val if method_type == 1 else common.k_fold_cross_validation
        fold_sum, fusion_mean, fusion_mean_scores, fusion_layer, fusion_layer_scores, fusion_random, fusion_convergence_max, fusion_convergence_min, layer_results, rwp = tools.crossVal(items, nrOfFolds, graph, nrOfNodes,
                       defaultClassMat, lbpMaxSteps, lbpThreshold, 
                       x_val, tools.giveCorrectData,
                       lbp.lbp, layerWeights, method.multiLayerCrossVal, False, percentOfTrainingNodes, None, tools.separate_layer, tools.prepareClassMatForFold)

        print 'Variable fusion mean: ' + str(fusion_mean)
        
        print "Writing a file..."
        results_sum_file = open(self.FILE_PATH+"_sum.csv","wb")
        results_mean_file = open(self.FILE_PATH+"_mean.csv","wb")
        writer_sum = csv.writer(results_sum_file)
        writer_mean = csv.writer(results_mean_file)
        writer_sum.writerows(fold_sum)
        writer_mean.writerows(fusion_mean)
        
        work_time = time.time()-timer
        print "DONE"
        print "time:  %f s" %work_time
        print 'Layer result: ' + str(layer_results)
        
        foldSumEstimated = tools.prepareToEvaluate(fold_sum, nrOfClasses)
        fusionMeanEstimated = tools.prepareToEvaluate(fusion_mean, nrOfClasses)
        fusionLayerEstimated = tools.prepareToEvaluate(fusion_layer, nrOfClasses)
        fusionRadomEstimated = tools.prepareToEvaluate(fusion_random, nrOfClasses)
        fusionConvergenceMaxEstimated = tools.prepareToEvaluate(fusion_convergence_max, nrOfClasses)
        fusionConvergenceMinEstimated = tools.prepareToEvaluate(fusion_convergence_min, nrOfClasses)
        fusionForLayersEstimated = {}
        for layer, class_mat in layer_results.iteritems():
            fusionForLayersEstimated[layer] = tools.prepareToEvaluate(class_mat, nrOfClasses)
        
        return fold_sum, fusion_mean_scores, fusion_layer_scores, fusion_random, fusion_convergence_max, fusion_convergence_min, layer_results, foldSumEstimated, fusionMeanEstimated, fusionLayerEstimated, fusionRadomEstimated, fusionConvergenceMaxEstimated, fusionConvergenceMinEstimated, fusionForLayersEstimated
                    
