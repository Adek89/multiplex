'''
Created on 13.03.2014

@author: apopiel
'''
import csv
import math
import time

import networkx as nx
import sklearn.metrics as metrics

import graph.reader.syntethic.MuNeGGraphReader as reader
from graph.evaluation.EvaluationTools import EvaluationTools
from graph.method.lbp.FlatLBP import FlatLBP
from graph.method.lbp.Multilayer_LBP import Multilayer_LBP
from graph.method.lbp.NetworkUtils import NetworkUtils
from graph.method.lbp.RwpLBP import RwpLBP
from graph.method.random_walk.RandomWalkMethods import RandomWalkMethods
from graph.reader.ExcelReader import ExcelReader


class DecisionFusion:
    
    #Parameters
    REAL_NUMBER_OF_NODES = 288
    NUMBER_OF_NODES = 100
    NUMBER_OF_GROUPS = 0
    AVERAGE_GROUP_SIZE = 50
     
    LAYERS_WEIGHTS = []
    REAL_LAYERS_WEIGHTS = [1, 2, 3]
    LAYERS_NAME = []
     
    GROUP_LABEL_HOMOGENITY = 1
    PROBABILITY_OF_EDGE_EXISTANCE_IN_SAME_GROUP = 9
    PROBABILITY_OF_EDGE_EXISTANCE_BETWEEN_OTHER_GROUPS = 1
     
    NUMBER_OF_FOLDS = 5
     
    LBP_MAX_STEPS = 100
    LBP_TRESHOLD = 0.01
    
    training = []
    validation = []
    
    FILE_PATH = "..\\results\\synthetic\\"
    
    
    nu = NetworkUtils()
    training = []
    validation = []

    realGraph = nx.MultiGraph()
    synthetic = nx.MultiGraph()
    
    realGraphClassMat = []
    syntheticClassMat = []
    
    realNrOfClasses = 0
    syntheticNrOfClasses = 0
    
    syntheticFlatResult = []
    syntheticFlatScores = []
    realFlatResult = []
    
    syntheticLBPFoldSum = []
    syntheticLBPFusionMean = []
    realLBPFoldSum = []
    realLBPFusionMean = []
    
    realLabels = []
    syntheticLabels = []
    
    realRWPFoldSum = []
    realRWPFusionMean = []
    syntheticRWPFoldSum = []
    syntheticRWPFusionMean = []
    gg = None
    nrOfLayers = 0
    percentOfTrainignNodes = 0.0
    counter = 0
    syntheticRwcResult = 0.0
    rwpResult = []
    method = 1
    syntheticFusionLayer = []
    syntheticFusionRandom = []
    syntheticFusionConvergenceMax = []
    syntheticFusionConvergenceMin = []
    syntheticFusionForLayers = []

    syntheticLBPFoldSumScores = []
    syntheticLBPFusionMeanScores = []
    syntheticFusionLayerScores = []
    syntheticFusionRandomScores = []
    syntheticFusionConvergenceMaxScores = []
    syntheticFusionForLayersScores = {}

    tprs_per_method = {}
    fprs_per_method = {}


    def __init__(self, nrOfNodes, AVG_GROUP_SIZE, grLabelHomogenity, prEdgeInGroup, prEdgeBetweenGroups, nrOfLayers, folds):
        self.NUMBER_OF_NODES = nrOfNodes
        self.AVERAGE_GROUP_SIZE = AVG_GROUP_SIZE
        self.GROUP_LABEL_HOMOGENITY = grLabelHomogenity
        self.PROBABILITY_OF_EDGE_EXISTANCE_IN_SAME_GROUP = prEdgeInGroup
        self.PROBABILITY_OF_EDGE_EXISTANCE_BETWEEN_OTHER_GROUPS = prEdgeBetweenGroups
        self.initLayers(nrOfLayers)
        self.nrOfLayers = nrOfLayers
        self.NUMBER_OF_FOLDS = folds
        self.fprs_per_method = {}
        self.tprs_per_method = {}


    def initLayers(self, nrOfLayers):
        self.LAYERS_WEIGHTS = []
        self.LAYERS_NAME = []
        for i in xrange(0, nrOfLayers):
            self.LAYERS_WEIGHTS.append(i + 1)
            self.LAYERS_NAME.append("L"+str(i+1))
        
    def processExperiment(self):
        self.generateSyntheticData()
        self.preprocessing()
        self.flatLBP()
        self.multiLayerLBP()
        self.evaluation()
        return self.fprs_per_method, self.tprs_per_method
        
    '''
    Prepare data
    '''      
    def readRealData(self):
        reader = ExcelReader()
        self.realGraph = reader.read('acta_vir')

    def build_file_name(self):
        homogenity = self.GROUP_LABEL_HOMOGENITY if self.GROUP_LABEL_HOMOGENITY in [5.5, 6.5, 7.5, 8.5, 9.5] else int(math.floor(self.GROUP_LABEL_HOMOGENITY))
        group = int(self.PROBABILITY_OF_EDGE_EXISTANCE_IN_SAME_GROUP )
        between_other_groups = self.PROBABILITY_OF_EDGE_EXISTANCE_BETWEEN_OTHER_GROUPS if self.PROBABILITY_OF_EDGE_EXISTANCE_BETWEEN_OTHER_GROUPS in [0.1, 0.5] else int(self.PROBABILITY_OF_EDGE_EXISTANCE_BETWEEN_OTHER_GROUPS)
        nodes = int(round(float(self.NUMBER_OF_NODES) / 100.0) * 100)
        return 'muneg_' + str(nodes) + '_' + str(self.AVERAGE_GROUP_SIZE) + '_' + str(
            homogenity) + '_' + str(group) + '_' + str(
            between_other_groups) + '_' + str(self.nrOfLayers) + '.gml'

    def build_output_file_name(self):
        homogenity = self.GROUP_LABEL_HOMOGENITY if self.GROUP_LABEL_HOMOGENITY in [5.5, 6.5, 7.5, 8.5, 9.5] else int(math.floor(self.GROUP_LABEL_HOMOGENITY))
        group = int(self.PROBABILITY_OF_EDGE_EXISTANCE_IN_SAME_GROUP )
        between_other_groups = self.PROBABILITY_OF_EDGE_EXISTANCE_BETWEEN_OTHER_GROUPS if self.PROBABILITY_OF_EDGE_EXISTANCE_BETWEEN_OTHER_GROUPS in [0.1, 0.5] else int(self.PROBABILITY_OF_EDGE_EXISTANCE_BETWEEN_OTHER_GROUPS)
        nodes = int(round(float(self.NUMBER_OF_NODES) / 100.0) * 100)
        return 'synt_' + str(nodes) + '_' + str(self.AVERAGE_GROUP_SIZE) + '_' + str(
            homogenity) + '_' + str(group) + '_' + str(
            between_other_groups) + '_' + str(self.nrOfLayers) + '_' + str(self.NUMBER_OF_FOLDS) + '.csv'

    def generateSyntheticData(self):
        start_time = time.time()
        self.synthetic = reader.read_from_gml('..\\results', self.build_file_name(),)
        print("---generation time: %s seconds ---" % str(time.time() - start_time))
        
    '''
    Preprocessing
    '''
    def preprocessing(self):
#         self.realGraphClassMat, self.realNrOfClasses = self.nu.createClassMat(self.realGraph)   
        self.syntheticClassMat, self.syntheticNrOfClasses = self.nu.createClassMat(self.synthetic)
    '''
    Algorithms
    '''        
    def flatLBP(self):
        start_time = time.time()
        flatLBP = FlatLBP()
        fold_sum, self.syntheticFlatResult = flatLBP.start(self.synthetic, self.NUMBER_OF_NODES, self.syntheticClassMat,
                                                 self.syntheticNrOfClasses, self.LBP_MAX_STEPS, self.LBP_TRESHOLD,
                                                 self.NUMBER_OF_FOLDS, self.percentOfTrainignNodes, self.method)
        print("---flatLBP time: %s seconds ---" % str(time.time() - start_time))
        self.syntheticFlatScores = [element[2] for element in fold_sum]
#         self.realFlatResult = flatLBP.start(self.realGraph, self.REAL_NUMBER_OF_NODES, self.realGraphClassMat, self.realNrOfClasses, self.LBP_MAX_STEPS, self.LBP_TRESHOLD, self.NUMBER_OF_FOLDS)
        
        
        
    def multiLayerLBP(self):
        multiLBP = Multilayer_LBP()

        fold_sum, fusion_mean, fusion_layer, fusion_random, fusion_convergence_max, fusion_convergence_min, layer_results, self.syntheticLBPFoldSum, self.syntheticLBPFusionMean, self.syntheticFusionLayer, self.syntheticFusionRadom, self.syntheticFusionConvergenceMax, self.syntheticFusionConvergenceMin, self.syntheticFusionForLayers = multiLBP.start(self.synthetic, self.syntheticClassMat, self.syntheticNrOfClasses, self.NUMBER_OF_NODES, self.NUMBER_OF_FOLDS, self.LBP_MAX_STEPS, self.LBP_TRESHOLD, self.LAYERS_WEIGHTS, self.percentOfTrainignNodes, self.method)
#         self.realLBPFoldSum, self.realLBPFusionMean = multiLBP.start(self.realGraph, self.realGraphClassMat, self.realNrOfClasses, self.REAL_NUMBER_OF_NODES, self.NUMBER_OF_FOLDS, self.LBP_MAX_STEPS, self.LBP_TRESHOLD, self.REAL_LAYERS_WEIGHTS)
        self.syntheticLBPFoldSumScores = [element[2] for element in fold_sum]
        self.syntheticLBPFusionMeanScores = [element[2] for element in fusion_mean]
        self.syntheticFusionLayerScores = [element[2] for element in fusion_layer]
        self.syntheticFusionRandomScores = [element[2] for element in fusion_random]
        self.syntheticFusionConvergenceMaxScores = [element[2] for element in fusion_convergence_max]
        self.syntheticFusionConvergenceMinScores = [element[2] for element in fusion_convergence_min]
        for layer, class_mat in layer_results.iteritems():
            self.syntheticFusionForLayersScores[layer] = [element[2] for element in class_mat]
        
        
    def rwpLBP(self):
        rwpLBP = RwpLBP()
        self.syntheticRWPFoldSum, self.syntheticRWPFusionMean, self.rwpResult = rwpLBP.start(self.synthetic, self.syntheticClassMat, self.syntheticNrOfClasses, self.NUMBER_OF_NODES, self.NUMBER_OF_FOLDS, self.LBP_MAX_STEPS, self.LBP_TRESHOLD, self.LAYERS_WEIGHTS, self.percentOfTrainignNodes, self.method)
#         self.realRWPFoldSum, self.realRWPFusionMean = rwpLBP.start(self.realGraph, self.realGraphClassMat, self.realNrOfClasses, self.REAL_NUMBER_OF_NODES, self.NUMBER_OF_FOLDS, self.LBP_MAX_STEPS, self.LBP_TRESHOLD, self.REAL_LAYERS_WEIGHTS)

    def rwc(self):
        rwc = RandomWalkMethods()
        self.syntheticRwcResult = rwc.random_walk_classical(self.synthetic, self.syntheticClassMat, self.LAYERS_WEIGHTS,
                                                                                self.NUMBER_OF_FOLDS, self.method, self.percentOfTrainignNodes)
    '''
    Evaluation
    '''
    def evaluation(self):
        self.syntheticLabels = self.prepareOriginalLabels(self.syntheticClassMat, self.syntheticNrOfClasses)   
        ev = EvaluationTools()
        fMacroFlatSynthetic = metrics.f1_score(self.syntheticLabels, self.syntheticFlatResult, pos_label=None, average='micro')
        self.append_roc_rates_for_average(self.syntheticFlatScores, "reduction")
        fMacroLBPSyntheticFoldSum = metrics.f1_score(self.syntheticLabels, self.syntheticLBPFoldSum, pos_label=None, average='micro')
        self.append_roc_rates_for_average(self.syntheticLBPFoldSumScores, "fusion_sum")
        fMacroLBPSyntheticFusionMean =  metrics.f1_score(self.syntheticLabels, self.syntheticLBPFusionMean, pos_label=None, average='micro')
        self.append_roc_rates_for_average(self.syntheticLBPFusionMeanScores, "fusion_mean")
        fMicroLBPFusionLayer = metrics.f1_score(self.syntheticLabels, self.syntheticFusionLayer, pos_label=None, average='micro')
        self.append_roc_rates_for_average(self.syntheticFusionLayerScores, "fusion_layer")
        fMicroLBPFusionRandom = metrics.f1_score(self.syntheticLabels, self.syntheticFusionRadom, pos_label=None, average='micro')
        self.append_roc_rates_for_average(self.syntheticFusionRandomScores, "fusion_random")
        fMicroLBPFusionConvergenceMax = metrics.f1_score(self.syntheticLabels, self.syntheticFusionConvergenceMax, pos_label=None, average='micro')
        self.append_roc_rates_for_average(self.syntheticFusionConvergenceMaxScores, "fusion_convergence_max")
        fMicroLBPFusionConvergenceMin = metrics.f1_score(self.syntheticLabels, self.syntheticFusionConvergenceMin, pos_label=None, average='micro')
        self.append_roc_rates_for_average(self.syntheticFusionConvergenceMinScores, "fusion_convergence_min")
        fMicroFromLayers = {}
        for layer, result in self.syntheticFusionForLayers.iteritems():
            fMicroFromLayers[layer] = metrics.f1_score(self.syntheticLabels, result, pos_label=None, average='micro')
        # fMacroRWPSyntheticFoldSum = metrics.f1_score(self.syntheticLabels, self.syntheticRWPFoldSum, pos_label=None, average='micro')
        # fMacroRWPSyntheticFusionMean = metrics.f1_score(self.syntheticLabels, self.syntheticRWPFusionMean, pos_label=None, average='micro')
        # fMacroRWPSyntheticResult = metrics.f1_score(self.syntheticLabels, self.rwpResult, pos_label=None, average='micro')
        # fMacroRWCSyntheticResult = metrics.f1_score(self.syntheticLabels, self.syntheticRwcResult, pos_label=None, average='micro')

        with open(self.FILE_PATH + self.build_output_file_name(),'wb') as csvfile:
            writer = csv.writer(csvfile)
        
            writer.writerow([self.NUMBER_OF_NODES, self.AVERAGE_GROUP_SIZE, self.GROUP_LABEL_HOMOGENITY,
                              self.PROBABILITY_OF_EDGE_EXISTANCE_IN_SAME_GROUP, self.PROBABILITY_OF_EDGE_EXISTANCE_BETWEEN_OTHER_GROUPS,
                                self.nrOfLayers, self.method, self.NUMBER_OF_FOLDS,
                            fMacroFlatSynthetic, fMacroLBPSyntheticFoldSum, fMacroLBPSyntheticFusionMean, fMicroLBPFusionLayer, fMicroLBPFusionRandom, fMicroLBPFusionConvergenceMax, fMicroLBPFusionConvergenceMin, [str(e) + ',' + str(fMicroFromLayers[e] if fMicroFromLayers.has_key(e) else '') for e in xrange(1,22)]])
        

    def append_roc_rates_for_average(self, scores, method):
        fpr, tpr, threashold = metrics.roc_curve(self.syntheticLabels, scores)
        self.tprs_per_method[method] = tpr
        self.fprs_per_method[method] = fpr

    def prepareOriginalLabels(self, defaultClassMat, nrOfClasses):
        classMatForEv = []
        for i in range(0, defaultClassMat.__len__()):
            maxi = 0
            for j in range(1, nrOfClasses):
                if (defaultClassMat[i][j] > defaultClassMat[i][maxi]):
                    maxi = j
            classMatForEv.append(maxi)
        return classMatForEv
    
    if __name__ == '__main__':        
        print 
