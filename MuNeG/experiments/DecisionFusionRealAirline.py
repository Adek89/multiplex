'''
Created on 13.03.2014

@author: apopiel
'''

import networkx as nx
import numpy as np
import scipy
import sklearn.metrics as metrics
from graph.evaluation.EvaluationTools import EvaluationTools
from graph.method.common.XValWithSampling import XValMethods
from graph.method.lbp.FlatLBP import FlatLBP
from graph.method.lbp.Multilayer_LBP import Multilayer_LBP
from graph.method.lbp.NetworkUtils import NetworkUtils
from graph.method.lbp.RwpLBP import RwpLBP
from graph.method.random_walk.RandomWalkMethods import RandomWalkMethods
from graph.reader.Airline2016.Airline2016Reader import Airline2016Reader


class DecisionFusion(object):
    
    #Parameters
    REAL_NUMBER_OF_NODES = 288
    NUMBER_OF_NODES = 100
    NUMBER_OF_GROUPS = 0
    AVERAGE_GROUP_SIZE = 50
     
    LAYERS_WEIGHTS = []
    REAL_LAYERS_WEIGHTS = []
    LAYERS_NAME = []
     
    GROUP_LABEL_HOMOGENITY = 1
    PROBABILITY_OF_EDGE_EXISTANCE_IN_SAME_GROUP = 9
    PROBABILITY_OF_EDGE_EXISTANCE_BETWEEN_OTHER_GROUPS = 1
     
    NUMBER_OF_FOLDS = 5
     
    LBP_MAX_STEPS = 25
    LBP_TRESHOLD = 0.001
    
    training = []
    validation = []
    
    file_path = ""
    
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
    realFlatResult = []
    realFlatScores = []
    
    syntheticLBPFoldSum = []
    syntheticLBPFusionMean = []
    realLBPFoldSum = []
    realLBPFoldSumScores = []
    realLBPFusionMean = []
    realLBPFusionMeanScores = []
    realRwcResult = []

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
    method = 0
    fun = 0
    terms_map = dict([])
    rwpResult = []
    realFusionLayer = []
    realFusionLayerScores = []
    realFusionRandom = []
    realFusionRandomScores = []
    realFusionConvergenceMax = []
    realFusionConvergenceMaxScores = []
    realFusionConvergenceMin = []
    realFusionConvergenceMinScores = []
    realFusionForLayers = []
    realFusionForLayersScores = {}

    fprs_per_method = {}
    tprs_per_method = {}
    keys = ["reduction", "fusion_sum", "fusion_mean", "fusion_layer", "fusion_random", "fusion_convergence_max", "fusion_convergence_min"]
    folds = []

    def __init__(self, method, fold, threshold):
        if method == 1:
            self.NUMBER_OF_FOLDS = fold
        else:
            self.percentOfTrainignNodes = fold
        self.method = method
        self.fprs_per_method = {}
        self.tprs_per_method = {}
        if threshold == 100:
            self.REAL_LAYERS_WEIGHTS = [i for i in xrange(1, 65)]
        elif threshold == 1000:
            self.REAL_LAYERS_WEIGHTS = [i for i in xrange(1, 14)]
        else:
            self.REAL_LAYERS_WEIGHTS = [i for i in xrange(1, 133)]



    def initLayers(self, nrOfLayers):
        for i in xrange(0, nrOfLayers):
            self.LAYERS_WEIGHTS.append(i + 1)
            self.LAYERS_NAME.append("L"+str(i+1))


    def prepareNumberOfGroups(self, nrOfNodes, nrOfGroups):
        while True:
            dividedInt = nrOfNodes % nrOfGroups
            if (not dividedInt == 0):
                nrOfNodes = nrOfNodes + 1
            else: 
                break
        self.NUMBER_OF_NODES = nrOfNodes
        self.AVERAGE_GROUP_SIZE = nrOfNodes / nrOfGroups
        
    def processExperiment(self, threshold, class_label):
        homogenity = self.readRealData(threshold, class_label)
        self.preprocessing()
        self.flatLBP()
        self.multiLayerLBP()
        # self.rwpLBP()
        # self.rwc()
        self.evaluation()
        return self.fprs_per_method, self.tprs_per_method, homogenity
        
    '''
    Prepare data
    '''      
    def readRealData(self, threshold, class_label):
        reader = Airline2016Reader()
        reader.read(threshold, classLabel = class_label)
        self.realGraph = reader.graph
        # ga = GraphAnalyser(self.realGraph)
        # ga.analyse()
        # self.terms_map = reader.create_go_terms_map()
        return reader.calcuclate_homogenity()

    '''
    Preprocessing
    '''
    def preprocessing(self):
        xval = XValMethods(self.realGraph)
        self.folds = xval.stratifies_x_val(self.realGraph.nodes(), self.NUMBER_OF_FOLDS)
        self.realGraphClassMat, self.realNrOfClasses = self.nu.createClassMat(self.realGraph)
    '''
    Algorithms
    '''        
    def flatLBP(self):
        flatLBP = FlatLBP()
        nrOfNodes = self.realGraph.nodes().__len__()
        fold_sum, self.realFlatResult = flatLBP.start(self.realGraph, nrOfNodes, self.realGraphClassMat, self.realNrOfClasses, self.LBP_MAX_STEPS, self.LBP_TRESHOLD, self.NUMBER_OF_FOLDS, self.percentOfTrainignNodes, self.method, self.folds)
        self.realFlatScores = [[element[i] for i in xrange(1, self.realGraphClassMat.shape[1]+1)] for element in fold_sum]
        
        
    def multiLayerLBP(self):
        multiLBP = Multilayer_LBP()
        nrOfNodes = self.realGraph.nodes().__len__()
        fold_sum, fusion_mean, fusion_layer, fusion_random, fusion_convergence_max, fusion_convergence_min, layer_results, self.realLBPFoldSum, self.realLBPFusionMean, self.realFusionLayer, self.realFusionRadom, self.realFusionConvergenceMax, self.realFusionConvergenceMin, self.realFusionForLayers = multiLBP.start(self.realGraph, self.realGraphClassMat, self.realNrOfClasses,
                                                                     nrOfNodes, self.NUMBER_OF_FOLDS, self.LBP_MAX_STEPS, self.LBP_TRESHOLD, self.REAL_LAYERS_WEIGHTS,
                                                                     self.percentOfTrainignNodes, self.method, self.folds)
        self.realLBPFoldSumScores = [element[2] for element in fold_sum]
        self.realLBPFusionMeanScores = [element[2] for element in fusion_mean]
        self.realFusionLayerScores = [element[2] for element in fusion_layer]
        self.realFusionRandomScores = [element[2] for element in fusion_random]
        self.realFusionConvergenceMaxScores = [element[2] for element in fusion_convergence_max]
        self.realFusionConvergenceMinScores = [element[2] for element in fusion_convergence_min]
        for layer, class_mat in layer_results.iteritems():
            self.realFusionForLayersScores[layer] = [element[2] for element in class_mat]


        
        
    def rwpLBP(self):
        rwpLBP = RwpLBP()
        nrOfNodes = self.realGraph.nodes().__len__()
        self.realRWPFoldSum, self.realRWPFusionMean, self.rwpResult = rwpLBP.start(self.realGraph, self.realGraphClassMat, self.realNrOfClasses,
                                                                   nrOfNodes, self.NUMBER_OF_FOLDS, self.LBP_MAX_STEPS, self.LBP_TRESHOLD, self.REAL_LAYERS_WEIGHTS,
                                                                   self.percentOfTrainignNodes, self.method)

    def rwc(self):
        rwc = RandomWalkMethods()
        self.realRwcResult = rwc.random_walk_classical(self.realGraph, self.realGraphClassMat, self.REAL_LAYERS_WEIGHTS,
                                                                                self.NUMBER_OF_FOLDS, self.method, self.percentOfTrainignNodes)

    '''
    Evaluation
    '''
    def evaluation(self):
        self.realLabels = self.prepareOriginalLabels(self.realGraphClassMat, self.realNrOfClasses)
        self.syntheticLabels = self.prepareOriginalLabels(self.syntheticClassMat, self.syntheticNrOfClasses)

        ids_to_remove = self.preprocess_for_evaluation()
        new_labels = []
        for id in xrange(0, len(self.realLabels)):
            if not id in ids_to_remove:
            # if True:
                new_labels.append(self.realLabels[id])
        new_reduction_scores = []
        new_fold_sum_scores = []
        new_fusion_mean_scores = []
        new_fusion_layer_scores = []
        new_fusion_random_scores = []
        new_fusion_max_conv_scores = []
        new_fusion_min_conv_scores = []
        new_fusion_for_layers = {}
        for layer, result in self.realFusionForLayersScores.iteritems():
            new_fusion_for_layers.update({layer : []})
        if not all(l == 0 for l in new_labels):
            for id in xrange(0, len(self.realLabels)):
                if not id in ids_to_remove:
                # if True:
                    new_reduction_scores.append(self.realFlatScores[id])
                    # new_fold_sum_scores.append(self.realLBPFoldSumScores[id])
                    # new_fusion_mean_scores.append(self.realLBPFusionMeanScores[id])
                    # new_fusion_layer_scores.append(self.realFusionLayerScores[id])
                    # new_fusion_random_scores.append(self.realFusionRandomScores[id])
                    # new_fusion_max_conv_scores.append(self.realFusionConvergenceMaxScores[id])
                    # new_fusion_min_conv_scores.append(self.realFusionConvergenceMinScores[id])
                    # for layer, result in self.realFusionForLayersScores.iteritems():
                    #     new_fusion_for_layers[layer].append(result[id])
        else:
            new_labels = self.realLabels
            for id in xrange(0, len(self.realLabels)):
                new_reduction_scores.append(self.realFlatScores[id])
                # new_fold_sum_scores.append(self.realLBPFoldSumScores[id])
                # new_fusion_mean_scores.append(self.realLBPFusionMeanScores[id])
                # new_fusion_layer_scores.append(self.realFusionLayerScores[id])
                # new_fusion_random_scores.append(self.realFusionRandomScores[id])
                # new_fusion_max_conv_scores.append(self.realFusionConvergenceMaxScores[id])
                # new_fusion_min_conv_scores.append(self.realFusionConvergenceMinScores[id])
                # for layer, result in self.realFusionForLayersScores.iteritems():
                #     new_fusion_for_layers[layer].append(result[id])
        ev = EvaluationTools()
        fMacroFlatReal = metrics.f1_score(self.realLabels, self.realFlatResult,pos_label=None, average='micro')
        for c_id in xrange(0, self.realNrOfClasses):
            self.append_roc_rates_for_average([score[c_id] for score in new_reduction_scores], [1 if l == c_id else 0 for l in new_labels], str(c_id))
        self.append_roc_rates_for_average([score[c_id]  for c_id in xrange(0, self.realNrOfClasses) for score in new_reduction_scores], [1 if l == c_id else 0 for c_id in xrange(0, self.realGraphClassMat.shape[1]) for l in new_labels], "micro")
        self.append_macro_results()
        # self.append_roc_rates_for_average(new_reduction_scores, new_labels, "reduction")
        # fMacroLBPRealFoldSum = metrics.f1_score(self.realLabels, self.realLBPFoldSum,pos_label=None, average='micro')
        # self.append_roc_rates_for_average(new_fold_sum_scores, new_labels, "fusion_sum")
        # fMacroLBPRealFusionMean = metrics.f1_score(self.realLabels, self.realLBPFusionMean,pos_label=None, average='micro')
        # self.append_roc_rates_for_average(new_fusion_mean_scores, new_labels, "fusion_mean")
        # fMicroLBPFusionLayer = metrics.f1_score(self.realLabels, self.realFusionLayer, pos_label=None, average='micro')
        # self.append_roc_rates_for_average(new_fusion_layer_scores, new_labels, "fusion_layer")
        # fMicroLBPFusionRandom = metrics.f1_score(self.realLabels, self.realFusionRadom, pos_label=None, average='micro')
        # self.append_roc_rates_for_average(new_fusion_random_scores, new_labels, "fusion_random")
        # fMicroLBPFusionConvergenceMax = metrics.f1_score(self.realLabels, self.realFusionConvergenceMax, pos_label=None, average='micro')
        # self.append_roc_rates_for_average(new_fusion_max_conv_scores, new_labels, "fusion_convergence_max")
        # fMicroLBPFusionConvergenceMin = metrics.f1_score(self.realLabels, self.realFusionConvergenceMin, pos_label=None, average='micro')
        # self.append_roc_rates_for_average(new_fusion_min_conv_scores, new_labels, "fusion_convergence_min")
        # fMicroFromLayers = {}
        # for layer, result in self.realFusionForLayers.iteritems():
        #     fMicroFromLayers[layer] = metrics.f1_score(self.realLabels, result, pos_label=None, average='micro')
        #     self.append_roc_rates_for_average(new_fusion_for_layers[layer], new_labels, 'L'+str(layer))

    def append_macro_results(self):
        all_fpr = np.unique(np.concatenate([self.fprs_per_method[str(i)] for i in range(self.realNrOfClasses)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.realNrOfClasses):
            mean_tpr += scipy.interp(all_fpr, self.fprs_per_method[str(i)], self.tprs_per_method[str(i)])
        mean_tpr /= self.realNrOfClasses
        self.fprs_per_method["macro"] = all_fpr
        self.tprs_per_method["macro"] = mean_tpr

    def preprocess_for_evaluation(self):
        ids_to_remove = []
        components = sorted(nx.connected_components(self.realGraph), key=len, reverse=True)
        for component in components:
            for fold in self.folds:
                ids_in_component = [n.id for n in component]
                if set(sorted(ids_in_component)).issubset(set(fold[1])):
                    ids_to_remove.append(ids_in_component)
        return sorted([item for sublist in ids_to_remove for item in sublist])


    def append_roc_rates_for_average(self, scores, real_labels, method):
        fpr, tpr, threashold = metrics.roc_curve(real_labels, scores)
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

    def calculate_homogenity(self, graph):
        # homogenity
        homogenity_distribution = []
        node_ids = []
        nodes = graph.nodes()
        sorted_nodes = sorted(nodes, key=lambda n: n.id)
        for n in sorted_nodes:
            neighbors_with_same_class = 0
            neighbors = graph.neighbors(n)
            for neighbor in neighbors:
                if neighbor.label == n.label:
                    neighbors_with_same_class = neighbors_with_same_class + 1
            homogenity_distribution.append(float(neighbors_with_same_class)/float(len(neighbors)) if len(neighbors) > 0 else 0.0)
            node_ids.append(n.id)
        return homogenity_distribution, node_ids
