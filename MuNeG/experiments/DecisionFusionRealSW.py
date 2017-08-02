'''
Created on 13.03.2014

@author: apopiel
'''
import csv

import networkx as nx
import sklearn.metrics as metrics

from graph.evaluation.EvaluationTools import EvaluationTools
from graph.method.lbp.FlatLBP import FlatLBP
from graph.method.lbp.Multilayer_LBP import Multilayer_LBP
from graph.method.lbp.NetworkUtils import NetworkUtils
from graph.method.lbp.RwpLBP import RwpLBP
from graph.method.random_walk.RandomWalkMethods import RandomWalkMethods
from graph.reader.StarWars.StarWarsReader import StarWarsReader


class DecisionFusion(object):
    
    #Parameters
    REAL_NUMBER_OF_NODES = 288
    NUMBER_OF_NODES = 100
    NUMBER_OF_GROUPS = 0
    AVERAGE_GROUP_SIZE = 50
     
    LAYERS_WEIGHTS = []
    REAL_LAYERS_WEIGHTS = [1, 2, 3, 4, 5]
    LAYERS_NAME = []
     
    GROUP_LABEL_HOMOGENITY = 1
    PROBABILITY_OF_EDGE_EXISTANCE_IN_SAME_GROUP = 9
    PROBABILITY_OF_EDGE_EXISTANCE_BETWEEN_OTHER_GROUPS = 1
     
    NUMBER_OF_FOLDS = 5
     
    LBP_MAX_STEPS = 100
    LBP_TRESHOLD = 0.01
    
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

    def __init__(self, method, fold):
        if method == 1:
            self.NUMBER_OF_FOLDS = fold
        else:
            self.percentOfTrainignNodes = fold
        self.method = method
        self.fprs_per_method = {}
        self.tprs_per_method = {}



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
        
    def processExperiment(self):
        self.readRealData()
        self.preprocessing()
        self.flatLBP()
        self.multiLayerLBP()
        # self.rwpLBP()
        # self.rwc()
        self.evaluation()
        return self.fprs_per_method, self.tprs_per_method
        
    '''
    Prepare data
    '''      
    def readRealData(self):
        reader = StarWarsReader()
        reader.read()
        self.realGraph = reader.graph
        # ga = GraphAnalyser(self.realGraph)
        # ga.analyse()
        # self.terms_map = reader.create_go_terms_map()

    '''
    Preprocessing
    '''
    def preprocessing(self):
        self.realGraphClassMat, self.realNrOfClasses = self.nu.createClassMat(self.realGraph)
    '''
    Algorithms
    '''        
    def flatLBP(self):
        flatLBP = FlatLBP()
        nrOfNodes = self.realGraph.nodes().__len__()
        fold_sum, self.realFlatResult = flatLBP.start(self.realGraph, nrOfNodes, self.realGraphClassMat, self.realNrOfClasses, self.LBP_MAX_STEPS, self.LBP_TRESHOLD, self.NUMBER_OF_FOLDS, self.percentOfTrainignNodes, self.method)
        self.realFlatScores = [element[2] for element in fold_sum]
        
        
    def multiLayerLBP(self):
        multiLBP = Multilayer_LBP()
        nrOfNodes = self.realGraph.nodes().__len__()
        fold_sum, fusion_mean, fusion_layer, fusion_random, fusion_convergence_max, fusion_convergence_min, layer_results, self.realLBPFoldSum, self.realLBPFusionMean, self.realFusionLayer, self.realFusionRadom, self.realFusionConvergenceMax, self.realFusionConvergenceMin, self.realFusionForLayers = multiLBP.start(self.realGraph, self.realGraphClassMat, self.realNrOfClasses,
                                                                     nrOfNodes, self.NUMBER_OF_FOLDS, self.LBP_MAX_STEPS, self.LBP_TRESHOLD, self.REAL_LAYERS_WEIGHTS,
                                                                     self.percentOfTrainignNodes, self.method)
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
        ev = EvaluationTools()
        fMacroFlatReal = metrics.f1_score(self.realLabels, self.realFlatResult,pos_label=None, average='micro')
        self.append_roc_rates_for_average(self.realFlatScores, "reduction")
        fMacroLBPRealFoldSum = metrics.f1_score(self.realLabels, self.realLBPFoldSum,pos_label=None, average='micro')
        self.append_roc_rates_for_average(self.realLBPFoldSumScores, "fusion_sum")
        fMacroLBPRealFusionMean = metrics.f1_score(self.realLabels, self.realLBPFusionMean,pos_label=None, average='micro')
        self.append_roc_rates_for_average(self.realLBPFusionMeanScores, "fusion_mean")
        fMicroLBPFusionLayer = metrics.f1_score(self.realLabels, self.realFusionLayer, pos_label=None, average='micro')
        self.append_roc_rates_for_average(self.realFusionLayerScores, "fusion_layer")
        fMicroLBPFusionRandom = metrics.f1_score(self.realLabels, self.realFusionRadom, pos_label=None, average='micro')
        self.append_roc_rates_for_average(self.realFusionRandomScores, "fusion_random")
        fMicroLBPFusionConvergenceMax = metrics.f1_score(self.realLabels, self.realFusionConvergenceMax, pos_label=None, average='micro')
        self.append_roc_rates_for_average(self.realFusionConvergenceMaxScores, "fusion_convergence_max")
        fMicroLBPFusionConvergenceMin = metrics.f1_score(self.realLabels, self.realFusionConvergenceMin, pos_label=None, average='micro')
        self.append_roc_rates_for_average(self.realFusionConvergenceMinScores, "fusion_convergence_min")
        fMicroFromLayers = {}
        for layer, result in self.realFusionForLayers.iteritems():
            fMicroFromLayers[layer] = metrics.f1_score(self.realLabels, result, pos_label=None, average='micro')
        # fMacroRWPRealFoldSum = metrics.f1_score(self.realLabels, self.realRWPFoldSum,pos_label=None, average='micro')
        # fMacroRWPRealFusionMean = metrics.f1_score(self.realLabels, self.realRWPFusionMean,pos_label=None, average='micro')
        # fMacroRWPReal = metrics.f1_score(self.realLabels, self.rwpResult,pos_label=None, average='micro')
        # fMacroRWCRealResult = metrics.f1_score(self.realLabels, self.realRwcResult,pos_label=None, average='micro')
        #
        # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic example')
        # plt.legend(loc="lower right")
        # plt.show()
        with open(self.file_path + 'real.csv', 'ab') as csvfile:
            writer = csv.writer(csvfile)
        
            writer.writerow([
                self.realGraph.nodes().__len__(), self.method, self.percentOfTrainignNodes if self.method == 2 else self.NUMBER_OF_FOLDS,
                            fMacroFlatReal, fMacroLBPRealFoldSum, fMacroLBPRealFusionMean, fMicroLBPFusionLayer, fMicroLBPFusionRandom, fMicroLBPFusionConvergenceMax, fMicroLBPFusionConvergenceMin, [str(e[0]) + ',' + str(e[1]) for e in fMicroFromLayers.iteritems()]])

    def append_roc_rates_for_average(self, scores, method):
        fpr, tpr, threashold = metrics.roc_curve(self.realLabels, scores)
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
