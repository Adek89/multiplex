'''
Created on 13.03.2014

@author: apopiel
'''
import networkx as nx
import csv
import time
from graph.reader.ExcelReader import ExcelReader
from graph.gen.GraphGenerator import GraphGenerator
from graph.method.lbp.Multilayer_LBP import Multilayer_LBP
from graph.method.lbp.NetworkUtils import NetworkUtils
from graph.method.lbp.FlatLBP import FlatLBP
from graph.method.lbp.RwpLBP import RwpLBP
from graph.evaluation.EvaluationTools import EvaluationTools
from graph.reader.DanioRerio.DanioRerioReader import DanioRerioReader
from graph.method.ensamble.EnsambleLearning import EnsambleLearning
from graph.analyser.GraphAnalyser import GraphAnalyser
import matplotlib.pyplot as plt
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
    method = 0
    fold = 0

    def __init__(self, method, fold, percentOfTrainignNodes):
        self.percentOfTrainignNodes = percentOfTrainignNodes
        self.method = method
        self.fold = fold


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
        self.rwpLBP()
        self.evaluation()
        
    '''
    Prepare data
    '''      
    def readRealData(self):
        reader = DanioRerioReader()
        reader.read('GO:0000059')
        self.realGraph = reader.graph
        ga = GraphAnalyser(self.realGraph, self.percentOfTrainignNodes, self.counter)
        ga.analyse()
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
        self.realFlatResult = flatLBP.start(self.realGraph, nrOfNodes, self.realGraphClassMat, self.realNrOfClasses, self.LBP_MAX_STEPS, self.LBP_TRESHOLD, self.NUMBER_OF_FOLDS, self.percentOfTrainignNodes, self.method)
        
        
        
    def multiLayerLBP(self):
        multiLBP = Multilayer_LBP()
        nrOfNodes = self.realGraph.nodes().__len__()
        self.realLBPFoldSum, self.realLBPFusionMean = multiLBP.start(self.realGraph, self.realGraphClassMat, self.realNrOfClasses,
                                                                     nrOfNodes, self.NUMBER_OF_FOLDS, self.LBP_MAX_STEPS, self.LBP_TRESHOLD, self.REAL_LAYERS_WEIGHTS,
                                                                     self.percentOfTrainignNodes)

        
        
    def rwpLBP(self):
        rwpLBP = RwpLBP()
        nrOfNodes = self.realGraph.nodes().__len__()
        self.realRWPFoldSum, self.realRWPFusionMean = rwpLBP.start(self.realGraph, self.realGraphClassMat, self.realNrOfClasses,
                                                                   nrOfNodes, self.NUMBER_OF_FOLDS, self.LBP_MAX_STEPS, self.LBP_TRESHOLD, self.REAL_LAYERS_WEIGHTS,
                                                                   self.percentOfTrainignNodes)

    '''
    Evaluation
    '''
    def evaluation(self):
        self.realLabels = self.prepareOriginalLabels(self.realGraphClassMat, self.realNrOfClasses) 
        self.syntheticLabels = self.prepareOriginalLabels(self.syntheticClassMat, self.syntheticNrOfClasses)   
        ev = EvaluationTools()
        fMacroFlatReal = ev.calculateFMacro(self.realLabels, self.realFlatResult, self.realNrOfClasses)
        fMacroLBPRealFoldSum = ev.calculateFMacro(self.realLabels, self.realLBPFoldSum, self.realNrOfClasses)
        fMacroLBPRealFusionMean = ev.calculateFMacro(self.realLabels, self.realLBPFusionMean, self.realNrOfClasses)
        fMacroRWPRealFoldSum = ev.calculateFMacro(self.realLabels, self.realRWPFoldSum, self.realNrOfClasses)
        fMacroRWPRealFusionMean = ev.calculateFMacro(self.realLabels, self.realRWPFusionMean, self.realNrOfClasses)
        
        with open(self.file_path + str(self.counter) + 'real.csv', 'ab') as csvfile:
            writer = csv.writer(csvfile)
        
            writer.writerow([
                self.realGraph.nodes().__len__(),self.percentOfTrainignNodes,
                            fMacroFlatReal, fMacroLBPRealFoldSum, fMacroLBPRealFusionMean, fMacroRWPRealFoldSum,
                            fMacroRWPRealFusionMean])
        
    def prepareOriginalLabels(self, defaultClassMat, nrOfClasses):
        classMatForEv = []
        for i in range(0, defaultClassMat.__len__()):
            maxi = 0
            for j in range(1, nrOfClasses):
                if (defaultClassMat[i][j] > defaultClassMat[i][maxi]):
                    maxi = j
            classMatForEv.append(maxi)
        return classMatForEv
