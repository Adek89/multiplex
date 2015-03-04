'''
Created on 13.03.2014

@author: apopiel
'''
from graph.reader.ExcelReader import ExcelReader
from graph.gen.GraphGenerator import GraphGenerator
from graph.method.lbp.Multilayer_LBP import Multilayer_LBP
import networkx as nx
from graph.method.lbp.NetworkUtils import NetworkUtils
from graph.method.lbp.FlatLBP import FlatLBP
from graph.method.lbp.RwpLBP import RwpLBP
from graph.evaluation.EvaluationTools import EvaluationTools
from graph.reader.Salon24.Salon24Reader import Salon24Reader
import csv
import time
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
    
    FILE_PATH = "/home/apopiel/tmp/output"
    # FILE_PATH = "output"
    
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
    sampledNodes = 0
    limit = 0

    def __init__(self, percentOfTrainignNodes, counter, sampledNodes, limit):
        self.percentOfTrainignNodes = percentOfTrainignNodes
        self.counter = counter
        self.sampledNodes = sampledNodes
        self.limit = limit

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
#         self.generateSyntheticData()
        self.preprocessing()
        self.flatLBP()
        self.multiLayerLBP()
        self.rwpLBP()
        self.evaluation()
        
    '''
    Prepare data
    '''      
    def readRealData(self):
        reader = Salon24Reader(self.limit)
        self.realGraph = reader.createNetwork()
        ensamble = EnsambleLearning(self.realGraph, 1, self.sampledNodes)
        self.realGraph = ensamble.sampleGraph()
        ga = GraphAnalyser(self.realGraph, self.percentOfTrainignNodes, self.counter)
        ga.analyse()

    def generateSyntheticData(self):
        start_time = time.time()
        self.gg = GraphGenerator(self.NUMBER_OF_NODES, self.AVERAGE_GROUP_SIZE, self.LAYERS_WEIGHTS,
                                 self.GROUP_LABEL_HOMOGENITY, self.PROBABILITY_OF_EDGE_EXISTANCE_IN_SAME_GROUP,
                                 self.PROBABILITY_OF_EDGE_EXISTANCE_BETWEEN_OTHER_GROUPS, self.LAYERS_NAME)
        self.synthetic = self.gg.generate()
        print("---generation time: %s seconds ---" % str(time.time() - start_time))
        
    '''
    Preprocessing
    '''
    def preprocessing(self):
        self.realGraphClassMat, self.realNrOfClasses = self.nu.createClassMat(self.realGraph)
#         self.syntheticClassMat, self.syntheticNrOfClasses = self.nu.createClassMat(self.synthetic)
    '''
    Algorithms
    '''        
    def flatLBP(self):
        start_time = time.time()
        flatLBP = FlatLBP()
        # self.syntheticFlatResult = flatLBP.start(self.synthetic, self.NUMBER_OF_NODES, self.syntheticClassMat,
        #                                          self.syntheticNrOfClasses, self.LBP_MAX_STEPS, self.LBP_TRESHOLD,
        #                                          self.NUMBER_OF_FOLDS, self.percentOfTrainignNodes)
        print("---flatLBP time: %s seconds ---" % str(time.time() - start_time))
        nrOfNodes = self.realGraph.nodes().__len__()
        self.realFlatResult = flatLBP.start(self.realGraph, nrOfNodes, self.realGraphClassMat, self.realNrOfClasses, self.LBP_MAX_STEPS, self.LBP_TRESHOLD, self.NUMBER_OF_FOLDS, self.percentOfTrainignNodes)
        
        
        
    def multiLayerLBP(self):
        multiLBP = Multilayer_LBP()
        # self.syntheticLBPFoldSum, self.syntheticLBPFusionMean = multiLBP.start(self.synthetic, self.syntheticClassMat, self.syntheticNrOfClasses, self.NUMBER_OF_NODES, self.NUMBER_OF_FOLDS, self.LBP_MAX_STEPS, self.LBP_TRESHOLD, self.LAYERS_WEIGHTS, self.percentOfTrainignNodes)
        nrOfNodes = self.realGraph.nodes().__len__()
        self.realLBPFoldSum, self.realLBPFusionMean = multiLBP.start(self.realGraph, self.realGraphClassMat, self.realNrOfClasses,
                                                                     nrOfNodes, self.NUMBER_OF_FOLDS, self.LBP_MAX_STEPS, self.LBP_TRESHOLD, self.REAL_LAYERS_WEIGHTS,
                                                                     self.percentOfTrainignNodes)

        
        
    def rwpLBP(self):
        rwpLBP = RwpLBP()
        # self.syntheticRWPFoldSum, self.syntheticRWPFusionMean = rwpLBP.start(self.synthetic, self.syntheticClassMat, self.syntheticNrOfClasses, self.NUMBER_OF_NODES, self.NUMBER_OF_FOLDS, self.LBP_MAX_STEPS, self.LBP_TRESHOLD, self.LAYERS_WEIGHTS, self.percentOfTrainignNodes)
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
        # fMacroFlatSynthetic = ev.calculateFMacro(self.syntheticLabels, self.syntheticFlatResult, self.syntheticNrOfClasses)
        # fMacroLBPSyntheticFoldSum = ev.calculateFMacro(self.syntheticLabels, self.syntheticLBPFoldSum, self.syntheticNrOfClasses)
        # fMacroLBPSyntheticFusionMean = ev.calculateFMacro(self.syntheticLabels, self.syntheticLBPFusionMean, self.syntheticNrOfClasses)
        # fMacroRWPSyntheticFoldSum = ev.calculateFMacro(self.syntheticLabels, self.syntheticRWPFoldSum, self.syntheticNrOfClasses)
        # fMacroRWPSyntheticFusionMean = ev.calculateFMacro(self.syntheticLabels, self.syntheticRWPFusionMean, self.syntheticNrOfClasses)
        
        fMacroFlatReal = ev.calculateFMacro(self.realLabels, self.realFlatResult, self.realNrOfClasses)
        fMacroLBPRealFoldSum = ev.calculateFMacro(self.realLabels, self.realLBPFoldSum, self.realNrOfClasses)
        fMacroLBPRealFusionMean = ev.calculateFMacro(self.realLabels, self.realLBPFusionMean, self.realNrOfClasses)
        fMacroRWPRealFoldSum = ev.calculateFMacro(self.realLabels, self.realRWPFoldSum, self.realNrOfClasses)
        fMacroRWPRealFusionMean = ev.calculateFMacro(self.realLabels, self.realRWPFusionMean, self.realNrOfClasses)
        
        with open(self.FILE_PATH + str(self.counter) + 'real.csv', 'ab') as csvfile:
            writer = csv.writer(csvfile)
        
            writer.writerow([
                self.realGraph.nodes().__len__(),self.percentOfTrainignNodes, self.sampledNodes, self.limit,
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
    
    if __name__ == '__main__':        
        print 
