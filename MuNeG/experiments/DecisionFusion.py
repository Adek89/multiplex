'''
Created on 13.03.2014

@author: apopiel
'''
import csv
import math
import time

import graph.method.lbp.LBPTools as tools
import graph.reader.syntethic.MuNeGGraphReader as reader
import networkx as nx
import sklearn.metrics as metrics
from graph.evaluation.EvaluationTools import EvaluationTools
from graph.method.common.XValWithSampling import XValMethods
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
     
    LBP_MAX_STEPS = 25
    LBP_TRESHOLD = 0.001
    
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
    folds = []


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
        self.postprocessing()
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
        self.synthetic = reader.read_from_gml('..\\results\\', self.build_file_name(),)
        print("---generation time: %s seconds ---" % str(time.time() - start_time))
        
    '''
    Preprocessing
    '''
    def preprocessing(self):
#         self.realGraphClassMat, self.realNrOfClasses = self.nu.createClassMat(self.realGraph)
        xval = XValMethods(self.synthetic)
        self.folds = xval.stratifies_x_val(self.synthetic.nodes(), self.NUMBER_OF_FOLDS)
        self.syntheticClassMat, self.syntheticNrOfClasses = self.nu.createClassMat(self.synthetic)
    '''
    Algorithms
    '''        
    def flatLBP(self):
        start_time = time.time()
        flatLBP = FlatLBP()
        fold_sum, self.syntheticFlatResult = flatLBP.start(self.synthetic, self.NUMBER_OF_NODES, self.syntheticClassMat,
                                                 self.syntheticNrOfClasses, self.LBP_MAX_STEPS, self.LBP_TRESHOLD,
                                                 self.NUMBER_OF_FOLDS, self.percentOfTrainignNodes, self.method, self.folds)
        print("---flatLBP time: %s seconds ---" % str(time.time() - start_time))
        self.syntheticFlatScores = [element[2] for element in fold_sum]
#         self.realFlatResult = flatLBP.start(self.realGraph, self.REAL_NUMBER_OF_NODES, self.realGraphClassMat, self.realNrOfClasses, self.LBP_MAX_STEPS, self.LBP_TRESHOLD, self.NUMBER_OF_FOLDS)
        
        
        
    def multiLayerLBP(self):
        multiLBP = Multilayer_LBP()

        fold_sum, fusion_mean, fusion_layer, fusion_random, fusion_convergence_max, fusion_convergence_min, layer_results, self.syntheticLBPFoldSum, self.syntheticLBPFusionMean, self.syntheticFusionLayer, self.syntheticFusionRadom, self.syntheticFusionConvergenceMax, self.syntheticFusionConvergenceMin, self.syntheticFusionForLayers = multiLBP.start(self.synthetic, self.syntheticClassMat, self.syntheticNrOfClasses, self.NUMBER_OF_NODES, self.NUMBER_OF_FOLDS, self.LBP_MAX_STEPS, self.LBP_TRESHOLD, self.LAYERS_WEIGHTS, self.percentOfTrainignNodes, self.method, self.folds)
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

        ids_to_remove = self.preprocess_for_evaluation()
        new_labels = []
        new_reduction_scores = []
        new_fold_sum_scores = []
        new_fusion_mean_scores = []
        new_fusion_layer_scores = []
        new_fusion_random_scores = []
        new_fusion_max_conv_scores = []
        new_fusion_min_conv_scores = []

        for id in xrange(0, len(self.syntheticLabels)):
            if not id in ids_to_remove:
            # if True:
                new_labels.append(self.syntheticLabels[id])
                new_reduction_scores.append(self.syntheticFlatScores[id])
                new_fold_sum_scores.append(self.syntheticLBPFoldSumScores[id])
                new_fusion_mean_scores.append(self.syntheticLBPFusionMeanScores[id])
                new_fusion_layer_scores.append(self.syntheticFusionLayerScores[id])
                new_fusion_random_scores.append(self.syntheticFusionRandomScores[id])
                new_fusion_max_conv_scores.append(self.syntheticFusionConvergenceMaxScores[id])
                new_fusion_min_conv_scores.append(self.syntheticFusionConvergenceMinScores[id])
        fMacroFlatSynthetic = metrics.f1_score(self.syntheticLabels, self.syntheticFlatResult, pos_label=None, average='micro')
        self.append_roc_rates_for_average(new_reduction_scores, new_labels, "reduction")
        fMacroLBPSyntheticFoldSum = metrics.f1_score(self.syntheticLabels, self.syntheticLBPFoldSum, pos_label=None, average='micro')
        self.append_roc_rates_for_average(new_fold_sum_scores, new_labels, "fusion_sum")
        fMacroLBPSyntheticFusionMean =  metrics.f1_score(self.syntheticLabels, self.syntheticLBPFusionMean, pos_label=None, average='micro')
        self.append_roc_rates_for_average(new_fusion_mean_scores, new_labels, "fusion_mean")
        fMicroLBPFusionLayer = metrics.f1_score(self.syntheticLabels, self.syntheticFusionLayer, pos_label=None, average='micro')
        self.append_roc_rates_for_average(new_fusion_layer_scores, new_labels, "fusion_layer")
        fMicroLBPFusionRandom = metrics.f1_score(self.syntheticLabels, self.syntheticFusionRadom, pos_label=None, average='micro')
        self.append_roc_rates_for_average(new_fusion_random_scores, new_labels, "fusion_random")
        fMicroLBPFusionConvergenceMax = metrics.f1_score(self.syntheticLabels, self.syntheticFusionConvergenceMax, pos_label=None, average='micro')
        self.append_roc_rates_for_average(new_fusion_max_conv_scores, new_labels, "fusion_convergence_max")
        fMicroLBPFusionConvergenceMin = metrics.f1_score(self.syntheticLabels, self.syntheticFusionConvergenceMin, pos_label=None, average='micro')
        self.append_roc_rates_for_average(new_fusion_min_conv_scores, new_labels, "fusion_convergence_min")
        fMicroFromLayers = {}
        for layer, result in self.syntheticFusionForLayers.iteritems():
            fMicroFromLayers[layer] = metrics.f1_score(self.syntheticLabels, result, pos_label=None, average='micro')
            self.append_roc_rates_for_average(self.syntheticFusionForLayersScores[layer], new_labels, 'L'+str(layer))
        # fMacroRWPSyntheticFoldSum = metrics.f1_score(self.syntheticLabels, self.syntheticRWPFoldSum, pos_label=None, average='micro')
        # fMacroRWPSyntheticFusionMean = metrics.f1_score(self.syntheticLabels, self.syntheticRWPFusionMean, pos_label=None, average='micro')
        # fMacroRWPSyntheticResult = metrics.f1_score(self.syntheticLabels, self.rwpResult, pos_label=None, average='micro')
        # fMacroRWCSyntheticResult = metrics.f1_score(self.syntheticLabels, self.syntheticRwcResult, pos_label=None, average='micro')

        with open("..\\results\\synthetic\\stats\\res_" + self.build_output_file_name(),'wb') as csvfile:
            writer = csv.writer(csvfile)
            self.write_method_results(writer, "expected", self.syntheticLabels)
            self.write_method_results(writer, "redution", self.syntheticFlatResult)
            self.write_method_results(writer, "fold_sum", self.syntheticLBPFoldSum)
            self.write_method_results(writer, "fusion_mean", self.syntheticLBPFusionMean)
            self.write_method_results(writer, "fusion_layer", self.syntheticFusionLayer)
            self.write_method_results(writer, "fusion_random", self.syntheticFusionRandom)
            self.write_method_results(writer, "fusion_convergence_max", self.syntheticFusionConvergenceMax)
            self.write_method_results(writer, "fusion_convergence_min", self.syntheticFusionConvergenceMin)
            for layer, result in self.syntheticFusionForLayers.iteritems():
                self.write_method_results(writer, "L"+str(layer), result)
        

    def postprocessing(self):
        graph_for_reduction = self.synthetic
        lbp_tools = tools.LBPTools(self.NUMBER_OF_NODES, self.synthetic, self.syntheticClassMat, self.LBP_MAX_STEPS, self.LBP_TRESHOLD, self.percentOfTrainignNodes)
        lbp_tools.separate_layer(self.synthetic, self.LAYERS_WEIGHTS, self.syntheticClassMat)

        with open("/lustre/scratch/apopiel/synthetic/stats/distributions_" + self.build_output_file_name(),'wb') as csvfile:
            writer = csv.writer(csvfile)
            homogenity_distribution, known_neighbors_distribution, unknown_neighbors_distribution, node_degree_distribution, node_ids = self.calculate_distributions(self.synthetic, "reduction")
            writer.writerow([self.NUMBER_OF_NODES, self.AVERAGE_GROUP_SIZE, self.GROUP_LABEL_HOMOGENITY,
                self.PROBABILITY_OF_EDGE_EXISTANCE_IN_SAME_GROUP, self.PROBABILITY_OF_EDGE_EXISTANCE_BETWEEN_OTHER_GROUPS,
                self.nrOfLayers, "reduction", self.NUMBER_OF_FOLDS,
                homogenity_distribution, known_neighbors_distribution, unknown_neighbors_distribution, node_degree_distribution, node_ids])


            for l in xrange(1, self.nrOfLayers+1):
                homogenity_distribution, known_neighbors_distribution, unknown_neighbors_distribution, node_degree_distribution, node_ids = self.calculate_distributions(lbp_tools.graphs[str(l)], "")
                writer.writerow([self.NUMBER_OF_NODES, self.AVERAGE_GROUP_SIZE, self.GROUP_LABEL_HOMOGENITY,
                self.PROBABILITY_OF_EDGE_EXISTANCE_IN_SAME_GROUP, self.PROBABILITY_OF_EDGE_EXISTANCE_BETWEEN_OTHER_GROUPS,
                self.nrOfLayers, "L" + str(l), self.NUMBER_OF_FOLDS,
                homogenity_distribution, known_neighbors_distribution, unknown_neighbors_distribution, node_degree_distribution, node_ids])

    def write_method_results(self, writer, method, results):
        writer.writerow([self.NUMBER_OF_NODES, self.AVERAGE_GROUP_SIZE, self.GROUP_LABEL_HOMOGENITY,
                         self.PROBABILITY_OF_EDGE_EXISTANCE_IN_SAME_GROUP,
                         self.PROBABILITY_OF_EDGE_EXISTANCE_BETWEEN_OTHER_GROUPS,
                         self.nrOfLayers, method, self.NUMBER_OF_FOLDS,
                         results])

    def calculate_distributions(self, graph, method):
        # homogenity
        homogenity_distribution = []
        known_neighbors_distribution = []
        unknown_neighbors_distribution = []
        node_degree_distribution = []
        node_ids = []
        nodes = graph.nodes()
        for train, test in self.folds:
            for node_id in test:
                neighbors_with_expected_class = 0
                known_neighbors = 0
                unknown_neighbors = 0

                if method == 'reduction':
                    node_object = filter(lambda n: n.id == node_id, nodes)[0]
                    neighbors = graph.neighbors(node_object)
                    for neighbor in neighbors:
                        if neighbor.id in train:
                            known_neighbors = known_neighbors + 1
                            if neighbor.label == node_object.label:
                                neighbors_with_expected_class = neighbors_with_expected_class + 1
                        else:
                            unknown_neighbors = unknown_neighbors + 1
                else:
                    node_objects = self.synthetic.nodes()
                    node_object = filter(lambda n: n.id == node_id, node_objects)[0]
                    neighbors = graph.neighbors(node_id)
                    for neighbor in neighbors:
                        if neighbor in train:
                            known_neighbors = known_neighbors + 1
                            neighbor_object = filter(lambda n: n.id == neighbor, node_objects)[0]
                            if neighbor_object.label == node_object.label:
                                neighbors_with_expected_class = neighbors_with_expected_class + 1
                        else:
                            unknown_neighbors = unknown_neighbors + 1
                homogenity_distribution.append(float(neighbors_with_expected_class) / float(known_neighbors) if known_neighbors > 0 else 0)
                known_neighbors_distribution.append(known_neighbors)
                unknown_neighbors_distribution.append(unknown_neighbors)
                node_degree_distribution.append(known_neighbors + unknown_neighbors)
                node_ids.append(node_id)
        return homogenity_distribution, known_neighbors_distribution, unknown_neighbors_distribution, node_degree_distribution, node_ids

    def preprocess_for_evaluation(self):
        ids_to_remove = []
        components = sorted(nx.connected_components(self.synthetic), key=len, reverse=True)
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




    
    if __name__ == '__main__':        
        print 
