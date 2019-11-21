import networkx as nx
import numpy as np
import scipy
import sklearn.metrics as metrics
from graph.method.lbp.FlatLBP import FlatLBP
from graph.method.lbp.NetworkUtils import NetworkUtils
from graph.reader.Airline2016.Airline2016Reader import Airline2016Reader
from graph.reader.Cora.CoraReader import CoraReader
from graph.reader.HighSchool.HighSchoolReader import HighSchoolReader
from graph.reader.HospitalWard.HospitalWardReader import HospitalWardReader


class DecisionFusion:
    number_of_folds = 0
    fprs_per_method = {}
    tprs_per_method = {}
    realGraph = nx.Graph()
    folds = []
    nu = NetworkUtils()
    realGraphClassMat = []
    realNrOfClasses = 0
    realFlatResult = []
    realFlatScores = []
    realLabels = []
    homogenities_during_experiment = {}


    def __init__(self, fold):
        self.number_of_folds = fold
        self.fprs_per_method = {}
        self.tprs_per_method = {}

    def readRealData(self, data, multilayer=False):
        if data == 'HighSchool':
            reader = HighSchoolReader()
        elif data == 'HospitalWard':
            reader = HospitalWardReader()
        elif data == 'Cora':
            reader = CoraReader()
        elif data == 'Airline':
            reader = Airline2016Reader(multilayer=multilayer)
        reader.read()
        self.realGraph = reader.graph

    def flatLBP(self):
        flatLBP = FlatLBP()
        nrOfNodes = self.realGraph.nodes().__len__()
        fold_sum, self.realFlatResult, self.homogenities_during_experiment = flatLBP.start(self.realGraph, nrOfNodes, self.realGraphClassMat, self.realNrOfClasses, 25, 0.001, self.number_of_folds, 0, 1, self.folds)
        self.realFlatScores = [[element[i] for i in xrange(1, self.realGraphClassMat.shape[1]+1)] for element in fold_sum]

    def evaluation(self):
        self.realLabels = self.prepareOriginalLabels(self.realGraphClassMat, self.realNrOfClasses)
        ids_to_remove = self.preprocess_for_evaluation()

        new_labels = []
        for id in xrange(0, len(self.realLabels)):
            if not id in ids_to_remove:
                new_labels.append(self.realLabels[id])
        new_reduction_scores = []
        if not all(l == 0 for l in new_labels):
            for id in xrange(0, len(self.realLabels)):
                if not id in ids_to_remove:
                    new_reduction_scores.append(self.realFlatScores[id])
        else:
            new_labels = self.realLabels
            for id in xrange(0, len(self.realLabels)):
                new_reduction_scores.append(self.realFlatScores[id])
        for c_id in xrange(0, self.realNrOfClasses):
            self.append_roc_rates_for_average([score[c_id] for score in new_reduction_scores], [1 if l == c_id else 0 for l in new_labels], str(c_id))
        self.append_roc_rates_for_average([score[c_id]  for c_id in xrange(0, self.realNrOfClasses) for score in new_reduction_scores], [1 if l == c_id else 0 for c_id in xrange(0, self.realGraphClassMat.shape[1]) for l in new_labels], "micro")
        self.append_macro_results()

    def prepareOriginalLabels(self, defaultClassMat, nrOfClasses):
        classMatForEv = []
        for i in range(0, defaultClassMat.__len__()):
            maxi = 0
            for j in range(1, nrOfClasses):
                if (defaultClassMat[i][j] > defaultClassMat[i][maxi]):
                    maxi = j
            classMatForEv.append(maxi)
        return classMatForEv

    def append_roc_rates_for_average(self, scores, real_labels, method):
        fpr, tpr, threashold = metrics.roc_curve(real_labels, scores)
        self.tprs_per_method[method] = tpr
        self.fprs_per_method[method] = fpr

    def append_macro_results(self):
        all_fpr = np.unique(np.concatenate([self.fprs_per_method[str(i)] for i in range(self.realNrOfClasses)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.realNrOfClasses):
            mean_tpr += scipy.interp(all_fpr, self.fprs_per_method[str(i)], self.tprs_per_method[str(i)])
        mean_tpr /= self.realNrOfClasses
        self.fprs_per_method["macro"] = all_fpr
        self.tprs_per_method["macro"] = mean_tpr

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

    def preprocess_for_evaluation(self):
        ids_to_remove = []
        components = sorted(nx.connected_components(self.realGraph), key=len, reverse=True)
        for component in components:
            for fold in self.folds:
                ids_in_component = [n.id for n in component]
                if set(sorted(ids_in_component)).issubset(set(fold[1])):
                    ids_to_remove.append(ids_in_component)
        return sorted([item for sublist in ids_to_remove for item in sublist])

