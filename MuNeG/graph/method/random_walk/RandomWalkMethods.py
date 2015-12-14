__author__ = 'Adrian'

from graph.method.common.CommonUtils import CommonUtils
from graph.method.common.XValWithSampling import XValMethods
from graph.method.random_walk.RwpIterative import RwpIterative
from graph.method.lbp.LBPTools import LBPTools
class RandomWalkMethods():

    def __init__(self):
        pass

    def random_walk_classical(self, graph, default_class_mat, layers, number_of_folds, method_type, percent_of_known_nodes=0.0):
        common = CommonUtils()

        nodes = graph.nodes()
        nr_of_nodes = len(nodes)
        items = nodes if method_type == 1 else range(nr_of_nodes)
        x_val_methods = XValMethods(graph)
        x_val = x_val_methods.stratifies_x_val if method_type == 1 else common.k_fold_cross_validation

        tools = LBPTools(number_of_folds, graph, default_class_mat)
        tools.separate_layer(graph, layers, default_class_mat)
        method = RwpIterative(tools.graphs)
        for training, validation in x_val(items, number_of_folds, percent_of_known_nodes):
            classMat, adjMat, sortedNodes = common.prepareFoldClassMat(graph, default_class_mat, validation)
            results = method.random_walk(graph, classMat, 1, 100, 1000)
            results_dict = method.prepare_classification_results(results)
            accuracy = method.calculate_accuracy(graph, results_dict)
            print accuracy
