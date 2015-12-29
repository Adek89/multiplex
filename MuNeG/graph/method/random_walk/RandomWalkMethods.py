__author__ = 'Adrian'

from graph.method.common.CommonUtils import CommonUtils
from graph.method.common.XValWithSampling import XValMethods
from graph.method.random_walk.RwpIterative import RwpIterative
from graph.method.lbp.LBPTools import LBPTools

import networkx as nx
import random as rand
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
        results_dict = {}
        removed_ids = []
        for training, validation in x_val(items, number_of_folds, percent_of_known_nodes):
            training, validation, removed_ids = self.find_and_correct_unknown_components(graph, training, validation, removed_ids)
            self.execute_experiment(common, default_class_mat, graph, layers, method, results_dict, validation)
        self.execute_experiment(common, default_class_mat, graph, layers, method, results_dict, sorted(removed_ids))
        results_dict = method.prepare_classification_results(results_dict)
        sorted_results = sorted(results_dict.keys(), key=lambda item : item.id)
        labels_list = [results_dict[res] for res in sorted_results]
        return labels_list

    def find_and_correct_unknown_components(self, graph, training, validation, removed_ids):
        components = nx.connected_components(graph)
        validation_set = set(validation)
        for component in components:
            component_ids = [node.id for node in component]
            ids_set = set(component_ids)
            if (ids_set.issubset(validation_set)):
                chosen_id = rand.sample(ids_set, 1)
                validation.remove(chosen_id[0])
                training = sorted(training + chosen_id)
                removed_ids = removed_ids + chosen_id
        return training, validation, removed_ids

    def execute_experiment(self, common, default_class_mat, graph, layers, method, results_dict, validation):
        classMat, adjMat, sortedNodes = common.prepareFoldClassMat(graph, default_class_mat, validation)
        results = method.random_walk(graph, classMat, layers, 1, 100)
        results_dict.update(results)
