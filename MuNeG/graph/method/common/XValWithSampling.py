__author__ = 'Adrian'

import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.cross_validation import StratifiedKFold

class XValMethods():

    graph = nx.MultiGraph()

    def __init__(self, graph):
        self.graph = graph

    def stratifies_x_val(self, nodes, nr_of_folds):
        sorted_nodes = sorted(nodes, key= lambda node: node.id)
        y = [node.label for node in sorted_nodes]
        str = StratifiedKFold(y, n_folds=nr_of_folds, shuffle=True)
        folds = []
        for train_index, test_index in str:
            folds.append((train_index.tolist(), test_index.tolist()))
        return folds


    #1 - degree
    #2 - clustering
    def sort_by_measure(self, measures_all, measure):
        measures_sorted = sorted(measures_all, key=lambda row: row[:, measure])
        rows_numbers = measures_all.shape[0]
        measures_sorted_matrix = sp.csr_matrix((rows_numbers, 3), dtype=np.double)
        for i in xrange(0, measures_sorted.__len__()):
            measures_sorted_matrix[i, :] = measures_sorted[i]
        return  measures_sorted_matrix

    def create_sample_list(self, measure):
       measures_all = self.loading_whole_graph()
       measures_sorted = self.sort_by_measure(measures_all, measure)
       id_list_sorted = [int(row[0,0]) for row in measures_sorted[:,0]]
       return id_list_sorted

    def loading_surveyed_matrix(self, x_list):
        degree = self.graph.degree(x_list)
        flatted_graph = self.flatGraph(self.graph)
        clustering = nx.clustering(flatted_graph, x_list)
        matrix = self.create_matrix(x_list, clustering, degree)
        return self.normalize(matrix)

    def create_matrix(self, nodes, clustering, degree):
        sorted_nodes = sorted(nodes, key=lambda node: node.id)
        measures_all = np.ndarray(shape=(nodes.__len__(), 3), dtype=np.double)
        measures_all[:, 0] = [node.id for node in sorted_nodes]
        degree_tuples = [deg for deg in degree.items()]
        sorted_degrees = sorted(degree_tuples, key=lambda deg: deg[0].id)
        measures_all[:, 1] = [deg[1] for deg in sorted_degrees]
        clustering_tuples = [clust for clust in clustering.items()]
        sorted_clustering = sorted(clustering_tuples, key=lambda clust: clust[0].id)
        measures_all[:, 2] = [clust[1] for clust in sorted_clustering]
        matrix_all = sp.csr_matrix(measures_all)
        return matrix_all

    def loading_whole_graph(self):
        degree = self.graph.degree()
        flatted_graph = self.flatGraph(self.graph)
        clustering = nx.clustering(flatted_graph)

        all_nodes = self.graph.nodes()
        matrix_all = self.create_matrix(all_nodes, clustering, degree)
        return self.normalize(matrix_all)

    def flatGraph(self, graph):
        G = nx.Graph()
        for u, v, data in graph.edges_iter(data=True):
            w = data['weight']
            if G.has_edge(u, v):
                G[u][v]['weight'] += w
            else:
                G.add_edge(u, v, weight=w)
        return G

    def normalize(self, matrix):
         matrix[:, 1] = matrix[:, 1]/max(max(matrix[:,1]))[0,0]
         matrix[:, 2] = matrix[:, 2]/max(max(matrix[:,2]))[0,0]
         return matrix
