__author__ = 'Adrian'

import networkx as nx

class XValMethods():

    graph = nx.MultiGraph()

    def __init__(self, graph):
        self.graph = graph

    def loading(self):
        degree = self.graph.degree_iter()
        pass
