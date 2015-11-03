__author__ = 'Adrian'

import networkx as nx

class XValMethods():

    graph = nx.MultiGraph()

    def __init__(self, graph):
        self.graph = graph

    def loading_whole_graph(self):
        degree = self.graph.degree()
        flatted_graph = self.flatGraph(self.graph)
        clustering = nx.clustering(flatted_graph)
        pass

    def flatGraph(self, graph):
        G = nx.Graph()
        for u, v, data in graph.edges_iter(data=True):
            w = data['weight']
            if G.has_edge(u, v):
                G[u][v]['weight'] += w
            else:
                G.add_edge(u, v, weight=w)
        return G