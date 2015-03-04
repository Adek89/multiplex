__author__ = 'Adek'

import numpy as np
import networkx as nx
import Queue
import random
from graph.method.ensamble.Edge import Edge
from graph.method.ica.SingleModelICA import SingleModelICA
import graph.method.ica.ClassifiersUtil as cu
class EnsambleLearning:

    graph = nx.MultiGraph()
    nrOfModels = 0
    ensambleSet = set([])
    nrOfNodesInSubgraph = 0
    PERCENT_TRAINING = 0.2

    def __init__(self, graph, nrOfModels, nrOfNodesInSubgraph):
        self.graph = graph
        self.nrOfModels = nrOfModels
        self.ensambleSet = set([])
        self.nrOfNodesInSubgraph = nrOfNodesInSubgraph

    def ensamble(self):
        models = cu.knownModels()
        for i in range(0, self.nrOfModels):
            sampledGraph = self.sampleGraph()
            model = self.learnModel(sampledGraph, models[i])
            self.ensambleSet.add(model)
        return self.ensambleSet

    def createSampledGraph(self, sampledEdges, sampledGraph, sampledNodes):
        newIds = 0
        for node in sampledNodes:
            node.id = newIds
            newIds += 1
        for edge in sampledEdges:
            edgeData = edge.data
            sampledGraph.add_edge(edge.node1, edge.node2, weight=edgeData['weight'], layer=edgeData['layer'],
                                  conWeight=edgeData['conWeight'])
        nodesWithoutEdge = sampledNodes.intersection(set(sampledGraph.nodes()))
        for node in nodesWithoutEdge:
            sampledGraph.add_node(node)

    def sampleNeighborhood(self, nodes, q):
        while (nodes.__len__() < self.nrOfNodesInSubgraph and q.qsize() > 0):
            node = q.get()
            nodes.add(node)
            neighbors = nx.neighbors(self.graph, node)
            neighbors = filter(lambda neighbor: neighbor not in nodes, neighbors)
            for n in neighbors:
                q.put(n)

    def collectEdges(self, edges, sampledEdges):
        for edge in edges:
            node1 = edge[0]
            node2 = edge[1]
            data = edge[2]
            edgeObj = Edge(node1, node2, data)
            sampledEdges.add(edgeObj)

    def sample(self, graphNodes, iterateUnitl, nrOfNodesList, sampledEdges, sampledNodes):
        for s in xrange(0, iterateUnitl):
            nodes = set([])
            edges = set([])
            q = Queue.Queue()
            random.shuffle(nrOfNodesList)
            node = graphNodes[nrOfNodesList[0]]
            nodes.add(node)
            neighbors = nx.neighbors(self.graph, node)
            for n in neighbors:
                q.put(n)
            self.sampleNeighborhood(nodes, q)
            allEdges = self.graph.edges(nodes, data=True)
            edges = filter(lambda edge: edge[0] in nodes and edge[1] in nodes, allEdges)
            sampledNodes = sampledNodes.union(nodes)
            self.collectEdges(edges, sampledEdges)
        return sampledNodes

    def sampleGraph(self):
        graphNodes = self.graph.nodes()
        nrOfNodes = graphNodes.__len__()
        nrOfNodesList = range(0, nrOfNodes)
        iterateUnitl = nrOfNodes / self.nrOfNodesInSubgraph
        sampledNodes = set([])
        sampledEdges = set([])
        sampledGraph = nx.MultiGraph()
        sampledNodes = self.sample(graphNodes, int(iterateUnitl), nrOfNodesList, sampledEdges, sampledNodes)
        self.createSampledGraph(sampledEdges, sampledGraph, sampledNodes)
        return sampledGraph


    def learnModel(self, sampledGraph, model):
        ica = SingleModelICA(sampledGraph, self.PERCENT_TRAINING, 5, model)
        ica.classify()
        return ica.classifier
