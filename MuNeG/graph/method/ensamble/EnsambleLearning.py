__author__ = 'Adek'

import numpy as np
import networkx as nx
import Queue
import random
from graph.method.ensamble.Edge import Edge
from graph.method.ica.SingleModelICA import SingleModelICA
from graph.gen.Node import Node
from graph.reader.Salon24.Salon24Node import Salon24Node
import graph.method.ica.ClassifiersUtil as cu
class EnsambleLearning:

    graph = nx.MultiGraph()
    models = []
    ensambleSet = set([])
    nrOfNodesInSubgraph = 0
    PERCENT_TRAINING = 0.2
    copiedNodesId = set([])
    copiedNodes = dict([])

    def __init__(self, graph, models, nrOfNodesInSubgraph):
        self.graph = graph
        self.models = models
        self.ensambleSet = set([])
        self.nrOfNodesInSubgraph = nrOfNodesInSubgraph

    def ensamble(self):
        for i in range(0, self.models.__len__()):
            sampledGraph = self.sampleGraph()
            model = self.learnModel(sampledGraph, self.models[i])
            self.ensambleSet.add(model)
        return self.ensambleSet, self.graph

    def createSampledGraph(self, sampledEdges, sampledGraph):
        newIds = 0
        for id, node in self.copiedNodes.items():
            node.id = newIds
            newIds += 1
        for edge in sampledEdges:
            edgeData = edge.data
            sampledGraph.add_edge(edge.node1, edge.node2, weight=edgeData['weight'], layer=edgeData['layer'],
                                  conWeight=edgeData['conWeight'])
        copiedNodesSet = set([el[1] for el in self.copiedNodes.items()])
        nodesWithoutEdge = copiedNodesSet.difference(set(sampledGraph.nodes()))
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

    def assignCopyOrCreate(self, copiedNodes, copiedNodesId, node):
        if (node.id in copiedNodesId):
            tempNode = copiedNodes[node.id]
        else:
            if isinstance(node, Node):
                tempNode = Node(node.group, node.label, node.id)
                copiedNodesId.add(node.id)
                copiedNodes.update({node.id: tempNode})
            else:
                tempNode = Salon24Node(node.name)
                tempNode.id = node.id
                tempNode.label = node.label
                copiedNodesId.add(node.id)
                copiedNodes.update({node.id: tempNode})
        return tempNode

    def collectEdges(self, edges, sampledEdges):
        for edge in edges:
            node1 = edge[0]
            tempNode1 = self.assignCopyOrCreate(self.copiedNodes, self.copiedNodesId, node1)
            node2 = edge[1]
            tempNode2 = self.assignCopyOrCreate(self.copiedNodes, self.copiedNodesId, node2)
            data = edge[2]
            edgeObj = Edge(tempNode1, tempNode2, data)
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
        self.createSampledGraph(sampledEdges, sampledGraph)
        return sampledGraph


    def learnModel(self, sampledGraph, model):
        ica = SingleModelICA(sampledGraph, self.PERCENT_TRAINING, 5, model)
        ica.classify()
        return ica.classifier
