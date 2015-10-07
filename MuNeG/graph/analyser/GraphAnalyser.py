from networkx.algorithms.bipartite.cluster import clustering
from sqlalchemy.sql.functions import coalesce

__author__ = 'Adek'
import networkx as nx
from graph.gen.GraphGenerator import GraphGenerator
from graph.reader.Salon24.Salon24Reader import Salon24Reader
from graph.method.ensamble.EnsambleLearning import EnsambleLearning
import csv
import time
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
class GraphAnalyser:

    nrOfNodes = 500
    nrOfGroups = 9
    avgNrOfGroups = 9
    grLabelHomogenity = 5
    probOfEdgeInSameGroup = 5
    probOfEdgeInOtherGrops = 0.1
    layerWeights = [1, 2]
    layerName = ["L1", "L2"]
    nrOfLayers = 2
    percentOfTrainingNodes = 0.2
    counter = 0
    graph = None
    # FILE_PATH = "/home/apopiel/tmp_local/"
    FILE_PATH = ""

    def __init__(self, graph, percentOfTrainignNodes, counter):
        self.percentOfTrainingNodes = percentOfTrainignNodes
        self.counter = counter
        self.graph = graph

    def prepareNumberOfGroups(self, nrOfNodes, nrOfGroups):
        while True:
            dividedInt = nrOfNodes % nrOfGroups
            if (not dividedInt == 0):
                nrOfNodes = nrOfNodes + 1
            else:
                break
        self.nrOfNodes = nrOfNodes
        self.avgNrOfGroups = nrOfNodes / nrOfGroups

    def initLayers(self, nrOfLayers):
        for i in xrange(0, nrOfLayers):
            self.layerWeights.append(i + 1)
            self.layerName.append("L"+str(i+1))

    def flatGraph(self, graph):
        G = nx.Graph()
        for u, v, data in graph.edges_iter(data=True):
            w = data['weight']
            if G.has_edge(u, v):
                G[u][v]['weight'] += w
            else:
                G.add_edge(u, v, weight=w)
        return G

    def getTrianglesList(self, G):
        triangles = nx.triangles(G)
        trianglesList = [triangles[k] for k in triangles]
        return trianglesList

    def getSquares(self, graph):
        squareDict = nx.square_clustering(graph)
        square = [squareDict[k] for k in squareDict]
        return square

    def getClustering(self, G):
        clustering = nx.clustering(G)
        clusteringList = [clustering[k] for k in clustering]
        return clusteringList

    def getDiameterAndShortestPath(self, graph):
        connComponents = nx.connected_components(graph)
        diameter = []
        shortestPath = []
        stop = sum(1 for x in connComponents)
        connComponents = nx.connected_components(graph)
        for k in xrange(0, stop):
            component = connComponents.next()
            graph_subgraph = graph.subgraph(component)
            diameter.append(nx.diameter(graph_subgraph))
            shortestPath.append(nx.average_shortest_path_length(graph_subgraph))
        return diameter, shortestPath

    def analyse(self):
        gg =  GraphGenerator(self.nrOfNodes, self.avgNrOfGroups, self.layerWeights,
                                 self.grLabelHomogenity, self.probOfEdgeInSameGroup,
                                 self.probOfEdgeInOtherGrops, self.layerName)
        graph = gg.generate()
        degree_sequence, dmax = self.drawDegreeDistribution(graph) #1
        nrOfEdges = graph.number_of_edges() #2
        avgDegree = float(sum(degree_sequence))/float(len(degree_sequence)) #3
        G = self.flatGraph(graph)
        trianglesList = self.getTrianglesList(G)
        nrOfTriangles = sum(trianglesList) #4
        clusteringList = self.getClustering(G)
        avgClustering = sum(clusteringList)/len(clusteringList) #6
        diameter, shortestPath = self.getDiameterAndShortestPath(graph)
        avgDiameter = float(sum(diameter))/float(len(diameter))
        avgShortestPath = sum(shortestPath)/len(shortestPath)
        layers = set([ (edata['layer']) for u,v,edata in graph.edges(data=True)])

        with open(self.FILE_PATH + 'graphparams' + str(self.counter) + '.csv', 'ab') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow([graph.nodes().__len__(), layers.__len__(), self.percentOfTrainingNodes, nrOfEdges, avgDegree,
                                dmax,
                                nrOfTriangles,
                                0, # avgSquareCount,
                                avgClustering,
                                avgDiameter, avgShortestPath, degree_sequence])


    def drawDegreeDistribution(self, graph):
        degree_sequence = sorted(nx.degree(graph).values(), reverse=True)  # degree sequence
        dmax = max(degree_sequence)
        stats.mstats.normaltest(degree_sequence)
        return degree_sequence, dmax

