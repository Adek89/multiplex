__author__ = 'Adek'

import networkx as nx
import MySQLdb
import MySQLdb.cursors as cursors
from graph.reader.Salon24.Salon24Node import Salon24Node
import numpy as np
class Salon24Reader:


    cur = None
    graph = nx.MultiGraph()
    politicsGraph = nx.MultiGraph()
    churchGraph = nx.MultiGraph()
    othersGraph = nx.MultiGraph()
    id = 0
    nodes = dict([])
    graphs = dict([])
    limit = 0


    def __init__(self, limit):
        self.limit = limit
        db = MySQLdb.connect(host="156.17.131.242", # your host, usually localhost
                     user="mkuli", # your username
                      passwd="trudnocorobic", # your password
           #           db="mkuli") # name of the data base
                    cursorclass = cursors.SSCursor)

        self.cur = db.cursor()

        # Use all the SQL you like
        self.cur.execute("SELECT count(*) FROM test.s24_comments")

        # print all the first cell of all the rows
        for row in self.cur.fetchall() :
            print row[0]

    def assignIdIfNewNode(self, changedNode, changedReceipient, node, receipient):
        if not changedNode:
            node.id = self.id
            self.id = self.id + 1
        if not changedReceipient:
            receipient.id = self.id
            self.id = self.id + 1

    def searchForExistance(self, node, receipient):
        changedNode = False
        changedReceipient = False
        if self.nodes.has_key(node.name):
            node = self.nodes.get(node.name)
            changedNode = True
        else:
            self.nodes.update({node.name: node})
        if self.nodes.has_key(receipient.name):
            receipient = self.nodes.get(receipient.name)
            changedReceipient = True
        else:
            self.nodes.update({receipient.name: receipient})
        # for currNode in tempGraph.nodes():
        #     if (currNode.name == node.name):
        #         node = currNode
        #         changedNode = True
        #     elif (currNode.name == receipient.name):
        #         receipient = currNode
        #         changedReceipient = True
        #     if changedNode and changedReceipient:
        #         break
        self.assignIdIfNewNode(changedNode, changedReceipient, node, receipient)
        return node, receipient

    def addToGraph(self, node, receipient, graph, url, hour):
        wasAuthor = False
        wasJournalist = False
        if hour >= 8 and hour <=16:
            wasJournalist = True
        if receipient.name == 'autor':
            receipient.name = self.recognizeAuthor(url)
            wasAuthor = True
        node, receipient = self.searchForExistance(node, receipient)
        graph.add_edge(node, receipient)
        if wasAuthor:
            node.label = 2
        if not wasAuthor and wasJournalist:
            node.label = 1

    def recognizeAuthor(self, url):
        start = url.find('http://') + 7
        end = url.find('.salon24.pl', start)
        author = url[start:end]
        return author

    def loadCategory(self, counter, row):
        time = row[4]
        url = row[3].lower()
        node = Salon24Node(row[0].lower())
        receipient = Salon24Node(row[1].lower())
        category = row[2]
        currentGraph = None
        if self.graphs.__contains__(category):
            currentGraph = self.graphs.get(category)
        else:
            currentGraph = nx.MultiGraph()
            self.graphs.update({category: currentGraph})
        if time == None:
            hour = 0
        else:
            hour = time.hour
        self.addToGraph(node, receipient, currentGraph, url, hour)
        row = self.cur.fetchone()
        counter = counter + 1
        if counter % 10 == 0:
            print counter
        return row

    def loadDataFromDB(self):
        self.cur.execute("select author_name, answer_to, category, url, full_time from test.s24_comments limit " + str(self.limit))
        row = self.cur.fetchone()
        counter = 0
        while row is not None:
            # print self.politicsGraph.nodes()
            # print nx.adjacency_matrix(self.politicsGraph)

            row = self.loadCategory(counter, row)
            counter += 1

    def addLayerToGraph(self, adjMat, sortedNodes, layer, weight):
        for i in range(0, adjMat.shape[0]):
            row = adjMat[i]
            sumRow =row.sum()
            nodeA = sortedNodes[i]
            for j in range(0, adjMat.shape[1]):
                nodeB = sortedNodes[j]
                value = float(adjMat[i, j]) / float(sumRow)
                if (value != 0):
                    self.graph.add_edge(nodeA, nodeB, weight=weight, conWeight=value, layer=layer)

    def createMultiplex(self):
        iter = 0
        values = sorted(self.graphs.iteritems(), key=lambda x: x[1])
        for value in values:
            iter += 1
            currentGraph = value[1]
            sortedGraph = sorted(currentGraph.nodes())
            adjMat = nx.adjacency_matrix(currentGraph, sortedGraph)
            self.addLayerToGraph(adjMat, sortedGraph, 'L' + str(iter), iter)

    def createNetwork(self):
        self.loadDataFromDB()
        self.createMultiplex()
        return self.graph