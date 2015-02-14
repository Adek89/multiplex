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
    nodes = set([])
    id = 0
    nodes = dict([])


    def __init__(self):
        db = MySQLdb.connect(host="156.17.131.242", # your host, usually localhost
                     user="mkuli", # your username
                      passwd="trudnocorobic", # your password
           #           db="mkuli") # name of the data base
                    cursorclass = cursors.SSCursor)

        self.cur = db.cursor()

        # Use all the SQL you like
        self.cur.execute("SELECT count(*) FROM test.salon24com")

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

    def addToGraph(self, node, receipient, graph, url):
        wasAuthor = False
        if receipient.name == 'autor':
            receipient.name = self.recognizeAuthor(url)
            wasAuthor = True
        node, receipient = self.searchForExistance(node, receipient)
        graph.add_edge(node, receipient)
        if wasAuthor:
            receipient.label = 1

    def recognizeAuthor(self, url):
        start = url.find('http://') + 7
        end = url.find('.salon24.pl', start)
        author = url[start:end]
        return author

    def loadCategory(self, counter, row):
        url = row[3].lower()
        node = Salon24Node(row[0].lower())
        receipient = Salon24Node(row[1].lower())
        category = row[2]
        if category == 'http://www.salon24.pl/c/3,news':
            self.addToGraph(node, receipient, self.politicsGraph, url)
        elif category == 'http://www.salon24.pl/c/38,kosciol':
            self.addToGraph(node, receipient, self.churchGraph, url)
            print 'church'
        else:
            self.addToGraph(node, receipient, self.othersGraph, url)
            print 'other'
        row = self.cur.fetchone()
        counter = counter + 1
        if counter % 10 == 0:
            print counter
        return row

    def loadDataFromDB(self):
        self.cur.execute(" select AuthorName, AnswerTo, Category, URL from test.salon24com t1 where category = 'http://www.salon24.pl/c/3,news' and  exists (select 1 from test.salon24com t2  where t1.AuthorName = t2.AuthorName and t1.AnswerTo = t2.AnswerTo and t2.Category = 'http://www.salon24.pl/c/38,kosciol')")
        row = self.cur.fetchone()
        counter = 0
        while row is not None:
            # print self.politicsGraph.nodes()
            # print nx.adjacency_matrix(self.politicsGraph)

            row = self.loadCategory(counter, row)
        self.cur.execute("select AuthorName, AnswerTo, Category, URL from test.salon24com t1 where category = 'http://www.salon24.pl/c/38,kosciol' and exists (select 1 from test.salon24com t2 where t1.AuthorName = t2.AuthorName and t1.AnswerTo = t2.AnswerTo and t2.Category = 'http://www.salon24.pl/c/3,news');")
        row = self.cur.fetchone()
        while row is not None:
            # print self.politicsGraph.nodes()
            # print nx.adjacency_matrix(self.politicsGraph)

            row = self.loadCategory(counter, row)
        self.cur.execute("SELECT AuthorName, AnswerTo, Category, URL FROM test.salon24com WHERE CATEGORY <> 'http://www.salon24.pl/c/38,kosciol' AND CATEGORY <> 'http://www.salon24.pl/c/3,news'")
        row = self.cur.fetchone()
        while row is not None:
            # print self.politicsGraph.nodes()
            # print nx.adjacency_matrix(self.politicsGraph)

            row = self.loadCategory(counter, row)

    def addLayerToGraph(self, adjMat, sortedNodes, layer, weight):
        for i in range(0, adjMat.shape[0]):
            row = adjMat[i]
            sumRow = np.sum(row)
            nodeA = sortedNodes[i]
            for j in range(0, adjMat.shape[1]):
                nodeB = sortedNodes[j]
                value = adjMat[i, j] / sumRow
                if (value != 0):
                    self.graph.add_edge(nodeA, nodeB, weight=weight, conWeight=value, layer=layer)

    def createMultiplex(self):
        print self.politicsGraph.nodes().__len__()
        sortedPoliticalComentators = sorted(self.politicsGraph.nodes())
        politicalComentatorsAdjMat = nx.adjacency_matrix(self.politicsGraph, sortedPoliticalComentators)
        sortedChurchComentators = sorted(self.churchGraph.nodes())
        churchComentatorsAdjMat = nx.adjacency_matrix(self.churchGraph, sortedChurchComentators)
        otherComentators = sorted(self.othersGraph.nodes())
        otherAdjMat = nx.adjacency_matrix(self.othersGraph, otherComentators)
        self.addLayerToGraph(politicalComentatorsAdjMat, sortedPoliticalComentators, 'L1', 1)
        self.addLayerToGraph(churchComentatorsAdjMat, sortedChurchComentators, 'L2', 2)
        self.addLayerToGraph(otherAdjMat, otherComentators, 'L3', 3)

    def createNetwork(self):
        self.loadDataFromDB()
	print self.politicsGraph.nodes().__len__()
	print self.politicsGraph.edges().__len__()
	print self.churchGraph.nodes().__len__()
	print self.churchGraph.edges().__len__()
	print self.othersGraph.nodes().__len__()
	print self.othersGraph.edges().__len__()
        self.createMultiplex()
	print self.graph.nodes().__len__()
	print self.graph.edges().__len__()
        return self.graph

