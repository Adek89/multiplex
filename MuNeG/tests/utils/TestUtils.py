__author__ = 'Adrian'
import numpy as np
from graph.gen.Group import Group
from graph.gen.Node import Node
class TestUtils():

    def prepareNodesAndEdges(self):
        groupRed = Group('r', 1)
        groupBlue = Group('b', 2)
        node1 = Node(groupRed, 1, 0)
        node2 = Node(groupBlue, 0, 1)
        node3 = Node(groupRed, 1, 2)
        node4 = Node(groupBlue, 0, 3)
        node5 = Node(groupBlue, 1, 4)
        nodes = ({node1, node2, node3, node4, node5})
        nodeList = [node1, node2, node3, node4, node5]
        edge1 = dict([('layer', 'L1'), ('conWeight', 0.5), ('weight', 1)])
        edge2 = dict([('layer', 'L2'), ('conWeight', 0.5), ('weight', 2)])
        edgesData = ([edge1, edge2])
        return edgesData, nodes, nodeList

    def generateEdges(self, nrOfEdges, nodes, edgesData):
            i = 0
            for i in range(0, nrOfEdges):
                if (i == 0):
                    yield (nodes[0], nodes[2], edgesData[0])
                elif (i == 1):
                    yield (nodes[1], nodes[3], edgesData[0])
                elif (i == 2):
                    yield (nodes[2], nodes[4], edgesData[0])
                elif (i == 3):
                    yield (nodes[0], nodes[1], edgesData[0])
                elif (i == 4):
                    yield (nodes[1], nodes[4], edgesData[0])
                elif (i == 5):
                    yield (nodes[0], nodes[2], edgesData[1])
                elif (i == 6):
                    yield (nodes[1], nodes[3], edgesData[1])
                elif (i == 7):
                    yield (nodes[1], nodes[4], edgesData[1])
                elif (i == 8):
                    yield (nodes[0], nodes[1], edgesData[1])
                elif (i == 9):
                    yield (nodes[2], nodes[3], edgesData[1])

    def prepareTestClassMat(self):
        defaultClassMat = np.ndarray(shape=(5, 2))
        defaultClassMat[0] = [0, 1]
        defaultClassMat[1] = [1, 0]
        defaultClassMat[2] = [0, 1]
        defaultClassMat[3] = [1, 0]
        defaultClassMat[4] = [0, 1]
        return defaultClassMat

    def prepareTestClassMatWithUnknownNodes(self):
        testClassMat = np.ndarray(shape=(5, 2))
        testClassMat[0] = [0, 1]
        testClassMat[1] = [1, 0]
        testClassMat[2] = [0, 1]
        testClassMat[3] = [1, 0]
        testClassMat[4] = [0.5, 0.5]
        return testClassMat

    def prepareEdgesList(self, edges, nodesList):
        edgesList = [(nodesList[0], nodesList[2], edges[0]),
                     (nodesList[1], nodesList[3], edges[0]),
                     (nodesList[2], nodesList[4], edges[0]),
                     (nodesList[0], nodesList[1], edges[0]),
                     (nodesList[1], nodesList[4], edges[0]),
                     (nodesList[0], nodesList[2], edges[1]),
                     (nodesList[1], nodesList[3], edges[1]),
                     (nodesList[1], nodesList[4], edges[1]),
                     (nodesList[0], nodesList[1], edges[1]),
                     (nodesList[2], nodesList[3], edges[1])]
        return edgesList