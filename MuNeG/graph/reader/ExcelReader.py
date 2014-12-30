'''
Created on 2 mar 2014

@author: Adek
'''
import xlrd
import networkx as nx
from os.path import dirname
from sets import Set
from graph.reader.ExcelNode import ExcelNode
import numpy as np
class ExcelReader:
    nodes = Set([])
    graph = nx.MultiGraph()
    
    messagesGraph = nx.MultiGraph()
    transactionCntGraph = nx.MultiGraph()
    transactionValueGraph = nx.MultiGraph()


    def __init__(self):
        '''
        Constructor
        '''
        
    def read(self, sheetName):
        directory = dirname(__file__)
        book = xlrd.open_workbook(directory + '\\acta_vir_v1.xlsx');
        sheet = book.sheet_by_name(sheetName)
        columnsNodeA = [1, 5, 6, 7]
        columnsNodeB = [2, 8, 9, 10]
        columnsEdges = [11, 12, 13, 14, 15, 16]
        index = 0
        for rowIndex in range(sheet.nrows - 1):
            row = rowIndex + 1
            nodeA, index = self.createExcelNode(sheet, row, columnsNodeA, index)
            nodeB, index = self.createExcelNode(sheet, row, columnsNodeB, index)
            self.createTempGraphs(sheet, row, columnsEdges, nodeA, nodeB)
        self.createEdgeWithParams()
        print nx.adjacency_matrix(self.graph)
        return self.graph
            
                    
    def createExcelNode(self, sheet, row, columns, index):
        node = ExcelNode()
        node.idFromFile = self.getValueFromCell(sheet, row, columns[0])
        if (node.idFromFile not in self.nodes):
            node.timicPoints = self.getValueFromCell(sheet, row, columns[1])
            if (node.timicPoints < 745):
                node.label = 0
            else:
                node.label = 1    
            node.gender = self.getValueFromCell(sheet, row, columns[2])
            node.age = self.getValueFromCell( sheet, row, columns[3])
            node.id = index
            self.nodes.add(node.idFromFile)
            self.graph.add_node(node)
            index = index  + 1
        else: 
            for currNode in self.graph.nodes():
                if (currNode.idFromFile == node.idFromFile):
                    node = currNode
                    break
        return node, index
    
    def createTempGraphs(self, sheet, row, columns, nodeA, nodeB):
        messagesAB = self.getValueFromCell(sheet, row, columns[0])
        messagesBA = self.getValueFromCell(sheet, row, columns[1])
        transactionsCntAB = self.getValueFromCell(sheet, row, columns[2])
        transactionsCntBA = self.getValueFromCell(sheet, row, columns[3])
        transactionsValueAB = self.getValueFromCell(sheet, row, columns[4])
        transactionsValueBA = self.getValueFromCell(sheet, row, columns[5])
        
        if (messagesAB != 0 or messagesBA != 0):
            self.messagesGraph.add_edge(nodeA, nodeB, weight = messagesAB)
            self.messagesGraph.add_edge(nodeB, nodeA, weight = messagesBA)
        if (transactionsCntAB != 0 or transactionsCntBA != 0):    
            self.transactionCntGraph.add_edge(nodeA, nodeB, weight = transactionsCntAB)
            self.transactionCntGraph.add_edge(nodeB, nodeA, weight = transactionsCntBA)
        if (transactionsValueAB != 0 or transactionsValueBA != 0):
            self.transactionValueGraph.add_edge(nodeA, nodeB, weight = transactionsValueAB)
            self.transactionValueGraph.add_edge(nodeB, nodeA, weight = transactionsValueBA)
        
    
    
    def createEdgeWithParams(self):
        
        sortedMessagesNodes = sorted(self.messagesGraph.nodes())
        messagesAdjMat = nx.adjacency_matrix(self.messagesGraph, sortedMessagesNodes)
        
        sortedTransactionCntNodes = sorted(self.transactionCntGraph.nodes())
        transactionCntAdjMat = nx.adjacency_matrix(self.transactionCntGraph, sortedTransactionCntNodes)
        
        sortedTransactionValueNodes = sorted(self.transactionValueGraph.nodes())
        transactionValueAdjMat = nx.adjacency_matrix(self.transactionValueGraph, sortedTransactionValueNodes)
        
        for i in range(0, messagesAdjMat.shape[0]):
            row = messagesAdjMat[i]
            sumRow = np.sum(row)
            nodeA = sortedMessagesNodes[i]
            for j in range(0, messagesAdjMat.shape[1]):
                nodeB = sortedMessagesNodes[j]
                value = messagesAdjMat[i, j]/sumRow
                if (value != 0):
                    self.graph.add_edge(nodeA, nodeB, conWeight = value, layer='L1')
                
        for i in range(0, transactionCntAdjMat.shape[0]):
            row = transactionCntAdjMat[i]
            sumRow = np.sum(row)
            nodeA = sortedTransactionCntNodes[i]
            for j in range(0, transactionCntAdjMat.shape[1]):
                nodeB = sortedTransactionCntNodes[j]
                if (value != 0):
                    self.graph.add_edge(nodeA, nodeB, conWeight = value, layer='L2')
                
        for i in range(0, transactionValueAdjMat.shape[0]):
            row = transactionValueAdjMat[i]
            sumRow = np.sum(row)
            nodeA = sortedTransactionValueNodes[i]
            for j in range(0, transactionValueAdjMat.shape[1]):
                nodeB = sortedTransactionValueNodes[j]
                if (value != 0):
                    self.graph.add_edge(nodeA, nodeB, conWeight = value, layer='L3')
        
        
    def getValueFromCell(self, sheet, row, col):
        return sheet.cell_value(row, col)    