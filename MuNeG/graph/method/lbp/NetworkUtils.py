'''
Created on 13.03.2014

@author: apopiel
'''
import numpy as np
from operator import attrgetter
from sets import Set
class NetworkUtils:
    '''
    classdocs
    '''
    

    
    def __init__(self):
        '''
        Constructor
        '''
    def createClassMat(self, graph):
        classMat = []
        nodes = graph.nodes()
        nrOfClasses = self.calculateNrOfLabels(nodes)
        sortedNodes = sorted(nodes, key=attrgetter('id'))
        for node in sortedNodes:
            row = []
            for i in range(0, nrOfClasses):
                if (node.label == i):
                    row.append(1.0)
                else:
                    row.append(0.0)
            classMat.append(row)
        classMatNumPy = np.asarray(classMat)    
        return classMatNumPy, nrOfClasses    
    
    def calculateNrOfLabels(self, nodes):     
        labelSet = Set([])
        for node in nodes:
            labelSet.add(node.label)
        return labelSet.__len__()