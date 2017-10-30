'''
Created on 19.02.2014

@author: apopiel
'''
import numpy as np


class LoopyBeliefPropagation:


    def __init__(self):
        '''
        Constructor
        '''
        
        '''
        Algorithm method
        
        @param adjMat: - maciez adjecencji; 1 w maciezy oznacza polaczenie od wierzcholka oznaczonego indeksem kolumny do wierzcholka oznaczonego indeksem wiersza
        @param classMat: - pionowy wektor klas (binarnych) rozszerzony do macierzy (rzedy - wezly, kolumny - klasy): 1 - jest ta etykieta, 0 - nie ma tej etykiety, 0.5 nie wiem
        @param repetitions: - warunek stopu
        @param trainingInstances: - lista indekow do wierzcholkow, ktorych klasy znamy
        @param testingInstances: - lista indeksow do wierzcholkow, ktoeych klas nie znamy
        
        '''
    def lbp(self, adjMat, classMat, repetitions, epsilon, trainingInstances, testingInstances):
        '''
        calculate phi, which will be fixed for whole propagation. Phi has meaning only for test nodes.
        For training nodes we will write values from class matrix
        '''
        phi = classMat.copy()
        for i in testingInstances:
            row = adjMat[i,:]
            row[0, testingInstances] = 0
            temp_neg = 1
            temp_pos = 1
            for elem in xrange(0, row.shape[1]):
                if row[0, elem] <> 0:
                    temp_neg = temp_neg * classMat[elem, 0]
                    temp_pos = temp_pos * classMat[elem, 1]
            phi[i,0] = classMat[i,0] * temp_neg
            phi[i,1] = classMat[i,1] * temp_pos

        messages = np.full(classMat.shape, 1)
        psi = [[0.9, 0.1], [0.1, 0.9]]
        for k in range(0, repetitions):
            pre_messages = messages.copy()
            messages = np.full(classMat.shape, 1)
            for i in testingInstances:
                sum = [0.0, 0.0]
                for j in xrange(0, 2): #number of classes
                    sum[0] = sum[0] + (psi[0][j] * phi[i,j])
                    sum[1] = sum[1] + (psi[1][j] * phi[i,j])
                sum[0] = sum[0] * pre_messages[i, 0]
                sum[1] = sum[1] * pre_messages[i, 1]
                pre_new_sum = self.normalize(sum)
                neighbours = adjMat[i,:]
                neighbours[0, trainingInstances] = 0
                for n in xrange(0, neighbours.shape[1]):
                    if neighbours[0,n] >= 1:
                        messages[n,0] = pre_new_sum[0] * messages[n, 0]
                        messages[n,1] = pre_new_sum[1] * messages[n, 1]
                        sum = [0.0, 0.0]
                        row = messages[n,:]
                        sum[0] = row[0]
                        sum[1] = row[1]
                        new_sum = self.normalize(sum)
                        messages[n,0] = new_sum[0]
                        messages[n,1] = new_sum[1]
            if (self.stopConditionReached(messages-pre_messages, epsilon)):
                break
        beliefs = classMat.copy()
        for i in testingInstances:
            beliefs[i,0] = phi[i,0] * messages[i,0]
            beliefs[i,1] = phi[i,1] * messages[i,1]
            row = beliefs[i,:]
            new_sum = self.normalize(row)
            beliefs[i,0] = new_sum[0]
            beliefs[i,1] = new_sum[1]
        return beliefs, k
            
    def stopConditionReached(self, delta, epsilon):
        absVal = np.abs(delta)
        maxRow = np.max(absVal, 0)
        maxValue = np.max(maxRow)
        return maxValue<epsilon
               
                    
    def normalize(self, sum):
        new_sum = [0.0, 0.0]
        faktor = sum[0] + sum[1]
        if faktor == 0.0:
            new_sum[0] = 0.5
            new_sum[1] = 0.5
        else:
            norm_faktor = 1.0/faktor
            new_sum[0] = sum[0] * norm_faktor
            new_sum[1] = sum[1] * norm_faktor
        return new_sum