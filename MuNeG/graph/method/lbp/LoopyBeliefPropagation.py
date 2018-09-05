'''
Created on 19.02.2014

@author: apopiel
'''
import math
import time

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
    def lbp(self, adjMat, classMat, repetitions, epsilon, trainingInstances, testingInstances, psi = [[0.9, 0.1], [0.1, 0.9]]):
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
        # psi = self.calculate_psi_based_on_homogenity(adjMat, classMat, trainingInstances, testingInstances)
        messages = np.full(classMat.shape, 1)
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

    def calculate_psi_based_on_homogenity(self, adjMat, classMat, trainingInstances, testingInstances):
        start = time.time()
        print("start of psi calculation: " + str(start))
        #arrays with adj rows for all known nodes
        neighbours_of_known_nodes = [adjMat[i,:] for i in trainingInstances]
        #filter test nodes in previously calculated array
        map(lambda n: self.set_no_connection(n, testingInstances), neighbours_of_known_nodes)

        #set 1 where there is a connection to known node(as default weights are set)
        known_neighbours_of_known_nodes = [n.copy() for n in neighbours_of_known_nodes]
        map(lambda n : self.remove_weights(n, trainingInstances), known_neighbours_of_known_nodes)


        #set 1 when neighbour have same class, 0 otherwise
        map(lambda (i, n): self.neigbours_with_same_class(n, classMat, classMat[trainingInstances[i],:]), enumerate(neighbours_of_known_nodes))
        sum_of_known_nodes_with_with_same_class = [sum([n[0,elem] for elem in xrange(n.shape[1])]) for n in neighbours_of_known_nodes]
        # number_of_neighbours_for_known_nodes =
        sum_of_all_neighbours = [sum([n[0,elem] for elem in xrange(n.shape[1])]) for n in known_neighbours_of_known_nodes]
        homogenity = [float(sum_of_known_nodes_with_with_same_class[i]/float(n)) if n <> 0 else float('nan') for i, n in enumerate(sum_of_all_neighbours)]
        homogenity = filter(lambda elem : not math.isnan(elem), homogenity)
        if len(homogenity) > 0:
            avg_homogenity = float(sum(homogenity))/float(len(homogenity))
        else:
            print "No connections between known nodes. I use default value of psi"
            avg_homogenity = 0.9
        end = time.time()
        print("time of calculation: " + str(end - start))
        return  [[avg_homogenity, 1.0-avg_homogenity], [1.0-avg_homogenity, avg_homogenity]]

    def set_no_connection(self, neighbours, testingInstances):
        neighbours[0, testingInstances] = 0

    def remove_weights(self, neighbours, trainingInstances):
        map(lambda i : self.reduct_weight(neighbours, i), trainingInstances)

    def reduct_weight(self, n, i):
        if n[0,i] <> 0:
            n[0,i] = 1
        else:
            n[0,i] = 0

    def neigbours_with_same_class(self, n, classMat, row_of_n):
        same_classes = [1 if classMat[x,0] == row_of_n[0]
                             and  classMat[x,0] == row_of_n[0]
                             and n[0,x] <> 0
                        else 0
                        for x in xrange(0, n.shape[1])]
        map(lambda (i, x) : self.assign_value_to_neighbours(n, i, x), enumerate(same_classes))

    def assign_value_to_neighbours(self, n, i, x):
        n[0,i] = 1 if x == 1 else 0



