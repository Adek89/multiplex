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
        nr_of_classes = classMat.shape[1]
        temp_values = {}
        phi = classMat.copy()
        for i in testingInstances:
            row = adjMat[i,:]
            row[0, testingInstances] = 0
            for c_id in xrange(0, nr_of_classes):
                temp_values.update({c_id:1.0})
            for elem in xrange(0, row.shape[1]):
                if row[0, elem] <> 0:
                    for c_id in xrange(0, nr_of_classes):
                        temp_phi = temp_values[c_id]
                        temp_values.update({c_id:temp_phi * classMat[elem, c_id]})
            for c_id in xrange(0, nr_of_classes):
                phi[i,c_id] = classMat[i,c_id] * temp_values[c_id]
        psi, avg_homogenity = self.calculate_psi_based_on_homogenity(adjMat, classMat, trainingInstances, testingInstances)
        messages = np.full(classMat.shape, 1)
        for k in range(0, repetitions):
            pre_messages = messages.copy()
            messages = np.full(classMat.shape, 1)
            for i in testingInstances:
                sum = [0.0 for c_id in xrange(0, nr_of_classes)]
                for j in xrange(0, nr_of_classes): #number of classes
                    for c_id in xrange(0, nr_of_classes):
                        sum[c_id] = sum[c_id] + (psi[c_id][j] * phi[i,j])
                for c_id in xrange(0, nr_of_classes):
                    sum[c_id] = sum[c_id] * pre_messages[i, c_id]
                pre_new_sum = self.normalize(sum, nr_of_classes)
                neighbours = adjMat[i,:]
                neighbours[0, trainingInstances] = 0
                for n in xrange(0, neighbours.shape[1]):
                    if neighbours[0,n] >= 1:
                        for c_id in xrange(0, nr_of_classes):
                            messages[n,c_id] = pre_new_sum[c_id] * messages[n, c_id]
                        sum = [0.0 for c_id in xrange(0, nr_of_classes)]
                        row = messages[n,:]
                        for c_id in xrange(0, nr_of_classes):
                            sum[c_id] = row[c_id]
                        new_sum = self.normalize(sum, nr_of_classes)
                        for c_id in xrange(0, nr_of_classes):
                            messages[n,c_id] = new_sum[c_id]
            if (self.stopConditionReached(messages-pre_messages, epsilon)):
                break
        phi_correlations = {}
        for c_id in xrange(0, nr_of_classes):
            maximum_class_value = max(psi[c_id])
            maximum_class = psi[c_id].index(maximum_class_value)
            phi_correlations.update({c_id : maximum_class})
        beliefs = classMat.copy()
        for i in testingInstances:
            for c_id in xrange(0, nr_of_classes):
                beliefs[i,c_id] = phi[i,phi_correlations[c_id]] * messages[i,c_id]
            row = beliefs[i,:]
            new_sum = self.normalize(row, nr_of_classes)
            for c_id in xrange(0, nr_of_classes):
                beliefs[i,c_id] = new_sum[c_id]
        return beliefs, k, avg_homogenity
            
    def stopConditionReached(self, delta, epsilon):
        absVal = np.abs(delta)
        maxRow = np.max(absVal, 0)
        maxValue = np.max(maxRow)
        return maxValue<epsilon
               
                    
    def normalize(self, sum_for_all_classes, nr_of_classes):
        new_sum = [0.0 for c_id in xrange(0, nr_of_classes)]
        faktor = sum(sum_for_all_classes)
        if faktor == 0.0:
            self.normalize_based_on_class_number(new_sum, nr_of_classes)
        else:
            norm_faktor = 1.0/faktor #faktor very near 0 gives norm_faktor as inf
            if math.isinf(norm_faktor):
                self.normalize_based_on_class_number(new_sum, nr_of_classes)
            else:
                for c_id in xrange(0, nr_of_classes):
                    new_sum[c_id] = sum_for_all_classes[c_id] * norm_faktor
        return new_sum

    def normalize_based_on_class_number(self, new_sum, nr_of_classes):
        for c_id in xrange(0, nr_of_classes):
            new_sum[c_id] = float(1.0 / nr_of_classes)

    def calculate_psi_based_on_homogenity(self, adjMat, classMat, trainingInstances, testingInstances):
        start = time.time()
        print("start of psi calculation: " + str(start))
        nr_of_classes = classMat.shape[1]
        #arrays with adj rows for all known nodes
        neighbours_of_known_nodes = [adjMat[i,:] for i in trainingInstances]
        #filter test nodes in previously calculated array
        map(lambda n: self.set_no_connection(n, testingInstances), neighbours_of_known_nodes)

        #set 1 where there is a connection to known node(as default weights are set)
        known_neighbours_of_known_nodes = [n.copy() for n in neighbours_of_known_nodes]
        map(lambda n : self.remove_weights(n, trainingInstances), known_neighbours_of_known_nodes)

        #set class id for all neighbours, when there is no connection set -1
        map(lambda (i, n): self.neigbours_classes(n, classMat, nr_of_classes), enumerate(neighbours_of_known_nodes))
        nodes_neighbours_classes_map = {}
        map(lambda (i, n): nodes_neighbours_classes_map.update({i : self.nr_of_neigbours_classes(n, nr_of_classes)}),enumerate(neighbours_of_known_nodes))
        sum_of_all_neighbours = [sum([n[0,elem] for elem in xrange(n.shape[1])]) for n in known_neighbours_of_known_nodes]

        homogenity = {}
        map(lambda (id, neighbour_classes): homogenity.update({id: [float(elem)/float(sum_of_all_neighbours[id]) if sum_of_all_neighbours[id] <> 0 else 1.0/float(nr_of_classes) for elem in neighbour_classes]}), nodes_neighbours_classes_map.iteritems())

        class_mats_of_known_nodes = classMat[trainingInstances,:]
        classes_of_known_nodes = [self.get_class_for_row(class_mats_of_known_nodes[id,:], nr_of_classes)for id in xrange(0, class_mats_of_known_nodes.shape[0])]

        avg_homogenity = {}
        map(lambda c_id: avg_homogenity.update({c_id: []}), xrange(0, nr_of_classes))
        nr_of_nodes_from_classes = [classes_of_known_nodes.count(c_id) for c_id in xrange(0, nr_of_classes)]
        map(lambda (id, h): avg_homogenity.update({classes_of_known_nodes[id] : self.add_homogenities(avg_homogenity[classes_of_known_nodes[id]], h)}) , homogenity.iteritems())
        map(lambda (id, h): avg_homogenity.update({id : [float(elem)/float(nr_of_nodes_from_classes[id]) for elem in h]}), avg_homogenity.iteritems())
        psi = [h for (c_id, h) in avg_homogenity.iteritems()]
        map(lambda (i): self.fill_empty_class(psi[i], i, nr_of_classes), xrange(0,nr_of_classes))
        end = time.time()
        print("time of calculation: " + str(end - start))
        return psi, avg_homogenity

    def add_homogenities(self, exisitng_h, h):
        if len(exisitng_h) == 0:
            return h
        else:
            for i in xrange(0, len(exisitng_h)):
                exisitng_h[i] = exisitng_h[i] + h[i]
            return exisitng_h

    def set_no_connection(self, neighbours, testingInstances):
        neighbours[0, testingInstances] = 0

    def remove_weights(self, neighbours, trainingInstances):
        map(lambda i : self.reduct_weight(neighbours, i), trainingInstances)

    def reduct_weight(self, n, i):
        if n[0,i] <> 0:
            n[0,i] = 1
        else:
            n[0,i] = 0

    def neigbours_classes(self, n, classMat, nr_of_classes):
        neighbours_classes = []
        for x in xrange(0, n.shape[1]):
            if n[0,x] == 0:
                neighbours_classes.append(-1)
            else:
                neighbours_classes.append(self.get_class_for_row(classMat[x,:], nr_of_classes))
        map(lambda (i, x) : self.assign_value_to_neighbours(n, i, x), enumerate(neighbours_classes))


    def get_class_for_row(self, class_mat_row, nr_of_classes):
        node_class = 0
        for c_id in xrange(0, nr_of_classes):
            if class_mat_row[c_id] > class_mat_row[node_class]:
                node_class = c_id
        return node_class

    def assign_value_to_neighbours(self, n, i, x):
        n[0,i] = x

    def nr_of_neigbours_classes(self, n, nr_of_classes):
        neighbours_counts = [0 for c_id in xrange(0, nr_of_classes)]
        for i in xrange(0, n.shape[1]):
            if n[0,i] <> -1:
                for c_id in xrange(0, nr_of_classes):
                    if n[0,i] == c_id:
                        neighbours_counts[c_id] = neighbours_counts[c_id] + 1
                        break
        return neighbours_counts

    def fill_empty_class(self, row, c_id, nr_of_classes):
        if len(row) == 0:
            for i in xrange(0, nr_of_classes):
                row.append(0.9 if i == c_id else 0.1/(nr_of_classes-1))




