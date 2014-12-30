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
        res_in = classMat
        for i in range(0, repetitions):
            '''
            propagacja poprzez mnozenie macierzy
            '''   
            res = np.dot(adjMat, res_in) 
            '''
            nadpisanie zmienionych znanych wartosci klas
            '''
            res[trainingInstances,:]=classMat[trainingInstances,:]
                    
            '''
            Normalizacja
            '''
            adjMatPy = np.array(adjMat)
            self.normalize(adjMatPy, testingInstances, np.finfo(np.double).tiny, res)
            
            '''
            Warunek stopu
            '''
            if (self.stopConditionReached(res-res_in, epsilon)):
                break
            
            res_in = res
        return res
            
    def stopConditionReached(self, delta, epsilon):
        absVal = np.abs(delta)
        maxRow = np.max(absVal, 0)
        maxValue = np.max(maxRow)
        return maxValue<epsilon
               
                    
    def normalize(self, adjMatPy, instances, sumElement, res):
            rows = adjMatPy[instances,:]      
            repmatInput = np.sum(rows, axis=1) + sumElement 
            reshaped = np.reshape(repmatInput, (repmatInput.size, 1))
            repmat = np.tile(reshaped, (1, res[0].size))
            #sprawdzic czy czy nie trzeba ./
            res[instances,:] = res[instances,:]/repmat