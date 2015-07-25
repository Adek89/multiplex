'''
Created on 18 mar 2014

@author: Adek
'''
import time
import sys
import threading
sys.path.append('/home/apopiel/MuNeG')
from experiments.EnsableMethodsReal import EnsambleMethods
from bin.graph.reader.Salon24 import Salon24Reader
from bin.graph.method.ensamble import EnsambleLearning
from bin.graph.analyser import GraphAnalyser


def readRealData(limit):
    reader = Salon24Reader(limit)
    realGraph = reader.createNetwork()
    ensamble = EnsambleLearning(realGraph, 1, 2)
    realGraph = ensamble.sampleGraph()

    nodes = realGraph.nodes()
    print 'count of ones: ' + str(filter(lambda node: node.label == 1, nodes).__len__())

    ga = GraphAnalyser(realGraph, 0.0, 1)
    ga.analyse()
    return realGraph

if __name__ == '__main__':
    start_time = time.time()

    limit = int(sys.argv[1])
    sampleNodes = int(sys.argv[2])

    realGraph = readRealData(limit)

    em1 = EnsambleMethods(realGraph, sampleNodes, 0.2)
    em2 = EnsambleMethods(realGraph, sampleNodes, 0.4)
    em3 = EnsambleMethods(realGraph, sampleNodes, 0.6)
    em4 = EnsambleMethods(realGraph, sampleNodes, 0.8)
    t1 = threading.Thread(target=em1.processExperiments)
    # t2 = threading.Thread(target=em2.processExperiments)
    # t3 = threading.Thread(target=em3.processExperiments)
    # t4 = threading.Thread(target=em4.processExperiments)
    t1.start()
    # t2.start()
    # t3.start()
    # t4.start()
    print("--- %s seconds -we--" % str(time.time() - start_time))

