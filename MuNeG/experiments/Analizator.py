
__author__ = 'Adek'
import time
import sys
sys.path.append('/home/apopiel/MuNeG')
from graph.analyser.GraphAnalyser import GraphAnalyser
from graph.reader.DanioRerio.DanioRerioReader import DanioRerioReader
if __name__ == '__main__':
    reader = DanioRerioReader()
    reader.read()

    graph = reader.graph
    ga = GraphAnalyser(graph)
    ga.analyse()
