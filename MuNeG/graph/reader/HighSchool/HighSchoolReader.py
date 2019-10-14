import sys
sys.path.append('D:\pycharm_workspace\multiplex\MuNeG')
PATH_TO_NETWORK = '../../../dataset/HighSchool/High-School_data_2013.csv'
import os
import csv
import networkx as nx
from graph.reader.HighSchool.HighSchoolNode import HighSchoolNode
class HighSchoolReader:

    graph = nx.Graph()
    nodes = dict([])
    classes_dict ={'2BIO1':0, '2BIO2':1, '2BIO3':2, 'MP':3, 'MP*1':4, 'MP*2':5, 'PC':6, 'PC*':7, 'PSI*':8}
    identity = 0

    def prepare_file(self, path):
        path = os.path.join(os.path.dirname(__file__), '%s' % path)
        with open(path) as f:
            line = csv.reader(f, delimiter='\t')
            for lineContent in line:
                    yield lineContent[0].split()

    def read(self):
        listOfLineElementsGenerator = self.prepare_file(PATH_TO_NETWORK)
        while(True):
            try:
                listOfLineElements = listOfLineElementsGenerator.next()
            except StopIteration:
                break
            intervalString = listOfLineElements[0]
            if (intervalString == ""):
                break
            interval = int(intervalString)
            leftId = int(listOfLineElements[1])
            rightId = int(listOfLineElements[2])
            leftStatus = listOfLineElements[3]
            rightStatus = listOfLineElements[4]
            leftNode = self.createOrGetNode(leftId, leftStatus)
            rightNode = self.createOrGetNode(rightId, rightStatus)
            self.graph.add_edge(leftNode, rightNode)


    def createOrGetNode(self, id, label):
        if self.nodes.has_key(id):
            return self.nodes.get(id)
        else:
            node = HighSchoolNode(self.identity, self.classes_dict.get(label))
            self.nodes.update({id:node})
            self.graph.add_node(node)
            self.identity = self.identity + 1
            return node
