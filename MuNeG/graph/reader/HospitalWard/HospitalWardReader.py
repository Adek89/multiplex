import sys
sys.path.append('D:\pycharm_workspace\multiplex\MuNeG')
PATH_TO_NETWORK = '..\\..\\..\\dataset\\HospitalWard\\detailed_list_of_contacts_Hospital.dat'
import os
import tokenize as token
import networkx as nx
from graph.reader.HospitalWard.HospitalWardNode import HospitalWardNode
class HospitalWardReader:

    graph = nx.Graph()
    nodes = dict([])
    classes_dict ={'ADM':0, 'MED':1, 'NUR':2, 'PAT':3}
    identity = 0

    def prepare_file(self, path):
        path = os.path.join(os.path.dirname(__file__), '%s' % path)
        f = open(path)
        tokens = token.generate_tokens(f.readline)
        return tokens

    def read(self):
        tokens = self.prepare_file(PATH_TO_NETWORK)
        while(True):
            intervalString = tokens.next()[1]
            if (intervalString == ""):
                break
            interval = int(intervalString)
            leftId = int(tokens.next()[1])
            rightId = int(tokens.next()[1])
            leftStatus = tokens.next()[1]
            rightStatus = tokens.next()[1]
            tokens.next() #line ending
            leftNode = self.createOrGetNode(leftId, leftStatus)
            rightNode = self.createOrGetNode(rightId, rightStatus)
            self.graph.add_edge(leftNode, rightNode)


    def createOrGetNode(self, id, status):
        if self.nodes.has_key(id):
            return self.nodes.get(id)
        else:
            node = HospitalWardNode(self.identity, self.classes_dict.get(status))
            self.nodes.update({id:node})
            self.graph.add_node(node)
            self.identity = self.identity + 1
            return node
