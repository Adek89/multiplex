import os
import sys
import tokenize as token

import networkx as nx

sys.path.append('D:\pycharm_workspace\multiplex\MuNeG')
from graph.reader.Cora.CoraNode import CoraNode


PATH_TO_NETWORK = '..\\..\\..\\dataset\\cora\\cora.cites'
PATH_TO_NODES = '..\\..\\..\\dataset\\cora\\cora.content'
class CoraReader:

    graph = nx.Graph()
    nodes = dict([])
    classes_dict ={'Case_Based':0, 'Genetic_Algorithms':1, 'Neural_Networks':2, 'Probabilistic_Methods':3, 'Reinforcement_Learning':4, 'Rule_Learning':5, 'Theory':6}
    identity = 0

    def prepare_file(self, path):
        path = os.path.join(os.path.dirname(__file__), '%s' % path)
        f = open(path)
        tokens = token.generate_tokens(f.readline)
        return tokens

    def createOrGetNode(self, id, label=""):
        if self.nodes.has_key(id):
            return self.nodes.get(id)
        else:
            node = CoraNode(id, label)
            self.nodes.update({id:node})
            self.graph.add_node(node)
            return node

    def read(self):
        self.read_nodes()
        self.read_edges()

    def read_edges(self):
        i = 0
        tokens = self.prepare_file(PATH_TO_NETWORK)
        while (True):
            string_source_id = tokens.next()[1]
            if string_source_id <> '':
                source_id = int(string_source_id)
            else:
                break
            destination_id = int(tokens.next()[1])
            tokens.next()  # line feed
            source_node = self.createOrGetNode(source_id)
            destination_node = self.createOrGetNode(destination_id)
            self.graph.add_edge(source_node, destination_node)

    def read_nodes(self):
        tokens = self.prepare_file(PATH_TO_NODES)
        while (True):
            string_id = tokens.next()[1]
            if (string_id <> ''):
                id = int(string_id)
            else:
                break
            while (True):
                try:
                    class_name = tokens.next()[1]
                    int(class_name)  # stop when class_name is string
                except:
                    break
            tokens.next()  # line ending
            node = CoraNode(self.identity, self.classes_dict[class_name])
            self.nodes.update({id:node})
            self.graph.add_node(node)
            self.identity = self.identity + 1