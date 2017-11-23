__author__ = 'Adrian'
import os
import tokenize as token

import networkx as nx
from graph.reader.DanioRerio.DanioRerioNode import DanioRerioNode

PATH_TO_NETWORK = '..\\..\\..\\dataset\\DanioRerio\\edge_list_danioRerio_layer%s.txt'
PATH_TO_NODES = '..\\..\\..\\dataset\\DanioRerio\\danioRerio_layout.txt'
PATH_TO_FUNCTIONS = '..\\..\\..\\dataset\\DanioRerio\\danioRerio_functions.txt'
class DanioRerioReader():

    graph = nx.MultiGraph()
    nodes = dict([])


    def prepare_graph(self):
        for i in xrange(1, 6):
            tokens = self.prepare_file(PATH_TO_NETWORK % str(i))
            while True:
                try:
                    id = int(tokens.next()[1]) - 1
                    node = self.load_or_prepare_node(id)
                    neighbor_id = int(tokens.next()[1]) - 1
                    neighbor = self.load_or_prepare_node(neighbor_id)
                    self.graph.add_edge(node, neighbor, weight=6-i, layer='L' + str(6-i), conWeight=0.5)
                    tokens.next()
                    tokens.next()
                except:
                    break

    def analyse_special_case(self, id, name, tokens):
        if id in (12, 88):
            name = name + tokens.next()[1]
        elif id in (13, 14):
            name = name + tokens.next()[1] + tokens.next()[1]
        return name

    def decorate_nodes(self):
        tokens = self.prepare_file(PATH_TO_NODES)
        tokens.next()
        tokens.next()
        tokens.next()
        for i in xrange(0, 155):
            id = int(tokens.next()[1]) - 1
            name = tokens.next()[1]
            node = self.load_or_prepare_node(id)
            name = self.analyse_special_case(id, name, tokens)
            node.name = name
            tokens.next()

    def assign_functions(self):
        tokens = self.prepare_file(PATH_TO_FUNCTIONS)
        for i in xrange(0, 155):
            id = int(tokens.next()[1])
            node = self.load_or_prepare_node(id - 1)
            next_token = tokens.next()[1]
            while next_token <> '\n':
                go_term = next_token
                next_token = tokens.next()[1]
                while next_token <> 'GO' and next_token <> '\n':
                    go_term = go_term + next_token
                    next_token = tokens.next()[1]
                node.functions.add(go_term)

    def prepare_file(self, path):
        path = os.path.join(os.path.dirname(__file__), '%s' % path)
        f = open(path)
        tokens = token.generate_tokens(f.readline)
        return tokens

    def load_or_prepare_node(self, id):
        if (self.nodes.has_key(id)):
            node = self.nodes.get(id)
        else:
            node = DanioRerioNode(id)
            self.nodes.update({id: node})
            self.graph.add_node(node)
        return node

    def assign_labels(self, label_fun):
        if not label_fun == '':
            for node in self.graph.nodes():
                if label_fun in node.functions:
                    node.label = 1

    def read(self, label_fun=''):
        self.prepare_graph()
        self.decorate_nodes()
        self.assign_functions()
        self.assign_labels(label_fun)


    def create_go_terms_map(self):
        map = dict([])
        for (id, node) in self.nodes.items():
            functions = node.functions
            for fun in functions:
                if not map.has_key(fun):
                    map.update({fun : 1})
                else:
                    count = map.get(fun)
                    map.update({fun : count + 1})
        return map

    def calcuclate_homogenity(self):
        results = []
        for node in self.graph.nodes():
            neighbors = nx.neighbors(self.graph, node)
            summ = 0
            for n in neighbors:
                if node.label == n.label:
                    summ = summ + 1
            results.append(float(summ)/float(len(neighbors)))
        homogenity = float(sum(results))/float(len(self.graph.nodes()))
        return homogenity



