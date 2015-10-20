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
                    self.graph.add_edge(node, neighbor, weight=i, layer='L' + str(i), conWeight=0.5)
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

    def read(self):
        self.prepare_graph()
        self.decorate_nodes()
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
                node.functions(go_term)


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
