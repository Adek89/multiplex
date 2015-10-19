__author__ = 'Adrian'
import os
import tokenize as token
import networkx as nx

from graph.reader.DanioRerio.DanioRerioNode import DanioRerioNode


PATH_TO_NETWORK = '..\\..\\..\\dataset\\DanioRerio\\edge_list_danioRerio_layer%s.txt'
class DanioRerioReader():

    graph = nx.MultiGraph()
    nodes = dict([])


    def read(self):
        nodes = dict([])
        for i in xrange(1,6):
            tokens = self.prepare_file(PATH_TO_NETWORK % str(i))
            id = int(tokens.next()[1])

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
