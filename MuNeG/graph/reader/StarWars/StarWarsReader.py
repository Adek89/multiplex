import os
import tokenize as token

import networkx as nx

from graph.reader.StarWars.StarWarsNode import StarWarsNode

PATH_TO_NODES = '..\\..\\..\\dataset\\StarWars\\StarWars_layout.txt'
PATH_TO_NETWORK = '..\\..\\..\\dataset\\StarWars\\StarWars_episode%s.edges'
class StarWarsReader():

    graph = nx.MultiGraph()
    nodes = dict([])

    def prepare_file(self, path):
        path = os.path.join(os.path.dirname(__file__), '%s' % path)
        f = open(path)
        tokens = token.generate_tokens(f.readline)
        return tokens

    def load_or_prepare_node(self, id):
        if (self.nodes.has_key(id)):
            node = self.nodes.get(id)
        else:
            raise ValueError('Node should be available in set')
        return node

    def read(self):
        self.graph = nx.MultiGraph()
        self.read_nodes()
        self.read_edges()

    def read_edges(self):
        for i in xrange(1, 7):
            tokens = self.prepare_file(PATH_TO_NETWORK % str(i))
            edges_iter = 0
            while True:
                try:
                    id_from = int(tokens.next()[1]) - 1
                    id_to = int(tokens.next()[1]) - 1
                    number_of_scenes = int(tokens.next()[1])
                    node_from = self.load_or_prepare_node(id_from)
                    node_to = self.load_or_prepare_node(id_to)
                    self.graph.add_edge(node_from, node_to, weight=i, layer='L' + str(i), conWeight=0.5,
                                        scenes=number_of_scenes)
                    tokens.next()
                    edges_iter = edges_iter + 1
                except:
                    break
            self.assert_number_of_edges(edges_iter, i)

    def assert_number_of_edges(self, edges_iter, i):
        if i == 1:
            assert edges_iter == 148
        elif i == 2:
            assert edges_iter == 103
        elif i == 3:
            assert edges_iter == 67
        elif i == 4:
            assert edges_iter == 61
        elif i == 5:
            assert edges_iter == 59
        elif i == 6:
            assert edges_iter == 60

    def read_nodes(self):
        tokens = self.prepare_file(PATH_TO_NODES)
        tokens.next()
        tokens.next()
        tokens.next()
        tokens.next()
        while True:
            try:
                id = int(tokens.next()[1]) - 1
                name = tokens.next()[1]
                label = int(tokens.next()[1])
                node = StarWarsNode(id, label, name)
                self.graph.add_node(node)
                self.nodes.update({id: node})
                tokens.next()
            except:
                break


if __name__ == "__main__":
    reader = StarWarsReader()
    reader.read()

