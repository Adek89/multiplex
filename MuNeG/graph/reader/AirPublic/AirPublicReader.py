__author__ = 'Adrian'

NR_OF_AIRPORTS = 450
NR_OF_LAYERS = 38
PATH_TO_NETWORK = '..\\..\\..\\dataset\\air-multi-public-dataset\\network.txt'
PATH_TO_AIRPORTS = '..\\..\\..\\dataset\\air-multi-public-dataset\\airports.txt'
import os
import tokenize as token
import networkx as nx
from graph.reader.AirPublic.AirPublicNode import AirPublicNode
class AirPublicReader:

    graph = nx.MultiGraph()
    nodes = dict([])

    def read(self):
        tokens = self.prepare_file(PATH_TO_NETWORK)
        self.prepare_layers(tokens)
        tokens = self.prepare_file(PATH_TO_AIRPORTS)
        self.decorate_nodes(tokens)


    def prepare_file(self, path):
        path = os.path.join(os.path.dirname(__file__), '%s' % path)
        f = open(path)
        tokens = token.generate_tokens(f.readline)
        return tokens

    def load_or_prepare_node(self, airport_id):
        if (self.nodes.has_key(airport_id)):
            node = self.nodes.get(airport_id)
        else:
            node = AirPublicNode(airport_id)
            self.nodes.update({airport_id: node})
            self.graph.add_node(node)
        return node

    def prepare_connections_for_airport(self, layer, node, nr_of_connections, tokens):
        for k in xrange(0, nr_of_connections):
            neighbor_id = int(tokens.next()[1])
            neighbor = self.load_or_prepare_node(neighbor_id)
            self.graph.add_edge(node, neighbor, weight=layer, layer='L' + str(layer), conWeight=0.5)
        tokens.next()

    def prepare_airports(self, layer, nr_of_airports_in_layer, tokens):
        for i in xrange(0, nr_of_airports_in_layer):
            airport_id = int(tokens.next()[1])
            node = self.load_or_prepare_node(airport_id)
            nr_of_connections = int(tokens.next()[1])
            self.prepare_connections_for_airport(layer, node, nr_of_connections, tokens)

    def prepare_layers(self, tokens):
        for layer in xrange(1, NR_OF_LAYERS):
            nr_of_airports_in_layer = int(tokens.next()[1])
            tokens.next()
            tokens.next()
            self.prepare_airports(layer, nr_of_airports_in_layer, tokens)
            tokens.next()

    def calculate_longitude(self, node, tokens, value):
        if (value <> '-'):
            node.longitude = float(value)
        else:
            node.longitude = float(value + tokens.next()[1])

    def decorate_nodes(self, tokens):
        for i in xrange(0, NR_OF_AIRPORTS):
            airport_id = int(tokens.next()[1])
            node = self.load_or_prepare_node(airport_id)
            node.name = tokens.next()[1]
            value = tokens.next()[1]
            self.calculate_longitude(node, tokens, value)
            node.latitude = float(tokens.next()[1])
            tokens.next()