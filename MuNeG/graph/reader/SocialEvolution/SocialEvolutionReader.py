import networkx as nx
PATH_TO_NODES = '..\\..\\..\\dataset\\SocialEvolution\\SocialEvolution\\Politics.csv'
PATH_TO_EDGES = '..\\..\\..\\dataset\\SocialEvolution\\SocialEvolution\\RelationshipsFromSurveys.csv\\RelationshipsFromSurveys.csv'
PATH_TO_CALLS = '..\\..\\..\\dataset\\SocialEvolution\\SocialEvolution\\Calls.csv\\Calls.csv'
PATH_TO_PROXIMITY = '..\\..\\..\\dataset\\SocialEvolution\\SocialEvolution\\Proximity.csv\\Proximity.csv'
PATH_TO_SMS = '..\\..\\..\\dataset\\SocialEvolution\\SocialEvolution\\SMS.csv\\SMS.csv'
import os
import csv
from graph.reader.SocialEvolution.SocialEvolutionNode import SocialEvolutionNode
class SocialEvolutionReader():

    graph = nx.MultiGraph()
    nodes = dict([])
    layers = dict([])

    def read(self, class_label):
        self.graph = nx.MultiGraph()
        self.load_nodes_with_labels(class_label)
        l_id = self.load_edges_and_layers()
        l_id = self.load_calls(l_id)
        l_id = self.read_proximity_layer(l_id)
        l_id = self.read_sms(l_id)
        pass

    def read_sms(self, l_id):
        path = os.path.join(os.path.dirname(__file__), '%s' % PATH_TO_SMS)
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            i = -1
            for row in reader:
                if i == -1:
                    i = i + 1
                else:
                    id_from = row[0]
                    id_to = row[3]
                    if id_to <> '' and id_from <> '':
                        id_from = int(id_from) - 1
                        id_to = int(id_to) - 1
                        node_from = None
                        node_to = None
                        if self.nodes.has_key(id_from):
                            node_from = self.nodes.get(id_from)
                        if self.nodes.has_key(id_to):
                            node_to = self.nodes.get(id_to)
                        if node_from == None:
                            node_from = SocialEvolutionNode(id_from, 0)
                            self.nodes.update({id_from: node_from})
                            self.graph.add_node(node_from)
                        if node_to == None:
                            node_to = SocialEvolutionNode(id_to, 0)
                            self.nodes.update({id_to: node_to})
                            self.graph.add_node(node_to)
                        self.graph.add_edge(node_from, node_to, weight=l_id, layer='L' + str(l_id), conWeight=0.5)
                i = i + 1
            l_id = l_id + 1
            return l_id

    def read_proximity_layer(self, l_id):
        path = os.path.join(os.path.dirname(__file__), '%s' % PATH_TO_PROXIMITY)
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            i = -1
            for row in reader:
                if i == -1:
                    i = i + 1
                else:
                    id_from = row[0]
                    id_to = row[1]
                    if id_to <> '' and id_from <> '':
                        id_from = int(id_from) - 1
                        id_to = int(id_to) - 1
                        node_from = None
                        node_to = None
                        if self.nodes.has_key(id_from):
                            node_from = self.nodes.get(id_from)
                        if self.nodes.has_key(id_to):
                            node_to = self.nodes.get(id_to)
                        if node_from == None:
                            node_from = SocialEvolutionNode(id_from, 0)
                            self.nodes.update({id_from: node_from})
                            self.graph.add_node(node_from)
                        if node_to == None:
                            node_to = SocialEvolutionNode(id_to, 0)
                            self.nodes.update({id_to: node_to})
                            self.graph.add_node(node_to)
                        self.graph.add_edge(node_from, node_to, weight=l_id, layer='L' + str(l_id), conWeight=0.5)
                i = i + 1
                print i
            l_id = l_id + 1
            return l_id

    def load_calls(self, l_id):
        path = os.path.join(os.path.dirname(__file__), '%s' % PATH_TO_CALLS)
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            i = -1
            for row in reader:
                if i == -1:
                    i = i + 1
                else:
                    id_from = row[0]
                    id_to = row[3]
                    if id_to <> '' and id_from <> '':
                        id_from = int(id_from) - 1
                        id_to = int(id_to) - 1
                        node_from = None
                        node_to = None
                        if self.nodes.has_key(id_from):
                            node_from = self.nodes.get(id_from)
                        if self.nodes.has_key(id_to):
                            node_to = self.nodes.get(id_to)
                        if node_from == None:
                            node_from = SocialEvolutionNode(id_from, 0)
                            self.nodes.update({id_from: node_from})
                            self.graph.add_node(node_from)
                        if node_to == None:
                            node_to = SocialEvolutionNode(id_to, 0)
                            self.nodes.update({id_to: node_to})
                            self.graph.add_node(node_to)
                        self.graph.add_edge(node_from, node_to, weight=l_id, layer='L' + str(l_id), conWeight=0.5)
                    i = i + 1
            l_id = l_id + 1
            return l_id

    def load_edges_and_layers(self):
        path = os.path.join(os.path.dirname(__file__), '%s' % PATH_TO_EDGES)
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            i = -1
            l_id = 1
            for row in reader:
                if i == -1:
                    i = i + 1
                else:
                    id_from = int(row[0]) - 1
                    id_to = int(row[1]) - 1
                    node_from = None
                    node_to = None
                    if self.nodes.has_key(id_from):
                        node_from = self.nodes.get(id_from)
                    if self.nodes.has_key(id_to):
                        node_to = self.nodes.get(id_to)
                    if node_from == None:
                        node_from = SocialEvolutionNode(id_from, 0)
                        self.nodes.update({id_from: node_from})
                        self.graph.add_node(node_from)
                    if node_to == None:
                        node_to = SocialEvolutionNode(id_to, 0)
                        self.nodes.update({id_to: node_to})
                        self.graph.add_node(node_to)
                    layer_name = row[2]
                    if self.layers.has_key(layer_name):
                        layer_id = self.layers.get(layer_name)
                    else:
                        self.layers.update({layer_name: l_id})
                        layer_id = l_id
                        l_id = l_id + 1
                    self.graph.add_edge(node_from, node_to, weight=layer_id, layer='L' + str(layer_id), conWeight=0.5)
                i = i + 1
            return l_id

    def load_nodes_with_labels(self, class_label):
        path = os.path.join(os.path.dirname(__file__), '%s' % PATH_TO_NODES)
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            i = -1
            for row in reader:
                if i == -1:
                    i = i + 1
                else:
                    node_id = int(row[0]) - 1
                    if self.nodes.has_key(node_id):
                        node = self.nodes.get(node_id)
                        prefered_party = row[2]
                        label = 1 if prefered_party == class_label else 0
                        node.label = label if label == 1 else node.label
                    else:
                        prefered_party = row[2]
                        label = 1 if prefered_party == class_label else 0
                        node = SocialEvolutionNode(node_id, label)
                        self.graph.add_node(node)
                        self.nodes.update({node_id: node})


if __name__ == "__main__":
    reader = SocialEvolutionReader()
    reader.read()