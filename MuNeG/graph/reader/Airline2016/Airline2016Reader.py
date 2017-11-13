import csv
import os

import networkx as nx
from graph.reader.Airline2016.Airline2016Node import Airline2016Node

PATH_TO_NODES = '/home/apopiel/multiplex/MuNeG/dataset/Airline2016/airline_nodes_class_%s.csv'
PATH_TO_EDGES = '/home/apopiel/multiplex/MuNeG/dataset/Airline2016/airline_network_2016.csv'
class Airline2016Reader():

    graph = nx.MultiGraph()
    nodes = dict([])
    layers = dict([])

    layers_edges = {'L78': 51, 'L79': 1370, 'L72': 2347, 'L73': 74, 'L70': 600, 'L124': 76, 'L76': 96, 'L77': 417, 'L74': 670, 'L75': 1, 'L118': 94, 'L119': 4, 'L116': 4, 'L117': 54, 'L114': 4, 'L115': 2, 'L112': 14, 'L113': 163, 'L110': 99, 'L111': 5, 'L20': 4, 'L122': 4, 'L21': 6, 'L58': 395, 'L23': 787, 'L22': 54, 'L25': 117, 'L24': 238, 'L27': 309, 'L26': 167, 'L29': 6, 'L28': 67, 'L71': 4, 'L43': 154, 'L42': 48, 'L41': 1391, 'L40': 110, 'L47': 60, 'L46': 780, 'L45': 105, 'L44': 5451, 'L49': 28, 'L48': 136, 'L109': 12, 'L108': 2, 'L101': 432, 'L100': 1485, 'L19': 17, 'L102': 165, 'L105': 505, 'L104': 1006, 'L107': 10, 'L106': 14, 'L36': 81, 'L37': 2719, 'L34': 36, 'L35': 512, 'L32': 62, 'L33': 906, 'L30': 430, 'L31': 1255, 'L38': 164, 'L39': 210, 'L50': 66, 'L51': 5, 'L52': 131, 'L53': 461, 'L54': 1886, 'L55': 523, 'L56': 71, 'L57': 27, 'L14': 189, 'L15': 213, 'L16': 5, 'L17': 8, 'L10': 40, 'L11': 326, 'L12': 474, 'L13': 70, 'L130': 2, 'L131': 2, 'L132': 2, 'L61': 18, 'L87': 14, 'L86': 627, 'L85': 739, 'L84': 501, 'L83': 178, 'L82': 105, 'L81': 418, 'L80': 72, 'L89': 10, 'L88': 534, 'L123': 5, 'L65': 967, 'L64': 6, 'L67': 48, 'L66': 25, 'L18': 17, 'L60': 90, 'L63': 316, 'L62': 2749, 'L103': 37, 'L69': 34, 'L68': 442, 'L6': 44, 'L7': 387, 'L4': 713, 'L120': 33, 'L2': 19, 'L3': 1465, 'L125': 28, 'L1': 322, 'L129': 4, 'L128': 12, 'L8': 186, 'L9': 116, 'L94': 437, 'L95': 1608, 'L96': 1124, 'L97': 940, 'L90': 9, 'L91': 407, 'L92': 10, 'L93': 306, 'L98': 887, 'L99': 724, 'L59': 40, 'L5': 12, 'L121': 14, 'L127': 1, 'L126': 2}
    name_quantity = {'22Q': 2, 'WE': 6, '5Y': 512, '5X': 906, 'FX': 1886, 'WN': 5451, '5V': 316, 'ZX': 4, 'S5': 940, 'VJT': 5, 'K3': 51, '1RQ': 48, 'F9': 407, '2TQ': 25, 'K5': 48, 'WP': 14, 'O6': 1, 'K8': 600, 'HBQ': 14, '29Q': 6, 'G7': 534, '1BQ': 40, '9K': 90, 'G4': 1370, 'ABX': 164, 'NC': 213, '9E': 724, 'RVQ': 2, 'X9': 105, 'PRQ': 8, 'SEB': 9, '8C': 81, '9S': 54, 'C5': 238, 'GV': 501, 'WRD': 4, 'CH': 6, 'ELL': 4, '0WQ': 713, 'N8': 34, 'AMQ': 787, '1SQ': 12, 'ADB': 5, 'B6': 505, '28Q': 10, 'XP': 437, '2O': 116, 'VI': 12, '1WQ': 12, 'OO': 1608, 'SY': 1006, 'HA': 76, '8V': 178, 'KD': 154, '09Q': 1465, 'KAH': 117, 'KO': 189, 'SNK': 17, 'KS': 105, 'KAQ': 167, 'X4': 10, 'J5': 70, 'PM': 14, 'EM': 210, 'DL': 2347, 'KH': 18, 'Q5': 17, '0MQ': 28, '1TQ': 19, 'U7': 322, '1EQ': 54, 'YX': 1124, '4W': 72, 'M6': 28, 'WST': 4, '3M': 62, 'NEW': 2, '4EQ': 44, 'YR': 67, '3F': 40, 'YV': 780, 'LF': 442, '1PQ': 4, 'PT': 165, '4Y': 326, 'I4': 14, '7H': 395, 'EE': 739, 'GL': 1391, '7S': 474, 'ZW': 430, 'AJQ': 461, 'V8': 186, 'PH': 4, 'PO': 66, '04Q': 887, 'Z3': 2, 'AA': 2719, '20Q': 5, 'AC': 94, 'ZK': 74, '2HQ': 37, 'GCH': 1, 'KAT': 10, '1YQ': 36, 'AS': 387, 'NK': 432, '1QQ': 2, 'KLQ': 309, 'VX': 99, 'AX': 417, 'CP': 306, 'EV': 1485, '4B': 60, 'AAT': 4, '1AQ': 670, '2E': 27, 'QX': 163, 'GFQ': 110, 'H6': 1255, '2F': 71, '8D': 131, '8E': 418, 'PFQ': 5, '27Q': 523, 'MW': 96, 'MQ': 627, 'L2': 136, 'QK': 2, '23Q': 33, 'UA': 2749, 'OH': 967}

    def read(self, threshold, classLabel):
        self.graph = nx.MultiGraph()
        id = self.read_nodes(classLabel)
        self.read_edges(id, threshold)
        pass

    def read_edges(self, id, threshold):
        node_id = id
        path = os.path.join(os.path.dirname(__file__), '%s' % PATH_TO_EDGES)
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            i = -1
            layers_iter = 1
            for layer, name_from, name_to in reader:
                if i == -1:
                    i = i + 1
                else:
                    if self.name_quantity[layer] > threshold:
                        if self.layers.has_key(layer):
                            layer_id = self.layers.get(layer)
                        else:
                            self.layers.update({layer: layers_iter})
                            layer_id = layers_iter
                            layers_iter = layers_iter + 1
                        node_from = self.nodes.get(name_from)
                        node_to = self.nodes.get(name_to)
                        if node_from == None:
                            node_from = Airline2016Node(node_id, 0, name_from)
                            self.graph.add_node(node_from)
                            self.nodes.update({name_from:node_from})
                            node_id = node_id + 1
                        elif node_to == None:
                            node_to = Airline2016Node(node_id, 0, name_to)
                            self.graph.add_node(node_to)
                            self.nodes.update({name_to:node_to})
                            node_id = node_id + 1
                        layer_name = 'L' + str(layer_id)
                        self.graph.add_edge(node_from, node_to, weight=layer_id, layer=layer_name, conWeight=0.5)
                        i = i + 1
        pass

    def read_nodes(self, classLabel):
        path = os.path.join(os.path.dirname(__file__), '%s' % PATH_TO_NODES % classLabel)
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            i = -1
            for name, readedClass in reader:
                if i == -1:
                    i = i + 1
                else:
                    node = Airline2016Node(i, int(readedClass), name)
                    self.graph.add_node(node)
                    self.nodes.update({name: node})
                    i = i + 1
        return i

    def calcuclate_homogenity(self):
        results = []
        for node in self.graph.nodes():
            neighbors = nx.neighbors(self.graph, node)
            summ = 0
            for n in neighbors:
                if node.label == n.label:
                    summ = summ + 1
            if len(neighbors) <> 0:
                results.append(float(summ)/float(len(neighbors)))
            else:
                results.append(0.0)
        homogenity = float(sum(results))/float(len(self.graph.nodes()))
        return homogenity

if __name__ == "__main__":
    reader = Airline2016Reader()
    reader.read()


