__author__ = 'Adrian'
import tokenize as token
import os
import string
import networkx as nx
from graph.gen.Group import Group
from graph.gen.Node import Node

def move_to_nodes(tokens):
    tokens.next()
    tokens.next()
    tokens.next()


def read_file(file_name, path):
    path = os.path.join(os.path.dirname(''), path, '%s' % file_name)
    f = open(path)
    tokens = token.generate_tokens(f.readline)
    return tokens


def load_node(groups, id, node_data, node_id_from_file):
    splits = string.split(node_data)
    node_id = int(splits[0][1:])
    label = int(splits[1])
    group_color = splits[2]
    group_id = int(splits[3][0:-1])
    if not groups.has_key(group_id):
        groups[group_id] = Group(group_color, group_id)
    node = Node(groups[group_id], label, node_id)
    node_id_from_file[id] = node
    return node


def read_nodes(graph, groups, node_id_from_file, tokens):
    while tokens.next()[1] == 'node':
        tokens.next()
        tokens.next()
        tokens.next()
        id = int(tokens.next()[1])
        tokens.next()
        tokens.next()
        node_data = tokens.next()[1]
        node = load_node(groups, id, node_data, node_id_from_file)
        graph.add_node(node)
        tokens.next()
        tokens.next()
        tokens.next()


def read_edges(graph, node_id_from_file, tokens):
    while True:
        tokens.next()
        tokens.next()
        tokens.next()
        source_node = int(tokens.next()[1])
        tokens.next()
        tokens.next()
        target_node = int(tokens.next()[1])
        tokens.next()
        tokens.next()
        layer_name = tokens.next()[1][1:-1]
        tokens.next()
        tokens.next()
        con_weight = int(tokens.next()[1])
        tokens.next()
        tokens.next()
        weight = int(tokens.next()[1])
        graph.add_edge(node_id_from_file[source_node], node_id_from_file[target_node], weight=weight, layer=layer_name,
                       conWeight=con_weight)
        tokens.next()
        tokens.next()
        tokens.next()
        if tokens.next()[1] != 'edge':
            break


def read_from_gml(path, file_name):
        tokens = read_file(file_name, path)
        move_to_nodes(tokens)

        graph = nx.MultiGraph()
        node_id_from_file = {}
        groups = {}
        read_nodes(graph, groups, node_id_from_file, tokens)
        read_edges(graph, node_id_from_file, tokens)
        return graph





