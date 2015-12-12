__author__ = 'Adrian'
import networkx as nx
import numpy as np
import random
import scipy.sparse as sp


class RwpIterative():

    d = dict()
    separated_networks = dict()

    def __init__(self, separated_networks):
        self.separated_networks = separated_networks


    def init_d_matrix(self, network):
        layer_number = self.separated_networks.__len__()
        for node in network.nodes():
            d_matrix_for_node = np.ndarray(shape=(layer_number, layer_number), dtype=np.double)
            for i, separated_network in self.separated_networks.items():
                for j in xrange(0, layer_number):
                    d_matrix_for_node[j][int(i)-1] = separated_network.degree(node.id)
            csr_d_matrix_for_node = sp.csr_matrix(d_matrix_for_node)
            self.d.update({node.id : csr_d_matrix_for_node})


    def random_walk(self, network, class_mat, start_layer, number_repetitions, depth):
        self.init_d_matrix(network)
        results=dict()

        unknown_filter = [i if class_mat[i][0] == 0.5 and class_mat[i][1] == 0.5 else -1  for i in xrange(0, class_mat.__len__())]
        filtered_indexes = filter(lambda x : x != -1, unknown_filter)
        unknown_nodes = [node for node in filter(lambda n: n.id in filtered_indexes, network.nodes())]

        for node in unknown_nodes:
            res_node=[]
            for i in xrange(number_repetitions):
                res_node.append(self.random_walk_recursive(network, unknown_nodes, node.id, None, start_layer, depth, 1))
            #print 'Finished for node: '+str(node)+'       Classses: '
            #print res_node
            results[node]=res_node
        return results

    def random_walk_recursive(self, network, unknown_nodes, current_node, visited, start_layer, depth, counter):
        decided_layer = self.decide_layer_change(current_node, start_layer)
        if (decided_layer != start_layer):
            return self.random_walk_recursive(network, current_node, visited, decided_layer, depth-1, counter+1)
        else:
            edge=self.draw_connection_at_node(self.separated_networks[str(start_layer)],current_node)

            #print 'Edge: '+str(edge)

            if (depth<=0 or edge is None): # or current_node==visited
                #print 'Depth: '+str(depth)+'    Visited already: '+str(current_node==visited)
                return (None, counter)
            else:
                nodes_list = list(network.nodes())
                new_node = filter(lambda n: n.id == edge[1], nodes_list)[0]
                if (edge is not None) and (new_node not in unknown_nodes):
                    #print 'Reached node with class: '+ str(network.node[edge[1]]['cls'])
                    #print 'Class reached: '+str(network.node[edge[1]]['cls'])
                    return (new_node.label,counter)
                else:
                    #remove comment if renadom walk does not allow retracking
                    #visited=current_node

                    #print 'Recursive sampling. Depth: '+str(depth)
                    #print 'Going to: '+str(edge[1])
                    #print 'recursion'
                    return self.random_walk_recursive(network, edge[1], visited, start_layer, depth-1, counter+1)

    def draw_connection_at_node(self, network, node):
        result=None
        neighbs = [edge for edge in nx.edges_iter(network, node)]
        if len(neighbs)>0:
            result = random.choice(neighbs)
        return result

    def decide_layer_change(self, current_node, start_layer):
        layer_matrix = self.d[current_node]
        row = layer_matrix[start_layer - 1, :]
        index = row.data.argmax()
        return index + 1
