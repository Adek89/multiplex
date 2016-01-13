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

    def random_walk(self, network, class_mat, layers, number_repetitions, depth):
        self.init_d_matrix(network)
        results=dict()

        unknown_filter = [i if class_mat[i][0] == 0.5 and class_mat[i][1] == 0.5 else -1  for i in xrange(0, class_mat.__len__())]
        filtered_indexes = filter(lambda x : x != -1, unknown_filter)
        unknown_nodes = [node for node in filter(lambda n: n.id in filtered_indexes, network.nodes())]

        founded_nodes = {}
        temp_founded_nodes = {}
        previous_results = {}
        is_stop_reached = False
        for i in xrange(number_repetitions):
            for node in unknown_nodes:
                res_node=[]
                temp_classes_for_node = []
                for current_layer in layers:
                    new_class_with_counter = self.random_walk_recursive(network, unknown_nodes, founded_nodes, node, node.id, {}, current_layer, depth, 1)
                    res_node , temp_classes_for_node = self.collect_temp_information(new_class_with_counter, res_node, temp_classes_for_node)
                self.update_temp_founded_nodes(node, temp_classes_for_node, temp_founded_nodes)
                results = self.collect_results(node, res_node, results)
            founded_nodes = temp_founded_nodes
            current_results = self.prepare_classification_results(results)
            if len(previous_results) != 0:
                is_stop_reached = self.check_stop_sign(current_results, previous_results, 0.1)

            if is_stop_reached:
                break
            previous_results = current_results
        return results

    def random_walk_recursive(self, network, unknown_nodes, founded_nodes, start_node, current_node, visited, current_layer, depth, counter):
        visited = self.init_visited_if_needed(current_layer, visited)
        decided_layer = self.decide_layer_change(current_node, current_layer, visited)
        if (decided_layer != current_layer):
            return self.random_walk_recursive(network, unknown_nodes, founded_nodes, start_node, current_node, visited, decided_layer, depth-1, counter+1)
        else:
            edge=self.draw_connection_at_node(self.separated_networks[str(current_layer)],current_node, visited[current_layer])

            #print 'Edge: '+str(edge)

            if (depth<=0 or edge is None): # or current_node==visited
                # print 'Depth: '+str(depth)+'    Visited already: '+str(current_node==visited)
                return (None, counter)
            else:
                nodes_list = list(network.nodes())
                new_node = filter(lambda n: n.id == edge[1], nodes_list)[0]
                if (edge is not None) and (new_node not in unknown_nodes):
                    print 'Reached node with class: '+ str(new_node.label)
                    #print 'Class reached: '+str(network.node[edge[1]]['cls'])
                    return (new_node.label,counter)
                if (edge is not None) and (new_node != start_node) and (new_node in founded_nodes.keys()):
                    return (random.choice(founded_nodes[new_node]),counter)
                else:
                    #remove comment if renadom walk does not allow retracking
                    #visited=current_node

                    #print 'Recursive sampling. Depth: '+str(depth)
                    #print 'Going to: '+str(edge[1])
                    #print 'recursion'
                    visited = self.update_visited_before_transition(current_layer, edge, visited)
                    return self.random_walk_recursive(network, unknown_nodes, founded_nodes, start_node, edge[1], visited, current_layer, depth-1, counter+1)



    def check_stop_sign(self, current_results, previous_results, threshold):
        changed = 0.0
        all = float(len(current_results))
        are_none_results = False
        for node in current_results.keys():
            if (current_results[node] != previous_results[node]):
                changed += 1.0
            if (current_results[node] == None):
                are_none_results = True
                break
        ratio = changed/all
        return  ratio < threshold and not are_none_results

    def collect_temp_information(self, new_class_with_counter, res_node, temp_classes_for_node):
        res_node.append(new_class_with_counter)
        new_class = new_class_with_counter[0]
        if new_class != None:
            temp_classes_for_node.append(new_class)
        return res_node, temp_classes_for_node

    def update_temp_founded_nodes(self, node, temp_classes_for_node, temp_founded_nodes):
        if temp_founded_nodes.has_key(node):
            temp_list = temp_founded_nodes.get(node)
            temp_list = temp_list + temp_classes_for_node
            temp_founded_nodes[node] = temp_list
        else:
            if len(temp_classes_for_node) != 0:
                temp_founded_nodes[node] = temp_classes_for_node
        return temp_founded_nodes

    def collect_results(self, node, res_node, results):
        if not results.has_key(node):
            results[node] = res_node
        else:
            results[node] = results[node] + res_node
        return results

    def init_visited_if_needed(self, current_layer, visited):
        if not visited.has_key(current_layer):
            visited[current_layer] = list()
        return visited

    def update_visited_before_transition(self, current_layer, edge, visited):
        visited_on_layer = visited.get(current_layer)
        visited_on_layer.append(edge)
        visited[current_layer] = visited_on_layer
        return visited

    def draw_connection_at_node(self, network, node, visited):
        result=None
        neighbs = [edge for edge in nx.edges_iter(network, node)]
        neighbs = filter(lambda x : x not in visited, neighbs)
        if len(neighbs)>0:
            result = random.choice(neighbs)
        return result

    def correct_layer_row(self, original_row, current_node, start_layer, visited):
        row = original_row.copy()
        for (i, visited_on_layer) in visited.items():
            visited_paths = filter(lambda edge: edge[0] == current_node, visited_on_layer)
            row[0, i - 1] = row[0, i - 1] - visited_paths.__len__()
        return row

    def decide_layer_change(self, current_node, start_layer, visited):
        layer_matrix = self.d[current_node]
        original_row = layer_matrix[start_layer - 1, :]
        row = self.correct_layer_row(original_row,current_node, start_layer, visited)
        row_np = self.row_to_numpy(row)
        indexes = np.argwhere(row_np == np.amax(row_np))
        indexes_list = indexes.flatten().tolist()
        return indexes_list[0] + 1 if len(indexes_list) == 1 else random.choice(indexes_list) + 1

    def row_to_numpy(self, row):
        row_np = np.ndarray(shape=row.shape, dtype=np.double)
        for i in xrange(0, row.shape[1]):
            row_np[0][i] = row[0, i]
        return row_np.flatten()

    def prepare_classification_results(self, results):
    #{u'1041': [('C', 2), ('C', 2)]}
    #selects most often appearing class
    #print results

        return dict(map(lambda x: (x[0], self.most_common_but_none([item[0] for item in x[1]])),results.iteritems()))

    def prepare_walk_length_results(self, results):
        #dlugosci wszystkich przejsc
        all_lenghts_list=list(map(lambda x: ([item[1] for item in x[1] if item[0]!=None]),results.iteritems()))
        return [val for list_item in all_lenghts_list for val in list_item]

    def most_common_but_none(self, lst):
        lst=filter(lambda a: a != None, lst)
        if len(lst)==0: lst.append(None)# None -> 0
        return max(lst, key=lst.count)


    def calculate_accuracy(self, net_original, results):
        #results dictionary {node_id: class_result}
        counter=0
        good=0
        areResults = False

        for item in results.iteritems():
            areResults = True
            if(item[0].label==item[1]):
                good+=1
            neighbs = [edge for edge in nx.edges_iter(net_original, item[0])]
            if(len(neighbs)>0):
                counter+=1

        return good/float(counter) if areResults else -1
