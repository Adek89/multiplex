import csv
import math
import sys
import time

import graph.reader.syntethic.MuNeGGraphReader as reader


def build_file_name(nodes, avg_in_group, homogenity, prob_in, prob_out, layers):
    homogenity = homogenity if homogenity in [5.5, 6.5, 7.5, 8.5, 9.5] else int(math.floor(homogenity))
    prob_in = int(math.floor(prob_in))
    prob_out = prob_out if prob_out in [0.1, 0.5] else int(math.floor(prob_out))
    return 'muneg_' + str(nodes) + '_' + str(avg_in_group) + '_' + str(
            homogenity) + '_' + str(prob_in) + '_' + str(
            prob_out) + '_' + str(layers) + '.gml'

def generateSyntheticData(nodes, avg_in_group, homogenity, prob_in, prob_out, layers):
    start_time = time.time()
    synthetic = reader.read_from_gml('..\\..\\results', build_file_name(nodes, avg_in_group, homogenity, prob_in, prob_out, layers))
    print("---generation time: %s seconds ---" % str(time.time() - start_time))
    return synthetic


if __name__ == "__main__":
    nodes = int(sys.argv[1])
    avg_in_group = int(sys.argv[2])
    homogenity = float(sys.argv[3])
    prob_in = float(sys.argv[4])
    prob_out = float(sys.argv[5])
    layers = int(sys.argv[6])
    graph = generateSyntheticData(nodes, avg_in_group, homogenity, prob_in, prob_out, layers)

    layers_set = set([])
    nodes_set = set([])
    edges = graph.edges(data=True)
    edges = sorted(edges, key=lambda e: e[0].id)
    for edge in edges:
        source = edge[0]
        dest = edge[1]
        data = edge[2]
        with open('muneg.edges_layer_' + str(data['layer']),'ab') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ', lineterminator='\n')
            writer.writerow([source.id, dest.id, data['conWeight']])
        layers_set.add((data['weight'], data['layer']))
        nodes_set.add(source)
        nodes_set.add(dest)

    layers_set = sorted(layers_set, key=lambda e : e[0])
    with open('muneg_layers.txt','ab') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ', lineterminator='\n')
            writer.writerow(['layerID', 'layerLabel'])

    for el in layers_set:
        with open('muneg_layers.txt','ab') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ', lineterminator='\n')
            writer.writerow([el[0], el[1]])

    with open('muneg_layout.txt','ab') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ', lineterminator='\n')
            writer.writerow(['nodeID', 'nodeLabel'])

    for el in sorted(graph.nodes(), key=lambda n : n.id):
        with open('muneg_layout.txt','ab') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ', lineterminator='\n')
            writer.writerow([el.id, 'n'+str(el.id)])

