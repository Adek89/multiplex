import csv
import pickle
import sys

import networkx as nx

sys.path.append('/home/apopiel/multiplex/MuNeG/')
from experiments.DecisionFusionGeneric import DecisionFusion
from graph.method.common.XValWithSampling import XValMethods

def node_stringizer(value):
    return value.__str__()

#input data
r = int(sys.argv[1])
direction = sys.argv[2]
reader = sys.argv[3]
nrOfLayers = int(sys.argv[4])
#base objects
df = DecisionFusion(2)
df.readRealData(reader, True)
xval = XValMethods(df.realGraph)
folds = [2, 3, 4, 5, 10, 20]
if direction == 'f':#ensure that folds are generated only once for repetition
    for f in folds:
        df.folds = xval.stratifies_x_val(df.realGraph.nodes(), f)
        folds_file = open("/lustre/scratch/apopiel/real_" + reader.lower() + "/stats/temp_graphs/folds" + str(f) + "_" + str(r) + ".tmp", "w");
        pickle.dump(df.folds, folds_file)
edges = df.realGraph.edges(data=True)
edges_between_different_labels = filter(lambda e : e[0].label <> e[1].label, edges)
edges_between_same_labels = filter(lambda e: e[0].label == e[1].label, edges)
edges_in_layers = {}
for l in xrange(1, nrOfLayers+1):
    edges_in_layers.update({l:[]})
if direction == 'f':
    for e in edges_between_different_labels:
        data = e[2]
        layer = data['weight']
        edges_in_layers[layer].append(e)
else:
    for e in edges_between_same_labels:
        data = e[2]
        layer = data['weight']
        edges_in_layers[layer].append(e)
stopCondition = True
i = 0
layer_fully_changed = {}
while stopCondition:
    homogenity_distribution, node_ids = df.calculate_homogenity(df.realGraph)
    avg_homogenity = float(sum(homogenity_distribution))/float(len(homogenity_distribution))
    i = i + 1
    nx.write_gml(df.realGraph, "/lustre/scratch/apopiel/real_" + reader.lower() + "/stats/temp_graphs/graph_" + str(r) + "_" + str(avg_homogenity) + ".gml", stringizer=node_stringizer)
    with open("/home/apopiel/multiplex/MuNeG/results/real_" + reader.lower() + "/homogenity_steps_" + str(r) +".csv",'ab') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([avg_homogenity, nx.density(df.realGraph), r])
    print "Data for homogenity: " + str(avg_homogenity) + " has been written"
    i = i + 1
    #change graph
    if len(layer_fully_changed.keys()) == nrOfLayers:
        stopCondition = False
    else:
        if direction == 'f':
            for l in xrange(1, nrOfLayers+1):
                if len(edges_in_layers[l]) > 1:
                    e1 = edges_in_layers[l][0]
                    df.realGraph.remove_edge(*e1[:2])
                    del edges_in_layers[l][0]

                    list_with_left_labels = filter(lambda edge: edge[0].label == e1[0].label and edge[0].id <> e1[0].id, edges_in_layers[l])
                    empty_left_labels = len(list_with_left_labels) == 0
                    if empty_left_labels:
                        list_with_right_labels = filter(lambda edge: edge[1].label == e1[0].label and edge[1].id <> e1[0].id, edges_in_layers[l])
                        if len(list_with_right_labels) > 0:
                            e2 = list_with_right_labels[0]
                        else:
                            layer_fully_changed.update({l:True})
                            break
                    else:
                        e2 = list_with_left_labels[0]
                    df.realGraph.remove_edge(*e2[:2])
                    edges_in_layers[l].remove(e2)

                    data = e1[2]
                    if empty_left_labels:
                        df.realGraph.add_edge(e1[0], e2[1], weight=data['weight'], layer=data['layer'], conWeight=data['conWeight'])
                    else:
                        df.realGraph.add_edge(e1[0], e2[0], weight=data['weight'], layer=data['layer'], conWeight=data['conWeight'])
                    df.realGraph = df.realGraph
                else:
                    layer_fully_changed.update({l:True})
        else:
            for l in xrange(1, nrOfLayers+1):
                if len(edges_in_layers[l]) > 1:
                    e1 = edges_in_layers[l][0]
                    df.realGraph.remove_edge(*e1[:2])
                    del edges_in_layers[l][0]

                    list_with_left_labels = filter(lambda edge: edge[0].label <> e1[0].label and edge[0].id <> e1[0].id, edges_in_layers[l])
                    empty_left_labels = len(list_with_left_labels) == 0
                    if empty_left_labels:
                        list_with_right_labels = filter(lambda edge: edge[1].label <> e1[0].label and edge[1].id <> e1[0].id, edges_in_layers[l])
                        if len(list_with_right_labels) > 0:
                            e2 = list_with_right_labels[0]
                        else:
                            layer_fully_changed.update({l:True})
                            break
                    else:
                        e2 = list_with_left_labels[0]
                    df.realGraph.remove_edge(*e2[:2])
                    edges_in_layers[l].remove(e2)

                    data = e1[2]
                    if empty_left_labels:
                        df.realGraph.add_edge(e1[0], e2[1], weight=data['weight'], layer=data['layer'], conWeight=data['conWeight'])
                    else:
                        df.realGraph.add_edge(e1[0], e2[0], weight=data['weight'], layer=data['layer'], conWeight=data['conWeight'])
                    df.realGraph = df.realGraph
                else:
                    layer_fully_changed.update({l:True})

