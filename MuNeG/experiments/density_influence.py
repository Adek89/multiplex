import random
import sys

import networkx as nx
import numpy as np

sys.path.append('D:\pycharm_workspace\multiplex\MuNeG')
sys.path.append('/home/apopiel/multiplex/MuNeG/')
from experiments.DecisionFusion import DecisionFusion
import sklearn.metrics as metrics
import csv
from graph.method.common.XValWithSampling import XValMethods

nodes = 100
size = 20
label = 6
probIn = 3
probBetween = 1
nrOfLayers = 5
fold = int(sys.argv[1])
rep = int(sys.argv[2])
direction = sys.argv[3]
keys = ["reduction", "fusion_sum", "fusion_mean", "fusion_layer", "fusion_random", "fusion_convergence_max", "fusion_convergence_min"]
for l in xrange(1, nrOfLayers+1):
    keys.append("L"+str(l))
aucs = {}
df = DecisionFusion(nodes, size, label, probIn, probBetween, nrOfLayers, fold)
df.generateSyntheticData()
graph = df.synthetic
edges = graph.edges(data=True)
edges_in_layers = {}
nr_of_edges_in_layers = []
nodes = graph.nodes()
tuples_of_nodes = []
for n in nodes:
    for n1 in nodes:
        if n <> n1:
            tuples_of_nodes.append((n, n1))
for l in xrange(1, nrOfLayers+1):
    edges_in_layer = filter(lambda e: e[2]['weight'] == l, edges)
    edges_in_layers.update({l:edges_in_layer})
    nr_of_edges_in_layers.append(len(edges_in_layer))
sum_of_edges = sum(nr_of_edges_in_layers)
p = [float(elem)/float(sum_of_edges) for elem in nr_of_edges_in_layers]

stopCondition = True
i = 0
layer_fully_changed = {}
xval = XValMethods(df.synthetic)
df.folds = xval.stratifies_x_val(df.synthetic.nodes(), df.NUMBER_OF_FOLDS)
while stopCondition:
    syntheticClassMat, syntheticNrOfClasses = df.nu.createClassMat(df.synthetic)
    df.syntheticClassMat = syntheticClassMat
    df.syntheticNrOfClasses = syntheticNrOfClasses
    df.flatLBP()
    df.multiLayerLBP()
    df.evaluation()
    aucs_in_iteration = []
    for key in keys:
        roc_auc = metrics.auc(df.fprs_per_method[key], df.tprs_per_method[key])
        aucs_in_iteration.append(roc_auc)
    aucs.update({i:aucs_in_iteration})
    homogenity_distribution, node_ids = df.calculate_homogenity(df.synthetic)
    avg_homogenity_before = float(sum(homogenity_distribution))/float(len(homogenity_distribution))
    with open("/lustre/scratch/apopiel/synthetic/stats/distributions_density_" + str(fold) + "_" + str(rep) + ".csv",'ab') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([nx.density(graph), aucs_in_iteration, fold, avg_homogenity_before, rep])
    i = i + 1

    choose_layer = True
    while choose_layer:
        layer_to_change = np.random.choice(np.arange(1, nrOfLayers+1), p=p)
        if not layer_fully_changed.has_key(layer_to_change) or not layer_fully_changed[layer_to_change]:
            choose_layer = False
    if direction == 'b':
        edges_in_layer = edges_in_layers[layer_to_change]
        id_to_remove = random.choice(xrange(0, len(edges_in_layer)))
        edge_to_remove = edges_in_layer[id_to_remove]
        del edges_in_layer[id_to_remove]
        edges_in_layers.update({layer_to_change: edges_in_layer})
        graph.remove_edge(*edge_to_remove[:2])
        df.synthetic = graph
        if len(edges_in_layer) == 0:
            layer_fully_changed.update({layer_to_change:True})
        if not layer_fully_changed.has_key(layer_to_change) or not layer_fully_changed[layer_to_change]:
            homogenity_distribution, node_ids = df.calculate_homogenity(df.synthetic)
            avg_homogenity_after = float(sum(homogenity_distribution))/float(len(homogenity_distribution))

            if (avg_homogenity_after > avg_homogenity_before):
                edges_between= filter(lambda e: e[0].label == e[1].label, edges_in_layer)
            else:
                edges_between= filter(lambda e: e[0].label <> e[1].label, edges_in_layer)
            id_to_remove = random.choice(xrange(0, len(edges_between)))
            edge_to_remove = edges_in_layer[id_to_remove]
            edges_in_layer.remove(edge_to_remove)
            edges_in_layers.update({layer_to_change: edges_in_layer})
            graph.remove_edge(*edge_to_remove[:2])

            df.synthetic = graph
            if len(edges_in_layer) == 0:
                layer_fully_changed.update({layer_to_change:True})
        if len(layer_fully_changed.keys()) == nrOfLayers:
            stopCondition = False
    else:
        edges_in_layer = edges_in_layers[layer_to_change]
        pairs_of_nodes_in_layer = [(e[0], e[1]) for e in edges_in_layer]
        pairs_of_nodes_in_layer.extend([(e[1], e[0]) for e in edges_in_layer])
        nodes_without_edges = filter(lambda pair : pair not in pairs_of_nodes_in_layer, tuples_of_nodes)
        if len(nodes_without_edges ) == 0:
            layer_fully_changed.update({layer_to_change:True})
        else:
            pair_to_add = random.choice(nodes_without_edges)

            graph.add_edge(pair_to_add[0], pair_to_add[1], weight=layer_to_change, layer='L'+str(layer_to_change), conWeight = 1)
            edge = filter(lambda e: (e[0] == pair_to_add[0] and e[1] == pair_to_add[1]) or (e[0] == pair_to_add[1] and e[1] == pair_to_add[0]) and e[2]['weight'] == layer_to_change, graph.edges(data=True))
            edge = edge[0]
            edges_in_layer.append(edge)
            edges_in_layers.update({layer_to_change:edges_in_layer})
            df.synthetic = graph

            if not layer_fully_changed.has_key(layer_to_change) or not layer_fully_changed[layer_to_change]:
                homogenity_distribution, node_ids = df.calculate_homogenity(df.synthetic)
                avg_homogenity_after = float(sum(homogenity_distribution))/float(len(homogenity_distribution))

                if (avg_homogenity_after > avg_homogenity_before):
                    nodes_without= filter(lambda pair: pair[0].label <> pair[1].label, nodes_without_edges)
                else:
                    nodes_without= filter(lambda pair: pair[0].label == pair[1].label, nodes_without_edges)
                if len(nodes_without) == 0:
                    layer_fully_changed.update({layer_to_change:True})
                else:
                    pair_to_add = random.choice(nodes_without)
                    graph.add_edge(pair_to_add[0], pair_to_add[1], weight=layer_to_change, layer='L'+str(layer_to_change), conWeight = 1)
                    edge =  filter(lambda e: (e[0] == pair_to_add[0] and e[1] == pair_to_add[1]) or (e[0] == pair_to_add[1] and e[1] == pair_to_add[0]) and e[2]['weight'] == layer_to_change, graph.edges(data=True))
                    edge = edge[0]
                    edges_in_layer.append(edge)
                    edges_in_layers.update({layer_to_change:edges_in_layer})
                    df.synthetic = graph
        if len(layer_fully_changed.keys()) == nrOfLayers:
            stopCondition = False


