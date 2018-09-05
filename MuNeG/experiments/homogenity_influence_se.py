import sys
sys.path.append('D:\pycharm_workspace\multiplex\MuNeG')
sys.path.append('/home/apopiel/multiplex/MuNeG/')

from experiments.DecisionFusionRealSocialEvolution import DecisionFusion
import sklearn.metrics as metrics
import csv
from graph.method.common.XValWithSampling import XValMethods
import networkx as nx

fold = int(sys.argv[1])
r = int(sys.argv[2])
direction = sys.argv[3]
nrOfLayers = 8
keys = ["reduction", "fusion_sum", "fusion_mean", "fusion_layer", "fusion_random", "fusion_convergence_max", "fusion_convergence_min"]
for l in xrange(1, nrOfLayers+1):
    keys.append("L"+str(l))
aucs = {}
df = DecisionFusion(1, fold)
df.readRealData("Democrat")
xval = XValMethods(df.realGraph)
df.folds = xval.stratifies_x_val(df.realGraph.nodes(), df.NUMBER_OF_FOLDS)
graph = df.realGraph
edges = graph.edges(data=True)
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
    realGraphClassMat, realNrOfClasses = df.nu.createClassMat(df.realGraph)
    df.realGraphClassMat = realGraphClassMat
    df.realNrOfClasses = realNrOfClasses
    df.flatLBP()
    df.multiLayerLBP()
    df.evaluation()
    aucs_in_iteration = []
    for key in keys:
        roc_auc = metrics.auc(df.fprs_per_method[key], df.tprs_per_method[key])
        aucs_in_iteration.append(roc_auc)
    aucs.update({i:aucs_in_iteration})
    homogenity_distribution, node_ids = df.calculate_homogenity(df.realGraph)
    avg_homogenity = float(sum(homogenity_distribution))/float(len(homogenity_distribution))
    with open("/lustre/scratch/apopiel/real_se/stats/distributions_homogenity_" + str(fold) + "_" + str(r) +".csv",'ab') as csvfile:
    # with open("D:\\pycharm_workspace\\multiplex\\MuNeG\\results\\real_se\\stats\\distributions_homogenity_" + str(fold) + "_" + str(r) +".csv",'ab') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([avg_homogenity, aucs_in_iteration, fold, nx.density(graph), r ])
    i = i + 1
    #change graph
    if len(layer_fully_changed.keys()) == nrOfLayers:
        stopCondition = False
    else:
        if direction == 'f':
            for l in xrange(1, nrOfLayers+1):
                if len(edges_in_layers[l]) > 1:
                    e1 = edges_in_layers[l][0]
                    graph.remove_edge(*e1[:2])
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
                    graph.remove_edge(*e2[:2])
                    edges_in_layers[l].remove(e2)

                    data = e1[2]
                    if empty_left_labels:
                        graph.add_edge(e1[0], e2[1], weight=data['weight'], layer=data['layer'], conWeight=data['conWeight'])
                    else:
                        graph.add_edge(e1[0], e2[0], weight=data['weight'], layer=data['layer'], conWeight=data['conWeight'])
                    df.realGraph = graph
                else:
                    layer_fully_changed.update({l:True})
        else:
            for l in xrange(1, nrOfLayers+1):
                if len(edges_in_layers[l]) > 1:
                    e1 = edges_in_layers[l][0]
                    graph.remove_edge(*e1[:2])
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
                    graph.remove_edge(*e2[:2])
                    edges_in_layers[l].remove(e2)

                    data = e1[2]
                    if empty_left_labels:
                        graph.add_edge(e1[0], e2[1], weight=data['weight'], layer=data['layer'], conWeight=data['conWeight'])
                    else:
                        graph.add_edge(e1[0], e2[0], weight=data['weight'], layer=data['layer'], conWeight=data['conWeight'])
                    df.realGraph = graph
                else:
                    layer_fully_changed.update({l:True})







