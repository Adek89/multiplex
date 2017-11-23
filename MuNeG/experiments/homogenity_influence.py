import sys
sys.path.append('D:\pycharm_workspace\multiplex\MuNeG')
from experiments.DecisionFusion import DecisionFusion
import sklearn.metrics as metrics
import csv
from graph.method.common.XValWithSampling import XValMethods

nodes = int(sys.argv[1])
size = int(sys.argv[2])
label = float(sys.argv[3])
probIn = float(sys.argv[4])
probBetween = int(sys.argv[5])
nrOfLayers = int(sys.argv[6])
probe = int(sys.argv[7])
keys = ["reduction", "fusion_sum", "fusion_mean", "fusion_layer", "fusion_random", "fusion_convergence_max", "fusion_convergence_min"]
for l in xrange(1, nrOfLayers+1):
    keys.append("L"+str(l))
for fold in [2]:
    aucs = {}
    df = DecisionFusion(nodes, size, label, probIn, probBetween, nrOfLayers, fold)
    df.generateSyntheticData()
    graph = df.synthetic
    edges = graph.edges(data=True)
    edges_between_different_labels = filter(lambda e : e[0].label <> e[1].label, edges)
    edges_in_layers = {}
    for l in xrange(1, nrOfLayers+1):
        edges_in_layers.update({l:[]})
    for e in edges_between_different_labels:
        data = e[2]
        layer = data['weight']
        edges_in_layers[layer].append(e)
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
        homogenity_distribution, known_neighbors_distribution, unknown_neighbors_distribution, node_degree_distribution, node_ids = df.calculate_distributions(df.synthetic, "reduction")
        avg_homogenity = float(sum(homogenity_distribution))/float(len(homogenity_distribution))
        with open("..\\results\\synthetic\\stats\\distributions_homogenity.csv",'ab') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([avg_homogenity, aucs_in_iteration])
        i = i + 1
        #change graph
        if len(layer_fully_changed.keys()) == nrOfLayers:
            stopCondition = False
        else:
            # trzebaby zaczac od wezlow z najwiekszym degree
            # sprawdzac czy nie tworzymy loopow
            for l in xrange(1, nrOfLayers+1):
                if len(edges_in_layers[l]) > 1:
                    e1 = edges_in_layers[l][0]
                    graph.remove_edge(*e1[:2])
                    del edges_in_layers[l][0]

                    list_with_left_labels = filter(lambda edge: edge[0].label == e1[0].label and edge[0].id <> e1[0].id, edges_in_layers[l])
                    empty_left_labels = len(list_with_left_labels) == 0
                    if empty_left_labels:
                        list_with_right_labels = filter(lambda edge: edge[1].label == e1[0].label and edge[1].id <> e1[0].id, edges_in_layers[l])
                        e2 = list_with_right_labels[0]
                    else:
                        e2 = list_with_left_labels[0]
                    graph.remove_edge(*e2[:2])
                    edges_in_layers[l].remove(e2)

                    data = e1[2]
                    if empty_left_labels:
                        graph.add_edge(e1[0], e2[1], weight=data['weight'], layer=data['layer'], conWeight=data['conWeight'])
                    else:
                        graph.add_edge(e1[0], e2[0], weight=data['weight'], layer=data['layer'], conWeight=data['conWeight'])
                    df.synthetic = graph
                else:
                    layer_fully_changed.update({l:True})







