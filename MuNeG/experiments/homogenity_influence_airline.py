import sys
sys.path.append('D:\pycharm_workspace\multiplex\MuNeG')
sys.path.append('/home/apopiel/multiplex/MuNeG/')

from experiments.DecisionFusionRealAirline import DecisionFusion
from graph.reader.Airline2016.Airline2016Node import Airline2016Node
import sklearn.metrics as metrics
import csv
from graph.method.common.XValWithSampling import XValMethods
import networkx as nx
import pickle

def airline_node_stringizer(value):
    return value.__str__()

def airline_node_destringizer(value):
    value_str = str(value)
    value_splitted = value_str.split()
    return Airline2016Node(int(value_splitted[0]),int(value_splitted[1]),value_splitted[2])

fold = int(sys.argv[1])
threshold = int(sys.argv[2])
class_label = sys.argv[3]
r = int(sys.argv[4])
direction = sys.argv[5]
restart = bool(int(sys.argv[6]))
nrOfLayers = 132
keys = ["reduction"]
# for l in xrange(1, nrOfLayers+1):
#     keys.append("L"+str(l))
aucs = {}
df = DecisionFusion(1, fold, threshold)
if not restart:
    df.readRealData(threshold, class_label)
    xval = XValMethods(df.realGraph)
    df.folds = xval.stratifies_x_val(df.realGraph.nodes(), df.NUMBER_OF_FOLDS)
    folds_file = open("D:\\pycharm_workspace\\multiplex\\MuNeG\\results\\real_airline\\stats\\temp_graphs\\folds" + str(fold) + "_" + str(r) + ".tmp", "w");
    pickle.dump(df.folds, folds_file)
    folds_file.close()
    graph = df.realGraph
else:
    avg_homogenity = float(sys.argv[7])
    graph = nx.read_gml("D:\\pycharm_workspace\\multiplex\\MuNeG\\results\\real_airline\\stats\\temp_graphs\\graph_" + str(fold) + "_" + str(r) + "_" + str(avg_homogenity) + ".gml", destringizer=airline_node_destringizer)
    graph_iter = graph.edges_iter(data=True)
    next_edge = next(graph_iter, None)
    while (next_edge <> None):
        layer_str_value = str(next_edge[2]["layer"])
        next_edge[2]["layer"] = layer_str_value
        next_edge = next(graph_iter, None)
    df.realGraph = graph
    xval = XValMethods(graph)
    folds_file = open("D:\\pycharm_workspace\\multiplex\\MuNeG\\results\\real_airline\\stats\\temp_graphs\\folds" + str(fold) + "_" + str(r) + ".tmp", "rb")
    df.folds = pickle.load(folds_file)
    folds_file.close()
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
if not restart:
    layer_fully_changed = {}
else:
    layer_fully_changed_file = open("D:\\pycharm_workspace\\multiplex\\MuNeG\\results\\real_airline\\stats\\temp_graphs\\layer_fully_changes_" + str(fold) + "_" + str(r) + "_" + str(avg_homogenity) + ".tmp", "rb")
    layer_fully_changed = pickle.load(layer_fully_changed_file)
    layer_fully_changed_file.close()
while stopCondition:
    realGraphClassMat, realNrOfClasses = df.nu.createClassMat(df.realGraph)
    df.realGraphClassMat = realGraphClassMat
    df.realNrOfClasses = realNrOfClasses
    df.flatLBP()
    # df.multiLayerLBP()
    df.evaluation()
    aucs_in_iteration = []
    roc_methods = [str(c_id) for c_id in xrange(0,realNrOfClasses)]
    roc_methods.append("micro")
    roc_methods.append("macro")
    for key in roc_methods:
        roc_auc = metrics.auc(df.fprs_per_method[key], df.tprs_per_method[key])
        aucs_in_iteration.append(roc_auc)
    aucs.update({i:aucs_in_iteration})
    homogenity_distribution, node_ids = df.calculate_homogenity(df.realGraph)
    avg_homogenity = float(sum(homogenity_distribution))/float(len(homogenity_distribution))
    # with open("/lustre/scratch/apopiel/real_sw/stats/distributions_homogenity_" + str(fold) + "_" + str(r) +".csv",'ab') as csvfile:
    with open("D:\\pycharm_workspace\\multiplex\\MuNeG\\results\\real_airline\\stats\\distributions_homogenity_" + str(fold) + "_" + str(r) +".csv",'ab') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([avg_homogenity, aucs_in_iteration, fold, nx.density(graph), r ])
    nx.write_gml(graph, "D:\\pycharm_workspace\\multiplex\\MuNeG\\results\\real_airline\\stats\\temp_graphs\\graph_" + str(fold) + "_" + str(r) + "_" + str(avg_homogenity) + ".gml", stringizer=airline_node_stringizer)
    # check how to dump files
    edges_in_layers_file = open("D:\\pycharm_workspace\\multiplex\\MuNeG\\results\\real_airline\\stats\\temp_graphs\\edges_in_layers_" + str(fold) + "_" + str(r) + "_" + str(avg_homogenity) + ".tmp", "w");
    layer_fully_changed_file = open("D:\\pycharm_workspace\\multiplex\\MuNeG\\results\\real_airline\\stats\\temp_graphs\\layer_fully_changes_" + str(fold) + "_" + str(r) + "_" + str(avg_homogenity) + ".tmp", "w")
    pickle.dump(edges_in_layers, edges_in_layers_file)
    pickle.dump(layer_fully_changed, layer_fully_changed_file)
    edges_in_layers_file.close()
    layer_fully_changed_file.close()
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





