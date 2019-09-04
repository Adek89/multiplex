import sys
sys.path.append('D:\\pycharm_workspace\\multiplex\\MuNeG\\')
sys.path.append('/home/apopiel/multiplex/MuNeG/')

from experiments.DecisionFusionGeneric import DecisionFusion
import sklearn.metrics as metrics
import csv
from graph.method.common.XValWithSampling import XValMethods
from graph.evaluation.EvaluationTools import EvaluationTools
import networkx as nx
import math

fold = int(sys.argv[1])
r = int(sys.argv[2])
direction = sys.argv[3]
reader = sys.argv[4]
aucs = {}
ev = EvaluationTools()
df = DecisionFusion(math.fabs(fold))
df.readRealData(reader)
xval = XValMethods(df.realGraph)
df.folds = xval.stratifies_x_val(df.realGraph.nodes(), fold)
graph = df.realGraph

edges = graph.edges(data=True)
edges_between_different_labels = filter(lambda e : e[0].label <> e[1].label, edges)
edges_between_same_labels = filter(lambda e: e[0].label == e[1].label, edges)

stopCondition = True
i = 0
while stopCondition:
    realGraphClassMat, realNrOfClasses = df.nu.createClassMat(df.realGraph)
    df.realGraphClassMat = realGraphClassMat
    df.realNrOfClasses = realNrOfClasses
    df.flatLBP()
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
    accuracy = ev.calculateAccuracy(df.realLabels, df.realFlatResult)
    with open("/lustre/scratch/apopiel/real_" + reader.lower() + "/stats/distributions_homogenity_" + str(fold) + "_" + str(r) +".csv",'ab') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([avg_homogenity, aucs_in_iteration, accuracy, fold, nx.density(graph), r ])
    i = i + 1
    #change graph
    if direction == 'f':
        if len(edges_between_different_labels) > 1:
            e1 = edges_between_different_labels[0]
            graph.remove_edge(*e1[:2])
            del edges_between_different_labels[0]

            list_with_left_labels = filter(lambda edge: edge[0].label == e1[0].label and edge[0].id <> e1[0].id, edges_between_different_labels)
            empty_left_labels = len(list_with_left_labels) == 0
            if empty_left_labels:
                list_with_right_labels = filter(lambda edge: edge[1].label == e1[0].label and edge[1].id <> e1[0].id, edges_between_different_labels)
                if len(list_with_right_labels) > 0:
                    e2 = list_with_right_labels[0]
                else:
                    stopCondition = True
                    break
            else:
                e2 = list_with_left_labels[0]
            graph.remove_edge(*e2[:2])
            edges_between_different_labels.remove(e2)

            data = e1[2]
            if empty_left_labels:
                graph.add_edge(e1[0], e2[1])
            else:
                graph.add_edge(e1[0], e2[0])
            df.realGraph = graph
        else:
            stopCondition = True
    else:
        if len(edges_between_same_labels) > 1:
            e1 = edges_between_same_labels[0]
            graph.remove_edge(*e1[:2])
            del edges_between_same_labels[0]

            list_with_left_labels = filter(lambda edge: edge[0].label <> e1[0].label and edge[0].id <> e1[0].id, edges_between_same_labels)
            empty_left_labels = len(list_with_left_labels) == 0
            if empty_left_labels:
                list_with_right_labels = filter(lambda edge: edge[1].label <> e1[0].label and edge[1].id <> e1[0].id, edges_between_same_labels)
                if len(list_with_right_labels) > 0:
                    e2 = list_with_right_labels[0]
                else:
                    stopCondition = True
                    break
            else:
                e2 = list_with_left_labels[0]
            graph.remove_edge(*e2[:2])
            edges_between_same_labels.remove(e2)

            data = e1[2]
            if empty_left_labels:
                graph.add_edge(e1[0], e2[1])
            else:
                graph.add_edge(e1[0], e2[0])
            df.realGraph = graph
        else:
            stopCondition = True