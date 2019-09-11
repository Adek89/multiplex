import csv
import sys

import networkx as nx

sys.path.append('/home/apopiel/multiplex/MuNeG/')
from experiments.DecisionFusionGeneric import DecisionFusion

def node_stringizer(value):
    return value.__str__()

r = int(sys.argv[1])
direction = sys.argv[2]
reader = sys.argv[3]
df = DecisionFusion(2)
df.readRealData(reader)
graph = df.realGraph
edges = graph.edges(data=True)
edges_between_different_labels = filter(lambda e : e[0].label <> e[1].label, edges)
edges_between_same_labels = filter(lambda e: e[0].label == e[1].label, edges)
stopCondition = False
i = 0
while not stopCondition:
    homogenity_distribution, node_ids = df.calculate_homogenity(df.realGraph)
    avg_homogenity = float(sum(homogenity_distribution))/float(len(homogenity_distribution))
    i = i + 1
    nx.write_gml(graph, "/lustre/scratch/apopiel/real_" + reader.lower() + "/stats/temp_graphs/graph_" + str(r) + "_" + str(avg_homogenity) + ".gml", stringizer=node_stringizer)
    with open("/lustre/scratch/apopiel/real_" + reader.lower() + "/stats/temp_graphs/homogenity_steps_" + str(r) +".csv",'ab') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([avg_homogenity, nx.density(graph), r])
    print "Data for homogenity: " + str(avg_homogenity) + " has been written"
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