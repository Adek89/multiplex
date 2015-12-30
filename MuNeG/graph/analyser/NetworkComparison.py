__author__ = 'Adrian'
import numpy as np
import networkx as nx
import scipy

import matplotlib.pyplot as plt
from graph.analyser.GraphAnalyser import GraphAnalyser
from graph.gen.GraphGenerator import GraphGenerator
def generate_model_networks(num_nodes):
    networks=[]


    #erdos_renyi_graph n
    for p in np.arange(0.01, 0.99, 0.01):
        G=nx.erdos_renyi_graph(num_nodes,p)
        G.name='Erdos p='+str(p)+' nodes='+str(num_nodes)
        networks.append(G)

    #newman_watts_strogatz_graph
    for p in np.arange(0.01, 0.99, 0.01):
        for n in np.arange(5,np.min([int(num_nodes/2.0),150]), 20):
            G=nx.newman_watts_strogatz_graph(num_nodes,n,p)
            G.name='Newman Watts Strogatz p='+str(p)+' n='+str(n)+' nodes='+str(num_nodes)
            networks.append(G)

    #barabasi
    for n in np.arange(5,num_nodes, 5):#np.arange(5,np.min([int(num_nodes/2.0),150]), 20):
        G=nx.barabasi_albert_graph(num_nodes,n)
        G.name='Barabasi n='+str(n)+' nodes='+str(num_nodes)
        networks.append(G)

    return networks

def calculate_network_measures(net, analyser):
    deg=nx.degree_centrality(net)
    clust=[]

    if(net.is_multigraph()):
        net = analyser.flatGraph(net)

    if(nx.is_directed(net)):
        tmp_net=net.to_undirected()
        clust=nx.clustering(tmp_net)
    else:
        clust=nx.clustering(net)



    if(nx.is_directed(net)):
        tmp_net=net.to_undirected()
        paths=nx.shortest_path(tmp_net, source=None, target=None, weight=None)
    else:
        paths=nx.shortest_path(net, source=None, target=None, weight=None)

    lengths = [map(lambda a: len(a[1]), x[1].items()[1:]) for x in paths.items()]
    all_lengths=[]
    for a in lengths:
        all_lengths.extend(a)
    max_value=max(all_lengths)
    #all_lengths = [x / float(max_value) for x in all_lengths]

    return deg.values(),clust.values(),all_lengths

def compare_two_distributions(a,b):
    ks,pval=scipy.stats.ks_2samp(a,b)

    return ks

def compare_network_with_generics(G, analyser, generics):
    deg1, clust1, leng1 = calculate_network_measures(G, analyser)
    compared_deg, compared_clust, compared_leng= [],[],[]

    for net in generics:
        print net.name
        deg2, clust2, leng2=calculate_network_measures(net, analyser)
        compared_deg.append(compare_two_distributions(deg1,deg2))
        compared_clust.append(compare_two_distributions(clust1,clust2))
        compared_leng.append(compare_two_distributions(leng1,leng2))

    return compared_deg, compared_clust, compared_leng


def calculate_measures_and_plot(g, generics, params):
    global res_deg, res_clust, res_leng, x
    res_deg, res_clust, res_leng = compare_network_with_generics(g, analyser, generics)
    deg_fig = plt.figure()
    plt.plot(range(len(res_deg)), res_deg, 'r', )
    plt.draw()
    deg_fig.savefig('..\\..\\results\\deg_group_' + str(params[0]) + '_prob_in_' + str(params[1]) + '_prob_out_' + str(params[2]) +'.png')
    plt.close(deg_fig)

    clust_fig = plt.figure()
    plt.plot(range(len(res_clust)), res_clust, 'b')
    plt.draw()
    clust_fig.savefig('..\\..\\results\\clust_group_' + str(params[0]) + '_prob_in_' + str(params[1]) + '_prob_out_' + str(params[2]) +'.png')
    plt.close(clust_fig)

    path_fig = plt.figure()
    plt.plot(range(len(res_leng)), res_leng, 'g')
    plt.draw()
    path_fig.savefig('..\\..\\results\\path_group_' + str(params[0]) + '_prob_in_' + str(params[1]) + '_prob_out_' + str(params[2]) +'.png')
    plt.close(path_fig)

    sum_fig = plt.figure()
    plt.plot(range(len(res_deg)), [sum(x) for x in zip(res_deg, res_clust, res_leng)], 'p')
    plt.draw()
    sum_fig.savefig('..\\..\\results\\sum_group_' + str(params[0]) + '_prob_in_' + str(params[1]) + '_prob_out_' + str(params[2]) +'.png')
    plt.close(sum_fig)
    # plt.plot(range(len(res_deg)),res_deg,'r',range(len(res_clust)),res_clust,'b', range(len(res_leng)),res_leng,'g',range(len(res_deg)),[sum(x) for x in zip(res_deg, res_clust, res_leng)],'p')


if __name__ == '__main__':
    layer_list = [1, 2, 3]
    generics=generate_model_networks(100)
    for group in [2, 3, 5, 8, 10]:
        for prob_in in [1, 3, 5, 8, 9]:
            for prob_out in [1, 3, 5, 9]:
                params = [group, prob_in, prob_out]
                GrGen=GraphGenerator(100,group, layer_list, 5, prob_in, prob_out, ['L1','L2','L3'])
                G=GrGen.generate()

                analyser = GraphAnalyser(G)

                calculate_measures_and_plot(G, generics, params)
    # position = nx.spring_layout(G, iterations=10)
    # edges,colors = zip(*nx.get_edge_attributes(G,'layer').items())
    # zip(*nx.get_edge_attributes(G,'layer').items())


    # nx.draw_networkx_nodes(G, position, GrGen.reds, node_color = 'r')
    # nx.draw_networkx_nodes(G, position, GrGen.blues, node_color = 'b')
    # nx.draw_networkx_edges(G, position)
    #nx.draw_networkx_labels(Gen, position)
    # nx.draw(G,edgelist=edges,edge_color=colors,node_color=['r' if i.label==0 else 'b' for i in G.nodes()],width=3)
    # plt.show()