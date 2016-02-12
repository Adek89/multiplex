import networkx as nx

import graph.analyser.StatisticalAnalysis as sa
import graph.reader.DanioRerio.DanioRerioReader as drr


def connect_to_danio_rerio():
    return sa.connect_to_data("156.17.131.228", "DanioRerio", "apopiel", "alamakota123")


def draw_data():
    data_array = {'reduction': reduction_data, 'rwc_iter': rwc_iter_data}
    sa.draw_plot(data_array)

    data_array = {'reduction': reduction_data, 'fusion_sum' : fusion_sum_data}
    sa.draw_plot(data_array)

    data_array = {'reduction': reduction_data, 'fusion_mean' : fusion_mean_data}
    sa.draw_plot(data_array)

    data_array = {'reduction': reduction_data, 'rwc_sum' : rwc_sum_data}
    sa.draw_plot(data_array)

    data_array = {'reduction': reduction_data, 'rwc_mean': rwc_mean_data}
    sa.draw_plot(data_array)

    data_array = {'reduction': reduction_data, 'rwc_last' : rwc_last_data}
    sa.draw_plot(data_array)


def perform_statistical_analysis():
    sa.compare_distributions(reduction_data, rwc_iter_data)


def perform_basic_analysis():
    global reduction_data, rwc_iter_data, fusion_sum_data, fusion_mean_data, rwc_sum_data, rwc_mean_data, rwc_last_data
    reduction_data = sorted(results.ado_results[0])
    rwc_iter_data = sorted(results.ado_results[1])
    fusion_sum_data = sorted(results.ado_results[2])
    fusion_mean_data = sorted(results.ado_results[3])
    rwc_sum_data = sorted(results.ado_results[4])
    rwc_mean_data = sorted(results.ado_results[5])
    rwc_last_data = sorted(results.ado_results[6])
    draw_data()
    perform_statistical_analysis()


def build_dicts(key, value, result_dict):
    if result_dict.has_key(key):
        reduction_list = result_dict.get(key)
        reduction_list.append(value)
        result_dict[key] = reduction_list
    else:
        reduction_list = [value]
        result_dict[key] = reduction_list
    return result_dict


def folds_analysis():
    reduction_dict = {}
    rwc_iter_dict = {}
    for row in results:
        fold = row[0]
        reduction_dict = build_dicts(fold, row[1], reduction_dict)
        rwc_iter_dict = build_dicts(fold, row[2], rwc_iter_dict)
    experiment_folds = [2, 3, 4, 5, 10, 20]
    fold_names = {2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 10: 'Ten', 20: 'Twenty'}
    sa.draw_boxplot_for_folds(reduction_dict, experiment_folds, fold_names, 'fold_box_plot.html')
    sa.draw_boxplots_for_folds(reduction_dict, rwc_iter_dict, experiment_folds, fold_names, 'fold_box_plots.html')


def qty_analysis():
    reduction_dict = {}
    rwc_iter_dict = {}
    for row in results:
        qty = row[0]
        reduction_dict = build_dicts(qty, row[1], reduction_dict)
        rwc_iter_dict = build_dicts(qty, row[2], rwc_iter_dict)
    experiment_values = []
    names = {}
    for qty in reduction_dict.keys():
        experiment_values.append(qty)
        names.update({qty: qty})
    sa.draw_boxplot_for_folds(reduction_dict, experiment_values, names, 'qty_box_plot.html')
    sa.draw_boxplots_for_folds(reduction_dict, rwc_iter_dict, experiment_values, names, 'qty_box_plots.html')


def homogenity_analysis():
    reader = drr.DanioRerioReader()
    homogenity_dict = {}
    reduction_dict = {}
    rwc_iter_dict = {}
    for row in results:
        fun = row[0]
        if not homogenity_dict.has_key(fun):
            reader.graph = nx.MultiGraph()
            reader.nodes = dict([])
            reader.read(fun)
            homogenity_dict.update({fun: reader.calcuclate_homogenity()})
        reduction_dict = build_dicts(homogenity_dict[fun], row[1], reduction_dict)
        rwc_iter_dict = build_dicts(homogenity_dict[fun], row[2], rwc_iter_dict)
    x = []
    reduction_list = []
    rwc_iter_list = []
    for item in sorted(reduction_dict.items()):
        x.append(item[0])
        # uncomment if you one to check for which fun is gomogenity calculated
        # for key, value in homogenity_dict.iteritems():
        #     if value == item[0]:
        #         x.append(key)
        #         break
        reduction_list.append(float(sum(item[1])) / float(len(item[1])))

        values_for_homogenity = rwc_iter_dict[item[0]]
        rwc_iter_list.append(float(sum(values_for_homogenity)) / float(len(values_for_homogenity)))
    sa.draw_line_chart(x, reduction_list, rwc_iter_list, 'homogenity_line.html')


def layers_density_analysis():
    global graph
    reader_new = drr.DanioRerioReader()
    reader_new.graph = nx.MultiGraph()
    reader_new.read()
    graph = reader_new.graph
    density_static_analysis(graph)


def density_static_analysis(graph):
    edges_in_layers_for_node = {}
    for node in graph.nodes():
        edges = graph.edges(node, data=True)
        layers = [e[2]['layer'] for e in edges]
        unique_layers = len(set(layers))
        if edges_in_layers_for_node.has_key(unique_layers):
            unique_layers_count = edges_in_layers_for_node[unique_layers]
            edges_in_layers_for_node[unique_layers] = unique_layers_count + 1
        else:
            edges_in_layers_for_node.update({unique_layers: 1})
    sa.draw_bar(edges_in_layers_for_node, 'edges_in_layers.html')


if __name__ == "__main__":
    cursor = connect_to_danio_rerio()
    cursor.execute('select reduction, rwc_iter, fusion_sum, fusion_mean, rwc_sum, rwc_mean, rwc_last from DanioRerio.daniorerio.results')
    results = cursor.fetchall()
    perform_basic_analysis()

    cursor.execute('select fold, reduction, rwc_iter, fusion_sum, fusion_mean, rwc_sum, rwc_mean, rwc_last from DanioRerio.daniorerio.results')
    results = cursor.fetchall()
    folds_analysis()

    cursor.execute('select qty, reduction, rwc_iter, fusion_sum, fusion_mean, rwc_sum, rwc_mean, rwc_last'
                   ' from DanioRerio.daniorerio.results res join DanioRerio.daniorerio.functions fun '
                   'on res.fun = fun.fun')
    results = cursor.fetchall()
    qty_analysis()

    cursor.execute('select fun, reduction, rwc_iter, fusion_sum, fusion_mean, rwc_sum, rwc_mean, rwc_last'
                   ' from DanioRerio.daniorerio.results ')
    results = cursor.fetchall()
    homogenity_analysis()

    layers_density_analysis()

    cursor.execute('select fun, reduction, rwc_iter, fusion_sum, fusion_mean, rwc_sum, rwc_mean, rwc_last'
                   ' from DanioRerio.daniorerio.results ')
    results = cursor.fetchall()

    map_functions_to_average_ones = {}
    map_average_ones_to_list_reduction = {}
    map_average_ones_to_list_rwc = {}
    count = 1
    for result in results:
        global graph_fun
        print str(count)
        layers_for_node = []
        fun = result[0]
        reader_next = drr.DanioRerioReader()
        reader_next.graph = nx.MultiGraph()
        reader_next.nodes = dict([])
        reader_next.read(fun)
        graph_fun = reader_next.graph
        if not map_functions_to_average_ones.has_key(fun):
            filtered_nodes = filter(lambda n: n.label == 1, graph_fun.nodes())
            for node in filtered_nodes:
                edges = graph_fun.edges(node, data=True)
                layers = [e[2]['layer'] for e in edges]
                unique_layers = len(set(layers))
                layers_for_node.append(unique_layers)
            average_nr_of_layers = float(sum(layers_for_node)) / float(len(layers_for_node)) if len(filtered_nodes) > 0 else 0.0
        if not map_functions_to_average_ones.has_key(fun):
            map_functions_to_average_ones.update({fun : average_nr_of_layers})
        else:
            average_nr_of_layers = map_functions_to_average_ones[fun]
        if map_average_ones_to_list_reduction.has_key(average_nr_of_layers):
            current_list_reduction = map_average_ones_to_list_reduction[average_nr_of_layers]
            current_list_reduction.append(result[1])
            map_average_ones_to_list_reduction[average_nr_of_layers] = current_list_reduction
            current_list_rwc = map_average_ones_to_list_rwc[average_nr_of_layers]
            current_list_rwc.append(result[2])
            map_average_ones_to_list_rwc[average_nr_of_layers] = current_list_rwc
        else:
            map_average_ones_to_list_reduction.update({average_nr_of_layers : [result[1]]})
            map_average_ones_to_list_rwc.update({average_nr_of_layers : [result[2]]})
        count = count + 1
    x = []
    reduction_y = []
    rwc_y = []
    for item in sorted(map_average_ones_to_list_reduction.items()):
        x.append(item[0])
        reduction_y.append(float(sum(item[1])) / float(len(item[1])))

        rwc_values = map_average_ones_to_list_rwc[item[0]]
        rwc_y.append(float(sum(rwc_values)) / float(len(rwc_values)))
    sa.draw_line_chart(x, reduction_y, rwc_y, 'layer_nr_analysis.html')


