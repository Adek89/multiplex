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
    sa.draw_boxplot_for_folds(reduction_dict, experiment_folds, fold_names)
    sa.draw_boxplots_for_folds(reduction_dict, rwc_iter_dict, experiment_folds, fold_names)


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
    sa.draw_boxplot_for_folds(reduction_dict, experiment_values, names)
    sa.draw_boxplots_for_folds(reduction_dict, rwc_iter_dict, experiment_values, names)


def homogenity_analysis():
    reader = drr.DanioRerioReader()
    homogenity_dict = {}
    reduction_dict = {}
    rwc_iter_dict = {}
    for row in results:
        fun = row[0]
        if not homogenity_dict.has_key(fun):
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
    sa.draw_line_chart(x, reduction_list, rwc_iter_list)


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
