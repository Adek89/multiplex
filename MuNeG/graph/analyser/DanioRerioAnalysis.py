import graph.analyser.StatisticalAnalysis as sa

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


if __name__ == "__main__":
    cursor = connect_to_danio_rerio()
    cursor.execute('select reduction, rwc_iter, fusion_sum, fusion_mean, rwc_sum, rwc_mean, rwc_last from DanioRerio.daniorerio.results')
    results = cursor.fetchall()
    reduction_data = sorted(results.ado_results[0])
    rwc_iter_data = sorted(results.ado_results[1])
    fusion_sum_data = sorted(results.ado_results[2])
    fusion_mean_data = sorted(results.ado_results[3])
    rwc_sum_data = sorted(results.ado_results[4])
    rwc_mean_data = sorted(results.ado_results[5])
    rwc_last_data = sorted(results.ado_results[6])

    draw_data()

    perform_statistical_analysis()
