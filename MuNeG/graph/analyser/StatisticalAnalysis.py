import adodbapi as ado
import numpy as np
import plotly
import plotly.graph_objs as go
import pylab as pl
import scipy.stats as stats


def connect_to_data(host, database, user, password):
    conn_str = """Provider=SQLOLEDB.1; User ID=%(user)s; Password=%(password)s;"Database=%(database)s;Data Source=%(host)s"""
    my_conn = ado.connect(conn_str, user, password, host, database)
    curs = my_conn.cursor()
    return curs

def compare_distributions(a, b):
    ks,pval = stats.ks_2samp(a,b)
    print 'Kolgomorov-Smirnov: ' + str(ks) + ' p-value: ' + str(pval)

    stat, pval = stats.mannwhitneyu(b, a)
    print 'Mann-Whitney: ' + str(stat) + ' p-value: ' + str(pval)

    stat, pval = stats.ttest_ind(b, a)
    print '2-sided T-Test: ' + str(stat) + ' p-value: ' + str(pval)

def fit_to_distribution(label, data):
    print 'Stats for ' + label
    mean = np.mean(data)
    std = np.std(data)
    median = np.median(data)
    mode = stats.mode(data)
    max = np.max(data)
    stat, pval = stats.kstest(data, 'norm')
    k2, norm_pval = stats.mstats.normaltest(data)

    print 'Mean: ' + str(mean)
    print 'Std: ' + str(std)
    print 'Median: ' + str(median)
    print 'Mode: ' + str(mode)
    print 'Max: ' + str(max)
    print 'Goodness of fit to norm: ' + str(stat) + ' p-value: ' + str(pval)
    print 'DAgostino fit to norm: ' + str(k2) + ' p-value: ' + str(norm_pval)
    return stats.norm.pdf(data, mean, std)

def draw_plot(data_dict):
    for data in data_dict.iteritems():
        fit =  fit_to_distribution(data[0], data[1])
        pl.plot(data[1], fit, label=data[0])
    pl.legend(loc='upper right')
    pl.show(block=True)

def draw_boxplot_for_folds(result_dict, folds, names, file_name):
    data = []
    for fold in folds:
        trace = go.Box(
        y=result_dict[fold],
        name=names[fold],
        boxmean=True,
        boxpoints='all'
    )
        data.append(trace)
    plot_url = plotly.offline.plot(data, filename=file_name)

def draw_boxplots_for_folds(dict1, dict2, folds, names, file_name):
    x = []
    y1 = []
    y2 = []
    for fold in folds:
        for res1, res2 in zip(dict1[fold], dict2[fold]):
            x.append(names[fold])
            y1.append(res1)
            y2.append(res2)
    trace0 = go.Box(
    y=y1,
    x=x,
    name='reduction',
    marker=dict(
        color='#3D9970'
    ),
    boxpoints=False
    )
    trace1 = go.Box(
    y=y2,
    x=x,
    name='rwc_iter',
    marker=dict(
        color='#FF4136'
    ),
    boxpoints=False
    )
    data = [trace0, trace1]
    layout = go.Layout(
    yaxis=dict(
        title='two methods',
        zeroline=False
    ),
    boxmode='group'
    )
    fig = go.Figure(data=data, layout=layout)
    plot_url = plotly.offline.plot(fig, filename=file_name)

def draw_line_chart(data_x, data_1, data_2, file_name):
    # Create a trace
    trace0 = go.Scatter(
        x = data_x,
        y = data_1,
        mode='lines+markers',
        line=dict(
            shape='spline'
        )
    )
    trace1 = go.Scatter(
        x = data_x,
        y = data_2,
        mode='lines+markers',
        line=dict(
            shape='spline'
        )
    )

    data = [trace0, trace1]

    plotly.offline.plot(data, filename=file_name)

def draw_bar(data_dict, file_name):
    x = []
    y = []
    for el_x, el_y in data_dict.iteritems():
        x.append(el_x)
        y.append(el_y)
    data = [
    go.Bar(
        x=x,
        y=y
    )]
    plotly.offline.plot(data, filename=file_name)

