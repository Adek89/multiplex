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
    pl.show(block=False)

def draw_boxplot(result_dict, folds):
    data = []
    for fold in folds:
        trace = go.Box(
        y=result_dict[fold],
        name=str(fold),
        boxmean=True
    )
        data.append(trace)
    plot_url = plotly.offline.plot(data, filename='box-plot.html')
    pass