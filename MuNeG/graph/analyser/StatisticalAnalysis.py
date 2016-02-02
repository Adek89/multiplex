import adodbapi as ado
import numpy as np
import pylab as pl
import scipy.stats as stats


def connect_to_data(host, database, user, password):
    conn_str = """Provider=SQLOLEDB.1; User ID=%(user)s; Password=%(password)s;"Database=%(database)s;Data Source=%(host)s"""
    my_conn = ado.connect(conn_str, user, password, host, database)
    curs = my_conn.cursor()
    return curs

def draw_histogram(data):
    fit = stats.norm.pdf(data, np.mean(data), np.std(data))
    pl.plot(data,fit,'-o')
    pl.hist(data, normed=True)
    pl.show()