import adodbapi as ado

def connect_to_data(host, database, user, password):
    conn_str = """Provider=SQLOLEDB.1; User ID=%(user)s; Password=%(password)s;"Initial Catalog=(database)s;DataSource=%(host)s"""
    my_conn = ado.connect(conn_str, user, password, database, host)

