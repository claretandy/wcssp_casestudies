import pyodbc

def mdb_connect(db_file, user='admin', password = '', old_driver=False):
    driver_ver = '*.mdb'
    if not old_driver:
        driver_ver += ', *.accdb'

    odbc_conn_str = ('DRIVER={Microsoft Access Driver (%s)}'
                     ';DBQ=%s;UID=%s;PWD=%s' %
                     (driver_ver, db_file, user, password))

    return pyodbc.connect(odbc_conn_str)

conn = mdb_connect('/home/sarah/Documents/wcssp/wp3/PS_2019.accdb')  # only absolute paths!
print(conn)
print(type(conn))