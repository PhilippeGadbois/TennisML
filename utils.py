import pyodbc as pdb
import pandas as pd


def sql_read(query):
    # Change Server to your server
    conn = pdb.connect(driver='{SQL Server}', server='DESKTOP-E5TG60B\SQLEXPRESS', Trusted_Connection=True)
    res = pd.read_sql(query, conn)
    conn.close()
    return res


def access_read(query):
    # Change DBQ to OnCourt.mdb PATH
    conn_str = (
        r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
        r'DBQ=C:\Program Files (x86)\OnCourt\OnCourt.mdb;'
        r'PWD=qKbE8lWacmYQsZ2'
    )
    conn = pdb.connect(conn_str)
    res = pd.read_sql(query, conn)
    conn.close()
    return res


def sql_write(df, table):
    conn = pdb.connect(driver='{SQL Server}', server='DESKTOP-E5TG60B\SQLEXPRESS', Trusted_Connection=True)
    df.to_sql(table, conn)
    conn.close()
