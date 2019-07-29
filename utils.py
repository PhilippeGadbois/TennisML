import pyodbc as pdb
import pandas as pd
import const
from sqlalchemy import create_engine


def sql_read(query):
    # Change Server to your server
    conn = pdb.connect(driver='{SQL Server}', server=const.SERVER, Trusted_Connection=True)
    res = pd.read_sql(query, conn)
    conn.close()
    return res


def access_read(query):
    # Change DBQ to OnCourt.mdb PATH
    conn_str = (
        r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
        r'DBQ=' + const.ONPATH +
        r'PWD=' + const.PWD
    )
    conn = pdb.connect(conn_str)
    res = pd.read_sql(query, conn)
    conn.close()
    return res

# You can create DSN in Control Panel > System and Security > Administrative Tools > ODBC Data Sources (64-bit)
# User DSN > Add... > SQL Server > Input Name to SQLDSN > Server is your server name (DESKTOP-E5TG60B\SQLEXPRESS in my case)
# Next > Next > Change the default database to: TennisML > Finish
def sql_write(df, table):
    engine = create_engine(const.DSN)
    df.to_sql(con=engine, name=table, schema='Stat', index=False, if_exists='replace')
