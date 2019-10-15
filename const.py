import pandas as pd

# SQL Server
SERVER = 'DESKTOP-E5TG60B\SQLEXPRESS'

# OnCourt.mdb file path
ONPATH = 'C:\Program Files (x86)\OnCourt\OnCourt.mdb;'

# OnCourt password
PWD = 'qKbE8lWacmYQsZ2'

# DSN
DSN = "mssql+pyodbc://SQLDSN"

# Discount Factor
DF = 0.8

# Fatigue Discount Factor
FDF = 0.75

# Surface Weighting (1- Hard, 2- Clay, 3-  Indoor, 4- Carpet, 5- Grass, 6- Acrylic)
SURFACE_WEIGHTING = pd.DataFrame({'Index': [1, 2, 3, 4, 5, 6],
                                  1: [1.0, 0.28, 0.35, 0.0, 0.24, 0.0],
                                  2: [0.28, 1.0, 0.31, 0.0, 0.14, 0.0],
                                  3: [0.35, 0.31, 1.0, 0.0, 0.25, 0.0],
                                  4: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                  5: [0.24, 0.14, 0.25, 0.0, 1.0, 0.0],
                                  6: [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}).set_index('Index')

# Uncertainty Threshold
UNC_T = 0.5

# Layers Size
L_SIZE = 64

# Betting Site
BETTING_SITE = 1